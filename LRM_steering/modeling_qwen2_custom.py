from __future__ import annotations
from typing import Optional, Tuple, Any

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
)
import os
target_layer = int(os.environ.get("TARGET_LAYER", "10"))
target_head = int(os.environ.get("TARGET_HEAD", "3"))
fire_threshold = float(os.environ.get("FIRE_THRESHOLD", "0.50"))
sub_alpha = float(os.environ.get("SUB_ALPHA", "0.15"))

# ---------------------------------------------------------------------
class CustomAttention(Qwen2Attention):
    """
    Wrapper that tolerates **three** call patterns:
    1. vLLM shape-tracing:  forward(hidden_states)                 ← 1 arg
    2. HF shape-tracing:    forward(hidden_states, None, …)        ← 2 args
    3. Real inference:      forward(hidden, rotary/pos, mask, …)   ← ≥2 args
    """

    def __init__(self, config, layer_idx: int | None = None):
        # vLLM may instantiate us with only (config,)
        super().__init__(config, layer_idx if layer_idx is not None else 0)

        # remember the *real* layer index if provided
        self.layer_idx: int = layer_idx if layer_idx is not None else -1

        # hyper-params from config
        self.tgt_layer = target_layer
        self.tgt_head = target_head
        self.tau = fire_threshold
        self.alpha = sub_alpha

        # run-time state (one μ per batch row)
        self.register_buffer(
            "mean_vec",
            torch.zeros((1, 1, config.hidden_size), dtype=torch.bfloat16),
            persistent=False,
        )
        self.max_prompt_len: int = 0
        self._initialized: bool = False

    # -----------------------------------------------------------------
    def _maybe_init_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        """Lazy init once per sequence batch."""
        if self._initialized:
            return
        bsz, h_size = hidden_states.shape[0], hidden_states.shape[-1]
        self.mean_vec = torch.zeros(
            bsz, 1, h_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if attention_mask is not None:
            # prompt = tokens where mask == 0 (HF convention)
            self.max_prompt_len = (attention_mask == 0).sum(-1).max().item()
        self._initialized = True

    # -----------------------------------------------------------------
    def forward(self, hidden_states, *args, **kwargs):
        """
        Accept any call signature, decide fast-path:
          • shape-tracing  → defer to parent, no hack
          • real run       → apply μ-subtraction + fire detection
        """

        # -------- 0) shape-tracing fast-exit --------------------------
        if len(args) == 0:
            # vLLM’s 1-arg trace
            return super().forward(hidden_states, *args, **kwargs)

        second_arg = args[0]
        if second_arg is None:
            # HF’s 2-arg trace (rotary_pos_emb is None)
            return super().forward(hidden_states, *args, **kwargs)

        # -------- 1) unpack common names ------------------------------
        # We don’t care whether the tensor is called rotary_pos_emb or
        # position_embeddings – pass through unchanged.
        pos_emb = second_arg
        attention_mask = args[1] if len(args) > 1 else kwargs.get("attention_mask")
        past_key_value = args[2] if len(args) > 2 else kwargs.get("past_key_value")
        # cache_position (vLLM) is in kwargs, keep it for super()

        self._maybe_init_state(hidden_states, attention_mask)

        # subtract α·μ from the *new* token only
        hidden_states[:, -1:] -= self.alpha * self.mean_vec

        # Ensure we get attention probs back
        kwargs["output_attentions"] = True

        # Call parent; preserve full positional list for signature match
        parent_outputs = super().forward(
            hidden_states,
            pos_emb,
            attention_mask,
            past_key_value,
            *args[3:],  # remaining positional args (cache_position, ...)
            **kwargs,
        )

        # Parent may return 2-tuple (vLLM) or 3-tuple (HF)
        if len(parent_outputs) == 2:
            attn_out, attn_weights = parent_outputs
            rest = ()  # nothing else
        else:
            attn_out, attn_weights, *rest = parent_outputs

        # -------- 2) our intervention logic ---------------------------
        if self.layer_idx == self.tgt_layer and attn_weights is not None:
            # attn_weights: (B, heads, Q, K)
            head_w = attn_weights[:, self.tgt_head, -1, :-1]   # (B, Kprev)
            fires = head_w.max(dim=-1).values > self.tau       # (B,)

            gen_slice = slice(self.max_prompt_len, attn_out.size(1) - 1)

            for b in range(hidden_states.size(0)):
                if fires[b] and gen_slice.start < gen_slice.stop:
                    mu = hidden_states[b, gen_slice].mean(dim=0, keepdim=True)
                    self.mean_vec[b:b + 1] = mu.detach()

        # -------- 3) return in same shape the parent used -------------
        return (attn_out, attn_weights, *rest) if rest else (attn_out, attn_weights)


# ---------------------------------------------------------------------
class CustomDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace self-attention with our custom one
        self.self_attn = CustomAttention(config, layer_idx)


# ---------------------------------------------------------------------
class Qwen2ForCausalLMCustom(Qwen2ForCausalLM):
    """
    Fully-drop-in replacement for Qwen2ForCausalLM.
    """
    def __init__(self, config):
        super().__init__(config)

        # swap every decoder layer
        for idx in range(len(self.model.layers)):
            self.model.layers[idx] = CustomDecoderLayer(config, idx)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overrides the standard forward method.
        
        The standard Qwen2ForCausalLM.forward computes logits. However, vLLM's
        engine expects the model's forward pass to return the hidden states,
        as it manages the logit computation itself.
        
        This override delegates the call to the underlying base model
        (self.model: Qwen2Model), which correctly returns hidden states,
        not logits.
        """
        return self.model(*args, **kwargs)
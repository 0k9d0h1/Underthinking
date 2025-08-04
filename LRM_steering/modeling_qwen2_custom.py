from __future__ import annotations
from typing import Optional, Tuple, Any, Generator
from collections import defaultdict

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
)


class FireRecorder:
    """A thread-safe-ish recorder for fire events."""
    def __init__(self):
        self.fired_seq_lengths = defaultdict(list)
        self.fire_counts = defaultdict(int)

    def record_fire(self, batch_idx: int, seq_len: int):
        """Records a fire event for a given sample in the batch."""
        self.fire_counts[batch_idx] += 1
        self.fired_seq_lengths[batch_idx].append(seq_len)

    def reset(self):
        """Clears all recorded data."""
        self.fired_seq_lengths.clear()
        self.fire_counts.clear()


# ---------------------------------------------------------------------
class CustomAttention(Qwen2Attention):
    """
    Wrapper that tolerates **three** call patterns:
    1. vLLM shape-tracing:  forward(hidden_states)                 ← 1 arg
    2. HF shape-tracing:    forward(hidden_states, None, …)        ← 2 args
    3. Real inference:      forward(hidden, rotary/pos, mask, …)   ← ≥2 args
    """

    def __init__(self, config, layer_idx: int | None = None, fire_recorder: FireRecorder = None):
        # vLLM may instantiate us with only (config,)
        super().__init__(config, layer_idx if layer_idx is not None else 0)

        # remember the *real* layer index if provided
        self.layer_idx: int = layer_idx if layer_idx is not None else -1

        # hyper-params from config
        self.tgt_layer = config.target_layer
        self.tgt_head = config.target_head
        self.tau = config.fire_threshold
        self.alpha = config.sub_alpha
        self.beta = 0.05
        
        self.fire_recorder = fire_recorder
        self.fire_recorder.reset()

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
        self.temp_mean_vec = hidden_states[:, -1, :]
        self._initialized = True

    # -----------------------------------------------------------------
    def forward(self, 
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position = None,
        **kwargs,
        ):
        """
        Accept any call signature, decide fast-path:
          • shape-tracing  → defer to parent, no hack
          • real run       → apply μ-subtraction + fire detection
        """
        self._maybe_init_state(hidden_states, attention_mask)
        self.temp_mean_vec = self.beta * hidden_states[:, -1, :] + (1 - self.beta) * self.temp_mean_vec

        # subtract α·μ from the *new* token only
        hidden_states[:, -1:, :] -= self.alpha * self.mean_vec

        # Ensure we get attention probs back
        kwargs["output_attentions"] = True

        # Call parent; preserve full positional list for signature match
        parent_outputs = super().forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value,
            cache_position,
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
            seq_len = past_key_value.get_seq_length()

            for b in range(hidden_states.size(0)):
                if fires[b]:
                    # Record the fire event with the current sequence length
                    self.fire_recorder.record_fire(b, seq_len)
                    self.mean_vec[b:b + 1] = self.temp_mean_vec[b:b + 1]

        # -------- 3) return in same shape the parent used -------------
        return (attn_out, attn_weights, *rest) if rest else (attn_out, attn_weights)


# ---------------------------------------------------------------------
class CustomDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int, fire_recorder: FireRecorder):
        super().__init__(config, layer_idx)
        # Replace self-attention with our custom one
        self.self_attn = CustomAttention(config, layer_idx, fire_recorder)


# ---------------------------------------------------------------------
class Qwen2ForCausalLMCustom(Qwen2ForCausalLM):
    """
    Fully-drop-in replacement for Qwen2ForCausalLM.
    """
    def __init__(self, config):
        super().__init__(config)
        self.fire_recorder = FireRecorder()

        # swap every decoder layer
        for idx in range(len(self.model.layers)):
            self.model.layers[idx] = CustomDecoderLayer(config, idx, self.fire_recorder)


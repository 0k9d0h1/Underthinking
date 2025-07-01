from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Qwen2DecoderLayer, Qwen2ForCausalLM
)


# ----------------------------------------------------------------------
class CustomAttention(Qwen2Attention):
    """Wrap the stock attention to watch a single (layer, head)."""
    def __init__(self, config, layer_id: int):
        super().__init__(config)
        self.layer_id = layer_id
        self.tgt_layer = config.target_layer
        self.tgt_head = config.target_head
        self.tau = config.fire_threshold
        self.alpha = config.sub_alpha

        self.register_buffer(
            "mean_vec",
            torch.zeros((1, 1, config.hidden_size), dtype=torch.bfloat16),
            persistent=False,
        )
        self.max_prompt_len = 0
        self._initialized = False

    # NOTE: we keep the original signature
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        if not self._initialized:
            bsz = hidden_states.shape[0]
            h_size = hidden_states.shape[-1]
            self.mean_vec = torch.zeros(bsz, 1, h_size, device=hidden_states.device, dtype=hidden_states.dtype)
            self.max_prompt_len = attention_mask.shape[-1]
            self._initialized = True
        hidden_states[:, -1:] -= self.alpha * self.mean_vec  # subtract mean from all tokens
        print(attention_mask)
        print(self.max_prompt_len)
        # ---- 1) normal attention
        attn_out, attn_weights, present_kv = super().forward(
            hidden_states,
            rotary_pos_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=True,          # we need the weights
            use_cache=use_cache,
        )

        # ---- 2) intervene only if this is the watched head/layer
        if self.layer_id == self.tgt_layer:
            # attn_weights: (B, heads, Q, K) – we only inspect current (last) query token
            head_w = attn_weights[:, self.tgt_head, -1, :-1]   # (B, Kprev)
            fires = head_w.max(dim=-1).values > self.tau    # bool tensor of shape (B,)
            gen_slice = slice(self.max_prompt_len, attn_out.size(1) - 1)
            for b in range(hidden_states.size(0)):
                if fires[b]:
                    mu = hidden_states[b, gen_slice].mean(dim=0, keepdim=True)
                    self.mean_vec[b:b+1] = mu.detach()

        return attn_out, attn_weights if output_attentions else None, present_kv


# ----------------------------------------------------------------------
class CustomDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_id: int):
        super().__init__(config)
        # replace attention with custom logic
        self.self_attn = CustomAttention(config, layer_id)


# ----------------------------------------------------------------------
class Qwen2ForCausalLMCustom(Qwen2ForCausalLM):
    """
    Drop-in replacement for `Qwen2ForCausalLM`.
    Put this file + __init__.py in a repo / local dir and load with
        AutoModel.from_pretrained(path, trust_remote_code=True)
    """
    def __init__(self, config):
        super().__init__(config)

        # swap every decoder layer
        for idx, layer in enumerate(self.model.layers):
            self.model.layers[idx] = CustomDecoderLayer(config, idx)

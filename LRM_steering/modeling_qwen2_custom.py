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
from transformers.cache_utils import Cache
import os


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

    def __init__(
        self, config, layer_idx: int | None = None, fire_recorder: FireRecorder = None
    ):
        # vLLM may instantiate us with only (config,)
        super().__init__(config, layer_idx if layer_idx is not None else 0)

        # remember the *real* layer index if provided
        self.layer_idx: int = layer_idx if layer_idx is not None else -1

        # hyper-params from config
        self.tgt_layer = config.target_layer
        self.alpha = config.sub_alpha
        self.fire_recorder = fire_recorder
        self.classifier = None

        # run-time state (one μ per batch row)
        self.mean_vec = torch.zeros(
            1,
            1,
            config.hidden_size,
            dtype=torch.bfloat16
        )
        self.max_prompt_len: int = 0
        self._initialized: bool = False

    def reset(self):
        """Resets the internal state of the attention module."""
        self._initialized = False
        self.prev_hidden_states = []
        self.approach_means = []

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
        self.mean_vec.copy_(torch.zeros(
            bsz,
            1,
            h_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ))
        self.prev_hidden_states = []
        self.approach_means = []
        self._initialized = True

    def _compute_mean_with_pc_removal(self, approach_means: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mean of the provided vectors after removing their first 
        principal component.
        
        Args:
            approach_means: A list of tensors, each representing a mean of hidden states.

        Returns:
            A tensor representing the new mean vector with the general component removed.
        """
        # 1. Stack the collected mean vectors into a single matrix (N, D)
        # where N is the number of means and D is the hidden size.
        # We squeeze to remove the batch dimension (assumed to be 1).
        approach_vectors = torch.stack(approach_means).squeeze(1)
        
        # PCA requires at least 2 vectors to find a direction of variance.
        if approach_vectors.shape[0] < 2:
            return torch.stack(approach_means).mean(dim=0)

        # 2. Center the data by subtracting the mean
        # It's crucial for PCA to operate on zero-mean data.
        mean_of_vectors = approach_vectors.mean(dim=0, keepdim=True)
        centered_vectors = approach_vectors - mean_of_vectors

        # 3. Use SVD (Singular Value Decomposition) to find the principal components.
        # SVD is a numerically stable way to perform PCA.
        # We only need the right singular vectors (V), which are the principal components.
        # Using float32 for better precision during SVD calculation.
        _, _, Vt = torch.linalg.svd(centered_vectors.to(torch.float32), full_matrices=False)
        
        # The first principal component (PC1) is the first row of Vt.
        pc1 = Vt[0, :].to(torch.bfloat16)

        # 4. Project the centered vectors onto the first principal component.
        # Projection = (vector @ pc1) * pc1
        projections_on_pc1 = torch.matmul(centered_vectors, pc1.unsqueeze(1)) * pc1.unsqueeze(0)
        
        # 5. Subtract the projections. The result is the original data with the
        # dominant "general" component removed. These are the "cleaned" vectors.
        cleaned_vectors = approach_vectors - projections_on_pc1
        
        # 6. Compute the new mean from these cleaned vectors.
        new_mean = cleaned_vectors.mean(dim=0)
        
        # Ensure the final mean has the original shape and dtype.
        # The final shape should be (1, 1, hidden_size) to match self.mean_vec
        return new_mean.unsqueeze(0).to(dtype=approach_means[0].dtype)

    # -----------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Accept any call signature, decide fast-path:
          • shape-tracing  → defer to parent, no hack
          • real run       → apply μ-subtraction + fire detection
        """
        self._maybe_init_state(hidden_states, attention_mask)
        self.prev_hidden_states.append(hidden_states[:, -1, :])

        # subtract α·μ from the *new* token only
        hidden_states[:, -1:] -= self.alpha * self.mean_vec.to(hidden_states.device)

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
        if self.layer_idx == self.tgt_layer:
            classifier_device = next(self.classifier.parameters()).device
            fires = (
                torch.sigmoid(
                    self.classifier(
                        hidden_states[:, -1, :].to(classifier_device).to(torch.float32)
                    )
                )
                > 0.5
            )
            seq_len = past_key_value.get_seq_length()
            if fires:
                prev_hidden_states_mean = torch.stack(self.prev_hidden_states).mean(
                    dim=0
                )
                self.approach_means.append(prev_hidden_states_mean)
                self.prev_hidden_states = []
                new_mean = self._compute_mean_with_pc_removal(self.approach_means)
                self.mean_vec.copy_(new_mean)
                self.fire_recorder.record_fire(
                    batch_idx=0,  # assuming batch size is 1 for simplicity
                    seq_len=seq_len,
                )

        # -------- 3) return in same shape the parent used -------------
        return (attn_out, attn_weights, *rest) if rest else (attn_out, attn_weights)


# ---------------------------------------------------------------------
class CustomDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int, fire_recorder: FireRecorder):
        super().__init__(config, layer_idx)
        # Replace self-attention with our custom one
        self.self_attn = CustomAttention(config, layer_idx, fire_recorder)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = self.mlp.gate_proj(hidden_states)
        up = self.mlp.up_proj(hidden_states)
        gate_act = self.mlp.act_fn(gate)
        gu_act = gate_act * up.to(gate_act.device)
        hidden_states = self.mlp.down_proj(gu_act)
        # hidden_states = self.mlp(hidden_states.to(device))
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

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

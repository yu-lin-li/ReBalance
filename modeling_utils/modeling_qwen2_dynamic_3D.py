# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2 model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B"
_CONFIG_FOR_DOC = "Qwen2Config"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2FlashAttention2(Qwen2Attention):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        steering_flag: Optional[torch.BoolTensor] = None,
        steering_vector: Optional[torch.FloatTensor] = None,
        steering_layer: Optional[int] = None,
        steering_coef: Optional[float] = 0.0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for l, decoder_layer in enumerate(self.layers):
            if steering_flag is not None and steering_layer == l:
                steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
                steering_flag = steering_flag.to(hidden_states.device)
                # support tensor (per-sample) or scalar coefficient
                if isinstance(steering_coef, torch.Tensor):
                    coef = steering_coef.to(hidden_states.dtype).to(hidden_states.device)[steering_flag].unsqueeze(1)
                else:
                    coef = torch.as_tensor(steering_coef, dtype=hidden_states.dtype, device=hidden_states.device)
                hidden_states[steering_flag, -1] += coef * steering_vector

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if steering_flag is not None and steering_layer == len(self.layers):
            steering_vector = steering_vector.to(hidden_states.dtype).to(hidden_states.device)
            steering_flag = steering_flag.to(hidden_states.device)
            if isinstance(steering_coef, torch.Tensor):
                coef = steering_coef.to(hidden_states.dtype).to(hidden_states.device)[steering_flag].unsqueeze(1)
            else:
                coef = torch.as_tensor(steering_coef, dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states[steering_flag, -1] += coef * steering_vector

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.new_round=False
        self.cur_steps = 0

        self.steering_flag = False
        self.steering_vector = None
        self.steering_layer = None
        self.steering_coef = 0.0
        self.steering_think_flag=None
        self._coefs = None
        self._step_prob_sum = None
        self._step_tok_count = None
        self._last_maxprob = None
        self._prev_step_mean = None  # for two-step variance


        self.steering_split_ids = None
        self.steering_think_start_id=None
        self.steering_think_end_id=None

        # --- dynamic steering internals (no extra CLI flags required)
        self._dyn_enabled = True
        self._coefs = None              # per-sample dynamic coefficients
        self._step_prob_sum = None      # sum of max-prob within a segment
        self._step_tok_count = None     # token count within a segment
        self._last_maxprob = None       # max-prob of the last token from previous step

        # Initialize weights and apply final processing
        self.post_init()

    def set_steering_flag(self, steering_flag, steering_layer=None, steer_vec=None,  steer_coef=0.0, tokenizer=None, dyn_hparams=None):
        self.steering_flag = steering_flag
        self.steering_vector = steer_vec
        self.steering_layer = steering_layer
        self.steering_coef = steer_coef
        self._dyn_hparams = dyn_hparams or {}
        self.steering_think_flag=None
        self.steering_split_ids = None
        self.steering_think_start_id=None
        self.steering_think_end_id=None
        if steering_flag:
            assert steering_layer is not None, "Steering layer must be provided for steering"
            assert steer_vec is not None, "Steering vector must be provided for steering"
            assert tokenizer is not None, "Tokenizer must be provided for steering"
            vocab = tokenizer.get_vocab()
            self.steering_split_ids = torch.LongTensor([vocab[token] for token in vocab.keys() if "ĊĊ" in token]).to(self.device)
            self.steering_think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
            self.steering_think_end_id =  tokenizer.encode("</think>", add_special_tokens=False)[0]
        # reset dynamic state
        self._coefs = None
        self._step_prob_sum = None
        self._step_tok_count = None
        self._last_maxprob = None

    
    def start_new_round(self):
        self.new_round=True
        self.cur_steps = 0
        self.steering_think_flag=None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.steering_flag:
            if self.new_round:
                self.new_round=False
                # self.steering_think_flag=torch.zeros(input_ids.shape[0], device=input_ids.device).to(torch.bool)
                self.steering_think_flag = (input_ids==self.steering_think_start_id).sum(1).to(torch.bool)
            else:
                assert input_ids.shape[1]==1, "use cache"
            last_tokens = input_ids[:,-1]
            self.steering_think_flag = torch.logical_or(self.steering_think_flag, last_tokens==self.steering_think_start_id)
            self.steering_think_flag = torch.logical_and(self.steering_think_flag, last_tokens!=self.steering_think_end_id)
            split_flag = torch.isin(last_tokens, self.steering_split_ids.to(input_ids.device))
            steering_flag = torch.logical_and(split_flag, self.steering_think_flag)
            if not torch.any(steering_flag):
                steering_flag = None
        else:
            steering_flag = None

        self.cur_steps += 1



        # --- dynamic coefficient tracking & update (confidence-based) ---
        if self.steering_flag and getattr(self, "_dyn_enabled", True):
            bsz = input_ids.shape[0]
            device = input_ids.device
            # init buffers
            if self._coefs is None or self._coefs.shape[0] != bsz:
                self._coefs = torch.full((bsz,), float(self.steering_coef), dtype=torch.float32, device=device)
            if self._step_prob_sum is None or self._step_prob_sum.shape[0] != bsz:
                self._step_prob_sum = torch.zeros(bsz, dtype=torch.float32, device=device)
                self._step_tok_count = torch.zeros(bsz, dtype=torch.long, device=device)

            # accumulate prev-step max-prob for non-split tokens
            if self._last_maxprob is not None:
                not_split_prev = ~split_flag
                self._step_prob_sum[not_split_prev] += self._last_maxprob[not_split_prev]
                self._step_tok_count[not_split_prev] += 1

            # on split boundary, update coef using mean of max-prob in the segment
            ready_mask = torch.logical_and(split_flag, self._step_tok_count > 0)
            if ready_mask.any():
                mean_max = self._step_prob_sum[ready_mask] / self._step_tok_count[ready_mask].float()
                # piecewise with smooth middle: <=0.70 -> -3.0; >=0.90 -> -0.1; else sigmoid around tau=0.8
                
                # === Dynamic steering: replace 1D F(c) with 2D F(c, v) under boundary constraints ===
                # 1) Segment-level mean confidence: mean_max (already computed).
                # 2) Two-step variance proxy: var_conf = ((cur - prev)**2)/4, written only for ready_mask.
                bsz = input_ids.shape[0]
                if (self._prev_step_mean is None) or (self._prev_step_mean.shape[0] != bsz):
                    self._prev_step_mean = torch.full(
                        (bsz,), float('nan'),
                        dtype=torch.float32, device=input_ids.device
                    )

                prev_vals = self._prev_step_mean[ready_mask]
                has_prev = torch.isfinite(prev_vals)
                var_conf = torch.zeros_like(mean_max, dtype=torch.float32, device=mean_max.device)
                var_conf[has_prev] = ((mean_max[has_prev] - prev_vals[has_prev]) ** 2) / 4.0

                # 3) Baseline 1D mapping F(c): controllable tanh-style fit (hyperparameters kept in place).
                import math as _m

                def _solve_k_for_tau(q25, q75, low_val, high_val, tau, iters=80):
                    m = 0.5 * (q25 + q75)
                    s = max(1e-9, 0.5 * (q75 - q25))
                    denom = max(high_val - low_val, 1e-12)
                    R_target = (2.0 * (tau - 0.5 * (low_val + high_val))) / denom
                    def ratio(k):
                        num = _m.tanh(k * max(0.0, 1.0 - m))
                        ks = k * s
                        den = _m.tanh(ks) if ks > 1e-12 else ks
                        return num / den if den > 0 else float('inf')
                    lo, hi = 1e-6, 1.0
                    while ratio(hi) > R_target and hi < 1e6:
                        hi *= 2.0
                    for _ in range(iters):
                        mid = 0.5 * (lo + hi)
                        if ratio(mid) > R_target:
                            lo = mid
                        else:
                            hi = mid
                    return 0.5 * (lo + hi)
                def build_F_linear(q25, q75, low_val=-3.0, high_val=0.0, tau=0.1):
                    # Midpoint/half-width definitions (aligned with the original parameterization).
                    m = 0.5 * (q25 + q75)
                    s = max(1e-9, 0.5 * (q75 - q25))  # Half-width of the linear interval, kept consistent with the original logic

                    # Affine re-parameterization (same as original; replace tanh span t with linear span s).
                    a = 0.5 * (low_val + high_val)
                    denom = max(high_val - low_val, 1e-12)
                    t_lin = s
                    b = (high_val - low_val) / (2.0 * max(t_lin, 1e-12))

                    # Target: make F(1) close to tau (linear-kernel normalized target).
                    # If R_target_lin ∈ [-1, 1], hit the target without saturation; otherwise saturate to ±1 (high/low).
                    R_target_lin = (2.0 * (tau - a)) / denom

                    # Linear kernel: scale (c - m) by k_lin, then clamp to [-s, s].
                    # Choose k_lin so that clamp(k_lin*(1-m), -s, +s)/s ≈ R_target_lin.
                    # When (1-m) is small or the target is out-of-range, this naturally saturates.
                    right_span = max(1e-12, 1.0 - m)
                    if R_target_lin >= 1.0:
                        k_lin = s / right_span  # Saturate directly to +1 ⇒ F(1)=high_val
                    elif R_target_lin <= -1.0:
                        # NOTE: driving the right endpoint to -1 is atypical, but allowed for interface robustness.
                        k_lin = s / right_span
                    else:
                        # In-range case: set k proportionally so that F(1)=tau without saturation.
                        k_lin = max(0.0, R_target_lin) * s / right_span
                        # If R_target_lin < 0, the right endpoint cannot become negative (it is clamped); keep a robust handling.
                        # Alternative: k_lin = abs(R_target_lin) * s / right_span.

                    def F1(c):
                        # Tensor handling and clamping (kept consistent with the surrounding code).
                        if not isinstance(c, torch.Tensor):
                            # Reuse dtype/device from the current context when possible; otherwise fall back to defaults.
                            try:
                                device = mean_max.device  # noqa: F821
                                dtype = mean_max.dtype    # noqa: F821
                            except Exception:
                                device, dtype = None, None
                            c = torch.as_tensor(c, dtype=dtype, device=device)
                        c = torch.nan_to_num(c, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                        a_t = torch.as_tensor(a, device=c.device, dtype=c.dtype)
                        b_t = torch.as_tensor(b, device=c.device, dtype=c.dtype)
                        k_t = torch.as_tensor(k_lin, device=c.device, dtype=c.dtype)
                        m_t = torch.as_tensor(m, device=c.device, dtype=c.dtype)
                        s_t = torch.as_tensor(s, device=c.device, dtype=c.dtype)

                        # Linear kernel: scale then saturate on [-s, s] (replacing tanh/normalization).
                        # g(c) = clamp(k_lin * (c - m), -s, +s)
                        g = torch.clamp(k_t * (c - m_t), min=-s_t, max=+s_t)

                        # Same structure: F = a + b*g, where b is normalized by (2*s) to preserve calibration.
                        # Ensures F(q25)=low_val and F(q75)=high_val; F(1) is controlled by k_lin and saturates to high_val if needed.
                        return a_t + b_t * g

                    return F1
                def build_F_hard(q25, q75, low_val=-3.0, high_val=0.0, tau=0.1):
                    # Hard step at midpoint m: c < m -> low_val; c >= m -> high_val.
                    m = 0.5 * (q25 + q75)

                    def F1(c):
                        # Tensor/device handling aligned with the surrounding code style.
                        if not isinstance(c, torch.Tensor):
                            try:
                                device = mean_max.device  # noqa: F821
                                dtype = mean_max.dtype    # noqa: F821
                            except Exception:
                                device, dtype = None, None
                            c = torch.as_tensor(c, dtype=dtype, device=device)

                        # Basic numeric robustness: clamp to [0, 1].
                        c = torch.nan_to_num(c, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                        m_t   = torch.as_tensor(m,        device=c.device, dtype=c.dtype)
                        low_t = torch.as_tensor(low_val,  device=c.device, dtype=c.dtype)
                        high_t= torch.as_tensor(high_val, device=c.device, dtype=c.dtype)

                        # Hard threshold at m; ties go to the right (high).
                        return torch.where(c < m_t, low_t, high_t)

                    return F1
                def build_F_poly(q25, q75, low_val=-3.0, high_val=0.0, tau=0.1):
                    # Polynomial smooth mapping: P(u) approximates tanh(u) with full polynomial differentiability (same interface).
                    import math as _m

                    # Midpoint and half-width (consistent with the original mapping).
                    m = 0.5 * (q25 + q75)
                    s = max(1e-9, 0.5 * (q75 - q25))

                    # Output affine transform: enforce F(q25)=low_val and F(q75)=high_val.
                    a = 0.5 * (low_val + high_val)
                    b = 0.5 * (high_val - low_val)

                    # Target: enforce F(1) ≈ tau by choosing w∈[0,1] with P(w) ≈ R_target.
                    denom = max(high_val - low_val, 1e-12)
                    R_target = (2.0 * (tau - a)) / denom  # Normalize to [-1,1]

                    # Define a 5th-order polynomial, normalized so P(±1)=±1.
                    def P_scalar(u: float) -> float:
                        # u in [-1, 1]
                        u3 = u * u * u
                        u5 = u3 * u * u
                        # Base polynomial: u - u^3/3 + 2u^5/15 (equals 4/5 at u=1), then normalized to 1.
                        base = u - (u3 / 3.0) + (2.0 * u5 / 15.0)
                        return (5.0 / 4.0) * base  # Normalize so that P(1)=1

                    # Invert P on [0,1]: for R∈[0,1], solve w via bisection so that P(w)=R.
                    def invert_P_on_0_1(R: float, iters: int = 50) -> float:
                        lo, hi = 0.0, 1.0
                        for _ in range(iters):
                            mid = 0.5 * (lo + hi)
                            if P_scalar(mid) < R:
                                lo = mid
                            else:
                                hi = mid
                        return 0.5 * (lo + hi)

                    # Compute k so that the right endpoint (c=1) reaches u_raw = k*(1-m)/s.
                    right_span = max(1e-12, 1.0 - m)
                    if R_target <= 0.0:
                        # If the right endpoint should be ≤ a, set u_raw→0 so F(1)≈a for stability.
                        k = 0.0
                    elif R_target >= 1.0:
                        # Saturate to +1 ⇒ F(1)=high_val.
                        k = s / right_span
                    else:
                        # Solve w in [0,1] with P(w)=R_target, then set k so u_raw(c=1)=w.
                        w = invert_P_on_0_1(R_target)
                        k = (w * s) / right_span

                    def F1(c):
                        # Align device and dtype with the current tensors.
                        if not isinstance(c, torch.Tensor):
                            try:
                                device = mean_max.device  # noqa: F821
                                dtype = mean_max.dtype    # noqa: F821
                            except Exception:
                                device, dtype = None, None
                            c = torch.as_tensor(c, dtype=dtype, device=device)

                        # Sanitize confidence values and clamp to the valid range.
                        c = torch.nan_to_num(c, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                        # Tensor form of P(u).
                        def P_tensor(u: torch.Tensor) -> torch.Tensor:
                            u3 = u * u * u
                            u5 = u3 * u * u
                            base = u - (u3 / 3.0) + (2.0 * u5 / 15.0)
                            return (5.0 / 4.0) * base

                        a_t = torch.as_tensor(a, device=c.device, dtype=c.dtype)
                        b_t = torch.as_tensor(b, device=c.device, dtype=c.dtype)
                        k_t = torch.as_tensor(k, device=c.device, dtype=c.dtype)
                        m_t = torch.as_tensor(m, device=c.device, dtype=c.dtype)
                        s_t = torch.as_tensor(s, device=c.device, dtype=c.dtype)

                        # Normalize and clamp: u = clamp(k*(c-m)/s, -1, 1).
                        u = torch.clamp(k_t * (c - m_t) / torch.maximum(s_t, torch.tensor(1e-12, device=c.device, dtype=c.dtype)),
                                        min=-1.0, max=1.0)

                        # Polynomial-smoothed output.
                        z = P_tensor(u)  # ∈ [-1,1]

                        return a_t + b_t * z

                    return F1
                def build_F_relu(q25, q75, low_val=-3.0, high_val=0.0, tau=0.1):
                    # ReLU variant: implement a symmetric clamp using ReLU primitives (same interface).

                    # Midpoint/half-width (same as the original logic).
                    m = 0.5 * (q25 + q75)
                    s = max(1e-9, 0.5 * (q75 - q25))

                    # Affine form: F = a + b*z with z ∈ [-1, 1].
                    a = 0.5 * (low_val + high_val)
                    denom = max(high_val - low_val, 1e-12)
                    b = 0.5 * denom  # = (high_val - low_val)/2

                    # Target: make F(1) ≈ tau.
                    # Normalized target R ∈ [-1, 1].
                    R_target = (2.0 * (tau - a)) / denom

                    # Right-span (1 - m).
                    right_span = max(1e-12, 1.0 - m)

                    # Choose k so that z(1) = clamp_sym(k*(1-m), s)/s.
                    # If R_target ≥ 1, saturate to high_val; if ≤ 0, stay near a.
                    if R_target >= 1.0:
                        k = s / right_span  # Saturate to +1
                    elif R_target <= 0.0:
                        k = 0.0             # No offset; F(1)≈a
                    else:
                        k = (R_target * s) / right_span  # Set proportionally in the normal range

                    def F1(c):
                        if not isinstance(c, torch.Tensor):
                            try:
                                device = mean_max.device  # noqa: F821
                                dtype = mean_max.dtype    # noqa: F821
                            except Exception:
                                device, dtype = None, None
                            c = torch.as_tensor(c, dtype=dtype, device=device)

                        # Sanitize then clamp to [0, 1].
                        c = torch.nan_to_num(c, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                        a_t = torch.as_tensor(a, device=c.device, dtype=c.dtype)
                        b_t = torch.as_tensor(b, device=c.device, dtype=c.dtype)
                        k_t = torch.as_tensor(k, device=c.device, dtype=c.dtype)
                        m_t = torch.as_tensor(m, device=c.device, dtype=c.dtype)
                        s_t = torch.as_tensor(s, device=c.device, dtype=c.dtype)

                        # Symmetric clamp implemented via ReLU.
                        # clamp_sym(x, s) = ReLU(x + s) - ReLU(x - s) - s
                        x = k_t * (c - m_t)
                        relu = torch.nn.functional.relu
                        clamp_sym = relu(x + s_t) - relu(x - s_t) - s_t  # ∈ [-s, s]

                        # Normalize to [-1, 1].
                        z = clamp_sym / torch.maximum(s_t, torch.tensor(1e-12, device=c.device, dtype=c.dtype))

                        return a_t + b_t * z

                    return F1

                def build_F(q25, q75, low_val=-3.0, high_val=0.0, tau=0.1):
                    m = 0.5 * (q25 + q75)
                    s = max(1e-9, 0.5 * (q75 - q25))
                    k = _solve_k_for_tau(q25, q75, low_val, high_val, tau)
                    t = _m.tanh(k * s)
                    a = 0.5 * (low_val + high_val)
                    b = (high_val - low_val) / (2.0 * max(t, 1e-12))
                    def F1(c):
                        if not isinstance(c, torch.Tensor):
                            c = torch.as_tensor(c, dtype=mean_max.dtype, device=mean_max.device)
                        c = torch.nan_to_num(c, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                        a_t = torch.as_tensor(a, device=c.device, dtype=c.dtype)
                        b_t = torch.as_tensor(b, device=c.device, dtype=c.dtype)
                        k_t = torch.as_tensor(k, device=c.device, dtype=c.dtype)
                        m_t = torch.as_tensor(m, device=c.device, dtype=c.dtype)
                        return a_t + b_t * torch.tanh(k_t * (c - m_t))
                    return F1

                # —— dynamic steering hyperparameters (configurable via CLI, except high_val_1 fixed) ——
                hp = getattr(self, "_dyn_hparams", None) or {}

                # confidence quantiles
                q25c = float(hp.get("q25c", 0.65))    # confidence q25
                q75c = float(hp.get("q75c", 0.90))    # confidence q75

                # targets for F(c)
                low_val_1 = float(hp.get("low_val_1", -1.00))  # F(q25c)
                high_val_1 = 0.01                               # F(1)  (FIXED; not exposed)

                # variance quantiles + 2D targets
                q25v = float(hp.get("q25v", 0.0005))          # variance q25
                q75v = float(hp.get("q75v", 0.01))          # variance q75
                low_val_2 = float(hp.get("low_val_2", -2.00))   # f(q25c, q75v)
                high_val_2 = float(hp.get("high_val_2", 0.1))   # f(1, q25v)

                # Safety: sort quantiles to avoid negative IQR if accidentally swapped.
                q25c, q75c = (min(q25c, q75c), max(q25c, q75c))
                q25v, q75v = (min(q25v, q75v), max(q25v, q75v))

                F1 = build_F(q25c, q75c, low_val=low_val_1, tau=high_val_1)  # baseline F(c)

                # 4) Build the 2D mapping F(c, v): monotone gating + normalized caps (never exceeds bounds).
                IQRc = max(1e-12, (q75c - q25c))
                IQRv = max(1e-12, (q75v - q25v))

                def _sigmoid(x: torch.Tensor) -> torch.Tensor:
                    return torch.sigmoid(x.clamp(-60.0, 60.0))

                # Tensor gate (batch form).
                def g_low_c_t(c):   return _sigmoid((q25c - c) / IQRc * 1200.0)   # Larger when c is lower
                def g_high_c_t(c):  return _sigmoid((c - q75c) / IQRc * 1200.0)   # Larger when c is higher
                def g_high_v_t(v):  return _sigmoid((v - q75v) / IQRv * 1200.0)   # Larger when v is higher
                def g_low_v_t(v):   return _sigmoid((q25v - v) / IQRv * 1200.0)   # Larger when v is lower

                # Scalar gate for calibration (same form as above), used to normalize weights.
                def _sigm_scalar(x: float) -> float:
                    x = max(min(x, 60.0), -60.0)
                    import math as __m
                    return 1.0 / (1.0 + __m.exp(-x))

                gp = _sigm_scalar((q25c - q25c) / IQRc * 12.0) * _sigm_scalar((q75v - q75v) / IQRv * 1200.0)  # ~= 0.25
                gm = _sigm_scalar((1.0  - q75c) / IQRc * 12.0) * _sigm_scalar((q25v - q25v) / IQRv * 1200.0)  # ~= 0.5
                eps = 1e-6

                # Baseline values at two calibration points (scalars).
                F_q25 = float(F1(torch.tensor(q25c, dtype=mean_max.dtype, device=mean_max.device)))
                F_1   = float(F1(torch.tensor(1.0,   dtype=mean_max.dtype, device=mean_max.device)))

                # Normalized, capped weights (∈[0,1]) to ensure outputs never cross low/high bounds.
                w_low  = (g_low_c_t(mean_max)  * g_high_v_t(var_conf)) / max(gp, eps)  # Low c × high v
                w_high = (g_high_c_t(mean_max) * g_low_v_t(var_conf))  / max(gm, eps)  # High c × low v
                w_low  = torch.clamp(w_low,  max=1.0)
                w_high = torch.clamp(w_high, max=1.0)

                # 5) Updated coefficient: baseline + (low-side weight)*(delta) + (high-side weight)*(delta).
                base = F1(mean_max)
                delta_low  = torch.as_tensor(low_val_2  - F_q25, dtype=mean_max.dtype, device=mean_max.device)
                delta_high = torch.as_tensor(high_val_2 - F_1,   dtype=mean_max.dtype, device=mean_max.device)
                updated = base + delta_low * w_low + delta_high * w_high

                # Final safety clamp to guarantee numeric bounds.
                updated = torch.clamp(updated, min=float(min(low_val_2, low_val_1)), max=float(max(high_val_1, high_val_2)))
                self._coefs[ready_mask] = updated

                # Reset per-segment accumulators for those samples.
                self._step_prob_sum[ready_mask] = 0.0
                self._step_tok_count[ready_mask] = 0

                # Update previous segment mean confidence (used to compute var_conf on the next boundary).
                self._prev_step_mean[ready_mask] = mean_max.detach()


            current_coef = self._coefs
        else:
            current_coef = torch.as_tensor(float(self.steering_coef), dtype=torch.float32, device=input_ids.device)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            steering_flag=steering_flag,
            steering_vector=self.steering_vector,
            steering_layer=self.steering_layer,
            steering_coef=current_coef,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])



        # cache last-step max probability for dynamic update on the next step
        if self.steering_flag and getattr(self, "_dyn_enabled", True):
            with torch.no_grad():
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                self._last_maxprob = probs.max(dim=-1).values
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

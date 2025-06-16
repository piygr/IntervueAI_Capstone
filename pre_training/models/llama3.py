import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pdb import set_trace 
from transformers import PretrainedConfig

class LLama3Config(PretrainedConfig):
    model_type = "llama3"

    def __init__(self,
                 vocab_size=32000,
                 hidden_size=1024,
                 intermediate_size=2816,
                 num_hidden_layers=12,
                 num_attention_heads=16,
                 num_key_value_heads=8,
                 hidden_act="silu",
                 max_position_embeddings=2048,
                 initializer_range=0.02,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 tie_word_embeddings=False,
                 rope_theta=10000.0,
                 rope_scaling=None,
                 use_parallel_attention=True,
                 use_swiglu=True,
                 use_rotary_embeddings=True,
                 use_grouped_query_attention=True,
                 output_attentions=False,
                 output_hidden_states=False,
                 use_return_dict=False,
                 **kwargs):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_parallel_attention = use_parallel_attention
        self.use_swiglu = use_swiglu
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_grouped_query_attention = use_grouped_query_attention
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        # self.use_return_dict = use_return_dict


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU activation function used in Llama3"""
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)

class LLama3Attention(nn.Module):
    """Multi-headed attention with grouped query attention"""
    def __init__(self, config: LLama3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # hidden_states (x)
        bsz, seq_len, _ = hidden_states.size() # (B, T, D)

        query_states = self.q_proj(hidden_states) # (B, T, D) -> (B, T, H * D/H)
        key_states = self.k_proj(hidden_states) # (B, T, D) -> (B, T, H_kv * D/H)
        value_states = self.v_proj(hidden_states) # (B, T, D) -> (B, T, H_kv * D/H)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, T, H * D/H) -> (B, T, H , D/H) -> (B, H, T, D/H)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (B, T, H_kv * D/H) -> (B, T, H_kv , D/H) -> (B, H_kv, T, D/H)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (B, T, H_kv * D/H) -> (B, T, H_kv , D/H) -> (B, H_kv, T, D/H)

        kv_seq_len = key_states.shape[-2] # should be same as seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = self._repeat_kv(key_states, self.num_key_value_groups) # (B, H_kv, T, D/H) -> (B, H, T, D/H)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups) # (B, H_kv, T, D/H) -> (B, H, T, D/H)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, H, T, D/H) * (B, H, D/H, T) -> (B, H, T, T)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]  # expand
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)  # float32
            attention_mask = (1.0 - attention_mask) * -10000.0  # large negative values
            attn_weights = attn_weights + attention_mask
            # attn_weights.masked_fill_(attention_mask==0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat k/v heads if n_kv_heads < n_heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LLama3MLP(nn.Module):
    """MLP module for Llama3 with SwiGLU activation"""
    def __init__(self, config: LLama3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LLama3DecoderLayer(nn.Module):
    """Decoder layer for Llama3"""
    def __init__(self, config: LLama3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLama3Attention(config=config)
        self.mlp = LLama3MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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

class LLama3Model(nn.Module):
    """Llama3 model with KV cache"""
    def __init__(self, config: LLama3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LLama3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache if use_cache else None,
            "hidden_states": all_hidden_states if output_hidden_states else None,
            "attentions": all_self_attns if output_attentions else None,
        }

class LLama3ForCausalLM(nn.Module):
    """Llama3 model for causal language modeling"""
    def __init__(self, config: LLama3Config):
        super().__init__()
        self.config = config
        self.model = LLama3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
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
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
        }

def create_llama3_1b() -> LLama3ForCausalLM:
    """Creates a 1B parameter version of Llama3"""
    config = LLama3Config(
        # vocab_size=32000,
        # hidden_size=1024,
        # intermediate_size=2816,
        # # num_hidden_layers=12,
        # # num_attention_heads=16,
        # # num_key_value_heads=16,
        # max_position_embeddings=2048,
    )
    return LLama3ForCausalLM(config)

def create_llama3_3b() -> LLama3ForCausalLM:
    """Creates a 3B parameter version of Llama3"""
    config = LLama3Config(
        vocab_size=32000,
        hidden_size=2048,  # Increased from 1024
        intermediate_size=5504,  # Increased from 2816
        num_hidden_layers=22,  # Increased from 12
        num_attention_heads=16,  # Kept same as 1B
        num_key_value_heads=16,  # Kept same as 1B
        max_position_embeddings=2048,
    )
    return LLama3ForCausalLM(config) 
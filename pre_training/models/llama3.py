import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pdb import set_trace 
import logging
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM, GenerationMixin
from transformers import AutoConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('llama3')

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
                 tie_word_embeddings=True,
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

    def forward(self, x: Optional[torch.Tensor] = None, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            if x is None:
                raise ValueError("Either x or seq_len must be provided")
            seq_len = x.shape[-2]
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
    logger.debug(f"Debug - q shape: {q.shape}, k shape: {k.shape}")
    logger.debug(f"Debug - cos shape: {cos.shape}, sin shape: {sin.shape}")
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
        if (bsz < 2):
            # set_trace()
            logger.debug(f"DEBUG - hidden_states.size() {hidden_states.size()}")

        query_states = self.q_proj(hidden_states) # (B, T, D) -> (B, T, H * D/H)
        key_states = self.k_proj(hidden_states) # (B, T, D) -> (B, T, H_kv * D/H)
        value_states = self.v_proj(hidden_states) # (B, T, D) -> (B, T, H_kv * D/H)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, T, H * D/H) -> (B, T, H , D/H) -> (B, H, T, D/H)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (B, T, H_kv * D/H) -> (B, T, H_kv , D/H) -> (B, H_kv, T, D/H)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (B, T, H_kv * D/H) -> (B, T, H_kv , D/H) -> (B, H_kv, T, D/H)

        kv_seq_len = key_states.shape[-2] # should be same as seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos, sin = self.rotary_emb(seq_len=kv_seq_len)
        logger.debug(f"Debug - query_states shape: {query_states.shape}, key_states shape: {key_states.shape}")
        logger.debug(f"Debug - cos shape: {cos.shape}, sin shape: {sin.shape}, kv_seq_len: {kv_seq_len}")
        # seq_len = query_states.shape[2]
        # cos = cos[:, :, :seq_len, :]
        # sin = sin[:, :, :seq_len, :]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = self._repeat_kv(key_states, self.num_key_value_groups) # (B, H_kv, T, D/H) -> (B, H, T, D/H)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups) # (B, H_kv, T, D/H) -> (B, H, T, D/H)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, H, T, D/H) * (B, H, D/H, T) -> (B, H, T, T)

        # Create causal mask if attention_mask is None
        # Create causal mask: lower triangular matrix
        causal_mask = torch.triu(torch.ones(seq_len, kv_seq_len, device=attn_weights.device), diagonal=1)
        causal_mask = causal_mask.bool()  # Convert to boolean mask
        # Expand to match attention weights shape (B, H, T, T)
        causal_mask = causal_mask[None, None, :, :].expand(bsz, self.num_heads, -1, -1)
        # Apply causal mask by setting masked positions to large negative values
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        if attention_mask is not None:
            # Handle provided attention_mask
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]  # expand to (B, 1, 1, T)
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)  # float32
            attention_mask = (1.0 - attention_mask) * -10000.0  # large negative values
            attn_weights = attn_weights + attention_mask

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
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

    def _create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask for autoregressive attention"""
        # Create lower triangular mask (including diagonal)
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask.bool()  # Convert to boolean mask
        return mask

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

        # Create positional embedding ids of size (batch_size, seq_length)
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device) # [0, 1, 2, ..., seq_length-1]
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1) # (B, seqLen)

        # set_trace()
        # Handle attention mask for causal language modeling
        if attention_mask is None:
            # For causal LM, we don't need to pass attention_mask to attention layers
            # The attention layers will create their own causal mask
            attention_mask = None
        else:
            # If attention_mask is provided, ensure it's properly formatted
            # attention_mask should be 1 for tokens to attend to, 0 for tokens to ignore
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) # (B, seqLen) -> (B, seqLen, Embed)

        # (B, seqLen, Embed)
        hidden_states = inputs_embeds
        # b_size, _, _ = hidden_states.size()
        # if (b_size < 2):
        #     set_trace()

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

# class LLama3ForCausalLM(nn.Module):
class LLama3ForCausalLM(PreTrainedModel, GenerationMixin):
    """Llama3 model for causal language modeling"""
    main_input_name = "input_ids"
    
    def __init__(self, config: LLama3Config):
        # super().__init__()
        super().__init__(config)
        self.config = config
        self.model = LLama3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # set_trace()

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

    @staticmethod
    def create_attention_mask(input_ids: torch.LongTensor, pad_token_id: int = 0) -> torch.Tensor:
        """
        Create attention mask for causal language modeling with padding.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            pad_token_id: ID of the padding token
            
        Returns:
            attention_mask: Mask of shape (batch_size, seq_length) where 1 indicates valid tokens, 0 indicates padding
        """
        # Create mask where 1 = valid token, 0 = padding token
        attention_mask = (input_ids != pad_token_id).long()
        return attention_mask

    @staticmethod
    def create_causal_attention_mask(seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal attention mask for autoregressive attention.
        
        Args:
            seq_length: Length of the sequence
            device: Device to create the mask on
            
        Returns:
            causal_mask: Lower triangular mask of shape (seq_length, seq_length)
        """
        # Create lower triangular mask (including diagonal)
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask.bool()  # Convert to boolean mask
        return mask

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
        # For causal language modeling, we need to ensure proper attention masking
        # If attention_mask is None, the model will use causal masking
        # If attention_mask is provided, it should be 1 for valid tokens, 0 for padding tokens
        
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
        
        # hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Create loss mask to ignore padding tokens in loss calculation
            # if attention_mask is not None:
            #     # Shift attention mask to align with shifted labels
            #     shift_mask = attention_mask[:, 1:].contiguous()
            #     # Flatten for loss calculation
            #     shift_logits_flat = shift_logits.view(-1, self.config.vocab_size)
            #     shift_labels_flat = shift_labels.view(-1)
            #     shift_mask_flat = shift_mask.view(-1)
                
            #     # Only compute loss on non-padded tokens
            #     loss_fct = nn.CrossEntropyLoss(reduction='none')
            #     loss_per_token = loss_fct(shift_logits_flat, shift_labels_flat)
            #     # Apply mask and take mean
            #     loss = (loss_per_token * shift_mask_flat).sum() / shift_mask_flat.sum()
            # else:
            #     # No padding mask provided, use standard loss
            #     if self.config.pad_token_id is not None:
            #         loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            #     else:
            #         loss_fct = nn.CrossEntropyLoss()
            #     loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            if self.config.pad_token_id is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            if (loss < 1):
                set_trace()

        # return {
        #     "loss": loss,
        #     "logits": logits,
        #     "past_key_values": outputs["past_key_values"],
        #     "hidden_states": outputs["hidden_states"],
        #     "attentions": outputs["attentions"],
        # }
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs["past_key_values"],
            hidden_states=outputs["hidden_states"],
            attentions=outputs["attentions"],
        )
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def resize_token_embeddings(self, new_num_tokens: int):
        self.model.embed_tokens = nn.Embedding(new_num_tokens, self.config.hidden_size)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
# Approx .5B params
llama3_config_1 = {
        "hidden_size":1024,
        "intermediate_size":2816,
        "num_hidden_layers":12,
        "num_attention_heads":16,
        "num_key_value_heads":8,
        # "max_position_embeddings":2048,
}

# approx 1B param - mostly official values
llama3_config_2 = {
        "hidden_size":1536,
        "intermediate_size":4096,
        "num_hidden_layers":22,
        "num_attention_heads":16,
        "num_key_value_heads":8,
        # "max_position_embeddings":2048,
}

# approx 1.6B param - mostly official values
llama3_config_3 = {
        "hidden_size":2048,
        "intermediate_size":5504,
        "num_hidden_layers":24,
        "num_attention_heads":16,
        "num_key_value_heads":8,
        # "max_position_embeddings":2048,
}

config_dict = {
    "0.5B": llama3_config_1,
    "1B": llama3_config_2,
    "1.5B": llama3_config_3
}

def create_llama3_1b(config_type, input_config:Optional[Dict] = {}) -> LLama3ForCausalLM:
    """Creates a 1B parameter version of Llama3"""
    # pad_token_id=0,
    # bos_token_id=1,
    # eos_token_id=2,
    config = LLama3Config(
        vocab_size=32000,
        # hidden_size=1024,
        # intermediate_size=2816,
        # num_hidden_layers=12,
        # hidden_size=2048,
        # intermediate_size=5504,
        # num_hidden_layers=24,
        # num_attention_heads=16,
        # num_key_value_heads=8,
        # max_position_embeddings=2048,
    )
    config.update(config_dict.get(config_type))
    config.update(input_config)
    _model = LLama3ForCausalLM(config)
    # AutoConfig.register("llama3", LLama3Config)
    # AutoModelForCausalLM.register(LLama3Config, LLama3ForCausalLM)
    return _model

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

def loadLlamaModelWithoutWeights(model_type, config_type, input_config:Optional[Dict] = {}):
    config = AutoConfig.from_pretrained(model_type)
    config.update(config_dict.get(config_type))
    config.update(input_config)
    return LlamaForCausalLM(config)  # No weights are loaded; this is from scratch

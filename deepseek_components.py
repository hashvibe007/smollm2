import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MLHAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        # Multi-Query Attention parameters
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Initialize rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.size()

        hidden_states = self.layer_norm(hidden_states)

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and apply rotary embeddings
        query_states = query_states.view(
            batch_size, seq_length, self.num_attention_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )

        query_states, key_states = self.rotary_emb(
            query_states, key_states, position_ids
        )

        # Repeat key/value states if num_key_value_heads < num_attention_heads
        if self.num_key_value_heads < self.num_attention_heads:
            key_states = key_states.repeat_interleave(
                self.num_attention_heads // self.num_key_value_heads, dim=2
            )
            value_states = value_states.repeat_interleave(
                self.num_attention_heads // self.num_key_value_heads, dim=2
            )

        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attention_probs = self.attention_dropout(attention_probs)

        output = torch.matmul(attention_probs, value_states)
        output = output.reshape(batch_size, seq_length, -1)
        output = self.o_proj(output)

        return output, attention_probs if output_attentions else None, past_key_value


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        router_jitter_noise: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_jitter_noise = router_jitter_noise

        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

        # Load balancing loss coefficient
        self.router_z_loss_coef = 0.001

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_length, hidden_size = hidden_states.shape

        # Get router logits and add noise for exploration
        router_logits = self.router(hidden_states)
        if self.training and self.router_jitter_noise > 0:
            router_logits += (
                torch.rand_like(router_logits) - 0.5
            ) * self.router_jitter_noise

        # Compute routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        routing_weights_max_k, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )

        # Normalize the routing weights
        routing_weights_max_k = routing_weights_max_k / routing_weights_max_k.sum(
            dim=-1, keepdim=True
        )

        # Compute load balancing loss
        # Ideal load would be uniform distribution across experts
        expert_load = routing_weights.mean(dim=(0, 1))
        target_load = torch.ones_like(expert_load) / self.num_experts
        load_balancing_loss = (
            F.mse_loss(expert_load, target_load) * self.router_z_loss_coef
        )

        # Process tokens through selected experts
        final_output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                final_output[expert_mask] += expert_output * routing_weights_max_k[
                    expert_mask
                ][..., expert_idx].unsqueeze(-1)

        return final_output, {"load_balancing_loss": load_balancing_loss}


class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.LongTensor] = None,
    ):
        if positions is None:
            positions = torch.arange(q.size(1), device=q.device)

        sincos = torch.stack(
            [
                torch.sin(positions[:, None] * self.inv_freq),
                torch.cos(positions[:, None] * self.inv_freq),
            ],
            dim=-1,
        ).expand(q.size(0), -1, -1, -1)

        sin, cos = sincos.chunk(2, dim=-1)

        # Rotate q and k
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
# 这里是vllm 的实现
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base) #[rotary_dim / 2]
        t = torch.arange(self.max_position_embeddings, dtype=torch.float) # [max_position_embeddings]

        freqs = torch.einsum("i,j -> ij", t, inv_freq) # [max_position_embeddings, rotary_dim / 2]
        print(freqs.shape)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1) # [max_position_embeddings, rotary_dim]
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size) #[num_tokens, num_heads, head_size]
        query_rot = query[..., :self.rotary_dim] #[num_tokens, num_heads, head_size]
        query_pass = query[..., self.rotary_dim:] # 空 （rotary_dim == head_size)
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

if __name__ == "__main__":
    # 以 12b mistral 为例， head_dim=128, num_attention_heads=32, kv_head=8
    head_dim = 4
    seq_len = 2
    num_attention_heads = 1
    kv_head = 1

    vllm_rotary_emb = RotaryEmbedding(
                        head_size=head_dim,
                        rotary_dim=head_dim,
                        max_position_embeddings=8192,
                        base=10000,
                        is_neox_style=False,
                        dtype=torch.bfloat16
                        ).to("cuda")

    batch_size = 1
    q_shape = (batch_size, num_attention_heads, seq_len, head_dim)
    k_shape = (batch_size, kv_head, seq_len, head_dim)
    query_states = torch.randn(*q_shape, device="cuda", dtype=torch.bfloat16)
    key_states = torch.randn(*k_shape, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(0, seq_len, dtype=torch.int64, device="cuda").unsqueeze(0) #（1, seq_len)

    query_states_1, key_states_1 = vllm_rotary_emb.forward(position_ids, query_states, key_states)



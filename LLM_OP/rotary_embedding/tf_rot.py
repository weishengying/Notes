import torch
import torch.nn as nn


# 这里是 tansformers 的实现
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2] # [bs, num_heads, seq_len, head_dim]
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
    print(f"cos shape: {cos.shape}, q shape: {q.shape}")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)) #[head_dim / 2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) #[bs, head_dim / 2, 1]
        position_ids_expanded = position_ids[:, None, :].float() # [bs, 1, seq_len]
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [bs, seq_len, head_dim / 2]]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() # [bs, seq_len, head_dim]
            sin = emb.sin() # [bs, seq_len, head_dim]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

if __name__ == "__main__":
    # 以 12b mistral 为例， head_dim=128, num_attention_heads=32, kv_head=8
    head_dim = 128
    seq_len = 2048
    num_attention_heads = 32
    kv_head = 8

    rotary_emb = MistralRotaryEmbedding(
                        head_dim,
                        max_position_embeddings=8192,
                        base=10000,
                        device="cuda"
                        )

    batch_size = 1
    q_shape = (batch_size, num_attention_heads, seq_len, head_dim)
    k_shape = (batch_size, kv_head, seq_len, head_dim)
    query_states = torch.randn(*q_shape, device="cuda", dtype=torch.bfloat16)
    key_states = torch.randn(*k_shape, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(0, seq_len, dtype=torch.int64, device="cuda").unsqueeze(0) #（1, seq_len)

    cos, sin = rotary_emb(query_states, position_ids)
    print(cos.shape)
    query_states_ref, key_states_ref = apply_rotary_pos_emb(query_states, key_states, cos, sin)

from dataclasses import dataclass
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x / rms) * self.gain.to(torch.float32)

        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # value branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ReLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


def compute_d_ff(d_model: int, multiple_of: int = 64) -> int:
    d_ff = int(8 * d_model / 3)
    return multiple_of * ((d_ff + multiple_of - 1) // multiple_of)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"

        k = torch.arange(0, d_head // 2, dtype=torch.float32)
        inv_freq = theta ** (-2.0 * k / d_head)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions]  # (seq_len, d_head/2)
        sin = self.sin[positions]  # (seq_len, d_head/2)

        x_even, x_odd = x[..., 0::2], x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        use_qk_norm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.context_length = context_length

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.use_rope = use_rope
        self.rope = rope if use_rope else None
        self.q_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()

        # In-layer KV cache buffers
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.cache_position: int = 0

    def reset_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Allocates key and value buffers matching the exact specified layer precision."""
        shape = (batch_size, self.n_heads, self.context_length, self.d_head)
        self.key_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.value_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.cache_position = 0

    def forward(
        self, x: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Positional indexing considering cached history length
        start_pos = self.cache_position if use_cache else 0
        positions = torch.arange(
            start_pos, start_pos + seq_len, device=x.device
        )

        # Projections -> (batch, n_heads, seq_len, d_head)
        q = (
            self.q_proj(x)
            .view(batch, seq_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, seq_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, seq_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        # QK Normalization & RoPE
        q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, positions), self.rope(k, positions)

        if use_cache:
            if self.key_cache is None or self.value_cache is None:
                raise RuntimeError(
                    "Cache buffers not initialized. Call model.reset_cache() first."
                )

            end_pos = start_pos + seq_len
            if end_pos > self.context_length:
                raise RuntimeError(
                    f"Exceeded context_length ({end_pos} > {self.context_length})."
                )

            # Insert newly computed keys and values into persistent buffers
            self.key_cache[:, :, start_pos:end_pos, :] = k
            self.value_cache[:, :, start_pos:end_pos, :] = v

            # Retrieve accumulated key/value sequence history
            k = self.key_cache[:, :, :end_pos, :]
            v = self.value_cache[:, :, :end_pos, :]

            self.cache_position = end_pos
            total_seq_len = end_pos
        else:
            total_seq_len = seq_len

        # Select masking strategy based on step type
        if use_cache:
            if seq_len == 1:
                # Single-token decode step: attend across past context
                is_causal = False
                attn_mask = None
            else:
                # Chunked/continuation prefill step: causal mask on current input relative to full context
                is_causal = False
                q_pos = torch.arange(
                    start_pos, start_pos + seq_len, device=x.device
                )[:, None]
                k_pos = torch.arange(total_seq_len, device=x.device)[None, :]
                attn_mask = k_pos <= q_pos
        else:
            # Uncached pass
            is_causal = True
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: "TransformerConfig", rope=None):
        super().__init__()

        if config.use_rmsnorm:
            self.attn_norm = RMSNorm(config.d_model)
            self.ffn_norm = RMSNorm(config.d_model)
        else:
            self.attn_norm = nn.Identity()
            self.ffn_norm = nn.Identity()

        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            context_length=config.context_length,
            rope=rope,
            use_qk_norm=config.use_qk_norm,
            use_rope=config.use_rope,
        )

        if config.ffn_type == "swiglu":
            self.ffn = SwiGLU(config.d_model, config.d_ff)
        elif config.ffn_type == "relu":
            d_ff_relu = int(config.d_ff * 1.5)
            self.ffn = ReLUFFN(config.d_model, d_ff_relu)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

    def forward(
        self, x: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), use_cache=use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


@dataclass
class TransformerConfig:
    vocab_size: int = 4000
    context_length: int = 256
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1344
    rope_theta: float = 10000.0
    use_qk_norm: bool = True
    use_rmsnorm: bool = True
    use_rope: bool = True
    ffn_type: str = "swiglu"


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.d_model
        )

        if config.use_rope:
            self.rope = RotaryPositionalEmbedding(
                d_head=config.d_model // config.n_heads,
                max_seq_len=config.context_length,
                theta=config.rope_theta,
            )
        else:
            self.rope = None

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, rope=self.rope)
                for _ in range(config.n_layers)
            ]
        )

        if config.use_rmsnorm:
            self.final_norm = RMSNorm(config.d_model)
        else:
            self.final_norm = nn.Identity()

        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            std = math.sqrt(2.0 / (module.in_features + module.out_features))
            nn.init.trunc_normal_(
                module.weight, std=std, a=-3 * std, b=3 * std
            )
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=1.0, a=-3.0, b=3.0)

    def reset_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initializes internal KV cache buffers matching the active model/weight precision."""
        if dtype is None:
            dtype = self.token_embeddings.weight.dtype

        for layer in self.layers:
            layer.attn.reset_cache(batch_size, device, dtype)

    def forward(
        self, token_ids: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, use_cache=use_cache)

        return self.lm_head(self.final_norm(x))

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= (
                self.token_embeddings.weight.numel()
                + self.lm_head.weight.numel()
            )
        return total
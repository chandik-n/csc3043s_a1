from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Step 1: root mean square over the LAST dimension, keeping the dimension for broadcasting
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Step 2: divide by the RMS and apply the learned gain
        out = (x / rms) * self.gain.to(torch.float32)

        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # value branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # projection back to d_model

    def forward(self, x):
        # SiLU(W1 x) is the gate; it multiplies W3 x element-wise; W2 projects back down.
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ReLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Matches SwiGLU input width, output width, and bias parameters for fair ablation
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


def compute_d_ff(d_model, multiple_of=64):
    d_ff = int(8 * d_model / 3)
    return multiple_of * ((d_ff + multiple_of - 1) // multiple_of)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_head, max_seq_len, theta=10000.0):
        super().__init__()
        assert (
            d_head % 2 == 0
        ), "RoPE rotates pairs of dimensions, so d_head must be even"

        k = torch.arange(0, d_head // 2, dtype=torch.float32)  # (d_head/2,)
        inv_freq = theta ** (-2.0 * k / d_head)  # theta_k
        positions = torch.arange(
            max_seq_len, dtype=torch.float32
        )  # (max_seq_len,)
        angles = torch.outer(
            positions, inv_freq
        )  # (max_seq_len, d_head/2)

        # Buffers, not parameters: these are fixed, and not saved into checkpoints.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions):
        # Step 1: look up the precomputed angles for these positions
        cos = self.cos[positions]  # (seq_len, d_head/2)
        sin = self.sin[positions]  # (seq_len, d_head/2)

        # Step 2: split the last dimension into the even and odd members of each pair
        x_even, x_odd = (
            x[..., 0::2],
            x[..., 1::2],
        )  # each (..., seq_len, d_head/2)

        # Step 3: apply the 2D rotation to every pair
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Step 4: interleave the pairs back into the original layout
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, d_model, n_heads, rope=None, use_qk_norm=True, use_rope=True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.use_rope = use_rope
        self.rope = rope if use_rope else None
        self.q_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()

    def forward(self, x):
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)

        # Step 1: project, then split into heads -> (batch, n_heads, seq_len, d_head)
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

        # Step 2: QK norm
        q, k = self.q_norm(q), self.k_norm(k)

        # Step 2b: Apply RoPE conditionally based on config
        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, positions), self.rope(k, positions)

        # Step 3: Fused PyTorch Scaled Dot-Product Attention with causal masking
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # (b, n_heads, seq_len, d_head)

        # Step 4: concatenate the heads and project back to the residual stream
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config, rope=None):
        super().__init__()

        # Conditionally set normalization layers
        if config.use_rmsnorm:
            self.attn_norm = RMSNorm(config.d_model)
            self.ffn_norm = RMSNorm(config.d_model)
        else:
            self.attn_norm = nn.Identity()
            self.ffn_norm = nn.Identity()

        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            rope=rope,
            use_qk_norm=config.use_qk_norm,
            use_rope=config.use_rope,
        )

        # Conditionally set FFN architecture based on config
        if config.ffn_type == "swiglu":
            self.ffn = SwiGLU(config.d_model, config.d_ff)
        elif config.ffn_type == "relu":
            # For exact total parameter matching with SwiGLU, expand d_ff by 1.5x
            # (SwiGLU has 3 matrices of size d_model x d_ff; ReLU has 2)
            d_ff_relu = int(config.d_ff * 1.5)
            self.ffn = ReLUFFN(config.d_model, d_ff_relu)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

    def forward(self, x):
        # Sub-layer 1: normalise, attend, add the residual.
        x = x + self.attn(self.attn_norm(x))
        # Sub-layer 2: normalise, feed-forward, add the residual.
        x = x + self.ffn(self.ffn_norm(x))
        return x


@dataclass
class TransformerConfig:
    vocab_size: int = 4000
    context_length: int = 256  # maximum sequence length
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1344  # ~ (8/3) * d_model, rounded to a multiple of 64
    rope_theta: float = 10000.0
    use_qk_norm: bool = True
    use_rmsnorm: bool = True
    use_rope: bool = True
    ffn_type: str = "swiglu"  # Fixed dataclass type annotation format


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.d_model
        )

        # Instantiated conditionally if use_rope is enabled
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

        # Conditionally set final norm
        if config.use_rmsnorm:
            self.final_norm = RMSNorm(config.d_model)
        else:
            self.final_norm = nn.Identity()

        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            std = math.sqrt(2.0 / (module.in_features + module.out_features))
            nn.init.trunc_normal_(
                module.weight, std=std, a=-3 * std, b=3 * std
            )
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        # Step 1: embed token IDs
        x = self.token_embeddings(token_ids)  # (batch, seq_len, d_model)

        # Step 2: run pre-norm blocks
        for layer in self.layers:
            x = layer(x)  # (batch, seq_len, d_model)

        # Step 3: final norm, project to vocabulary logits
        return self.lm_head(self.final_norm(x))  # (batch, seq_len, vocab_size)

    def num_parameters(self, non_embedding=False):
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= (
                self.token_embeddings.weight.numel()
                + self.lm_head.weight.numel()
            )
        return total
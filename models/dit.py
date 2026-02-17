"""
DiT (Diffusion Transformer) for VAE-DiT TTS.

Architecture:
  - Input projection: [x_t ∥ prompt_mask] → Linear → dit_dim
  - N × DiTBlock: AdaLN-Zero Self-Attn → Cross-Attn → AdaLN-Zero FFN
  - Output projection: AdaLN → Linear → latent_dim

References:
  - DiT: https://arxiv.org/abs/2212.09748
  - ACE-Step: cross-attention conditioning
  - F5-TTS: infilling-based prompt conditioning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Building Blocks
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to input tensor x of shape (B, H, T, D)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[-2]].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out


class SiLUGatedFFN(nn.Module):
    """SiLU-gated Feed-Forward Network (same as LLaMA MLP)."""

    def __init__(self, dim: int, mult: float = 2.5):
        super().__init__()
        hidden = int(dim * mult)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional cross-attention and RoPE."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int = 64,
        is_cross: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.is_cross = is_cross
        inner_dim = heads * head_dim

        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        kv_input = kv if self.is_cross and kv is not None else x

        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, -1, self.heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (only for self-attention)
        if rope_cos is not None and not self.is_cross:
            q = apply_rotary_emb(q, rope_cos, rope_sin)
            k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Attention with optional mask
        attn_mask = None
        if kv_mask is not None:
            # kv_mask: (B, T_kv) → (B, 1, 1, T_kv) for broadcast
            attn_mask = kv_mask.unsqueeze(1).unsqueeze(2).bool()
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))
            attn_mask = attn_mask.to(q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# =============================================================================
# Timestep Embedding
# =============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, freq_dim: int = 256, embed_dim: int = 1024, scale: float = 1000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.scale = scale
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0, 1]."""
        t_scaled = t * self.scale
        half_dim = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        ) # (freq_dim // 2)
        args = t_scaled.unsqueeze(-1) * freqs.unsqueeze(0) # (B, freq_dim // 2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)  # (B, embed_dim)


# =============================================================================
# DiT Block
# =============================================================================

class DiTBlock(nn.Module):
    """
    DiT Block with:
      1. AdaLN-Zero + Self-Attention (RoPE)
      2. Cross-Attention (text conditioning)
      3. AdaLN-Zero + FFN (SiLU-gated)
    """

    def __init__(self, dim: int, heads: int, head_dim: int = 64, ff_mult: float = 2.5):
        super().__init__()
        # AdaLN-Zero: 6 modulation values (shift, scale, gate) × 2 (self-attn, ffn)
        # self.scale_shift_table = nn.Parameter(
        #     torch.randn(1, 6, dim) / dim**0.5
        # )
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim))
        # Self-Attention
        self.self_attn_norm = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(dim, heads, head_dim, is_cross=False)

        # Cross-Attention
        self.cross_attn_norm = RMSNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, heads, head_dim, is_cross=True)

        # FeedForward
        self.ff_norm = RMSNorm(dim)
        self.ff = SiLUGatedFFN(dim, ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # AdaLN modulation parameters
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (
            self.scale_shift_table + time_emb.unsqueeze(1)
        ).chunk(6, dim=1)
        # Each: (B, 1, dim)

        # 1. Self-Attention with AdaLN-Zero
        h = self.self_attn_norm(x) * (1 + scale_sa) + shift_sa
        h = self.self_attn(h, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + h * gate_sa

        # 2. Cross-Attention (standard residual, no AdaLN gate)
        h = self.cross_attn_norm(x)
        h = self.cross_attn(h, kv=text_kv, kv_mask=text_mask)
        x = x + h

        # 3. FFN with AdaLN-Zero
        h = self.ff_norm(x) * (1 + scale_ff) + shift_ff
        h = self.ff(h)
        x = x + h * gate_ff

        return x


# =============================================================================
# DiT Model
# =============================================================================

class DiT(nn.Module):
    """
    Diffusion Transformer for VAE-DiT TTS.

    Input:
      - x_t:      (B, T_total, latent_dim)  noisy latent sequence
      - mask:     (B, T_total)              1=prompt (known), 0=to generate
      - timestep: (B,)                      diffusion timestep
      - text_kv:  (B, L_text, dit_dim)      text encoder output (projected)
      - text_mask:(B, L_text)               text attention mask

    Output:
      - velocity: (B, T_total, latent_dim)  predicted flow velocity
    """

    def __init__(
        self,
        latent_dim: int = 64,
        dit_dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        head_dim: int = 64,
        ff_mult: float = 2.5,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dit_dim = dit_dim
        self.depth = depth

        # Input projection: [x_t ∥ prompt_mask] → dit_dim
        self.proj_in = nn.Linear(latent_dim + 1, dit_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(256, dit_dim)

        # Rotary position embedding
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dit_dim, heads, head_dim, ff_mult)
            for _ in range(depth)
        ])

        # Output: AdaLN → Linear → latent_dim
        self.norm_out = RMSNorm(dit_dim)
        self.proj_out = nn.Linear(dit_dim, latent_dim)
        self.out_scale_shift = nn.Parameter(torch.zeros(1, 2, dit_dim))

        # Initialize output projection to near-zero (better training start)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x_t.shape

        # Concatenate prompt mask as extra channel
        if mask.dim() == 2:
            mask_input = mask.unsqueeze(-1)  # (B, T, 1)
        else:
            mask_input = mask
        x = self.proj_in(torch.cat([x_t, mask_input], dim=-1))  # (B, T, dit_dim)

        # Timestep embedding
        time_emb = self.time_embed(timestep)  # (B, dit_dim)

        # RoPE
        rope_cos, rope_sin = self.rotary_emb(T, x.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, time_emb, text_kv, text_mask, rope_cos, rope_sin)

        # Output projection with AdaLN
        shift, scale = (self.out_scale_shift + time_emb.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale) + shift
        velocity = self.proj_out(x)  # (B, T, latent_dim)

        return velocity

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

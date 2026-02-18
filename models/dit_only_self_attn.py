"""
DiT (Self-Attention Only) for VAE-DiT TTS.

Variant of DiT that removes cross-attention entirely.
Instead, text embeddings (from mT5) are repeat-padded to match audio frame
length T, then concatenated on the dim axis with the audio features.
The combined representation goes through self-attention only.

Architecture:
  - text_kv repeat-padded to (B, T, text_dim) → text_proj → (B, T, dit_dim)
  - Input projection: [x_t ∥ prompt_mask] → Linear → dit_dim
  - audio_feat + text_feat (element-wise add)
  - N × DiTBlock: AdaLN-Zero Self-Attn → AdaLN-Zero FFN (no cross-attn)
  - Output projection: AdaLN → Linear → latent_dim
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional

# Reuse building blocks from the original dit.py
from models.dit import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_emb,
    SiLUGatedFFN,
    TimestepEmbedding,
    MultiHeadAttention,
)


# =============================================================================
# DiT Block (Self-Attention Only, no Cross-Attention)
# =============================================================================

class DiTBlockSelfAttnOnly(nn.Module):
    """
    DiT Block with:
      1. AdaLN-Zero + Self-Attention (RoPE)
      2. AdaLN-Zero + FFN (SiLU-gated)
    No cross-attention — text info is already fused into the input.
    """

    def __init__(self, dim: int, heads: int, head_dim: int = 64, ff_mult: float = 2.5):
        super().__init__()
        # AdaLN-Zero: 6 modulation values
        #   self-attn: shift_sa, scale_sa, gate_sa
        #   ffn:       shift_ff, scale_ff, gate_ff
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim))

        # Self-Attention
        self.self_attn_norm = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(dim, heads, head_dim, is_cross=False)

        # FeedForward
        self.ff_norm = RMSNorm(dim)
        self.ff = SiLUGatedFFN(dim, ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # AdaLN modulation parameters (6 values)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (
            self.scale_shift_table + time_emb.unsqueeze(1)
        ).chunk(6, dim=1)
        # Each: (B, 1, dim)

        # 1. Self-Attention with AdaLN-Zero
        h = self.self_attn_norm(x) * (1 + scale_sa) + shift_sa
        h = self.self_attn(h, rope_cos=rope_cos, rope_sin=rope_sin, kv_mask=padding_mask)
        x = x + h * gate_sa

        # 2. FFN with AdaLN-Zero
        h = self.ff_norm(x) * (1 + scale_ff) + shift_ff
        h = self.ff(h)
        x = x + h * gate_ff

        return x


# =============================================================================
# DiT Model (Self-Attention Only)
# =============================================================================

class DiTSelfAttnOnly(nn.Module):
    """
    Diffusion Transformer (self-attention only variant) for VAE-DiT TTS.

    Text conditioning is done by:
      1. Repeat-expanding text_kv from (B, L, text_dim) to (B, T, text_dim)
      2. Projecting to dit_dim
      3. Adding to audio features (element-wise)

    Input:
      - x_t:      (B, T, latent_dim)    noisy latent sequence
      - mask:     (B, T)                1=prompt (known), 0=to generate
      - timestep: (B,)                  diffusion timestep
      - text_kv:  (B, L, text_dim)      text encoder output (e.g. mT5)
      - text_mask:(B, L)                text attention mask (1=valid, 0=pad)

    Output:
      - velocity: (B, T, latent_dim)    predicted flow velocity
    """

    def __init__(
        self,
        latent_dim: int = 64,
        dit_dim: int = 1024,
        text_dim: int = 1024,  # mT5-large output dim
        depth: int = 22,
        heads: int = 16,
        head_dim: int = 64,
        ff_mult: float = 2.5,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dit_dim = dit_dim
        self.text_dim = text_dim
        self.depth = depth
        self.gradient_checkpointing = False

        # Input projection: [x_t ∥ prompt_mask] → dit_dim
        self.proj_in = nn.Linear(latent_dim + 1, dit_dim)

        # Text projection: text_dim → dit_dim (project mT5 output to dit space)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, dit_dim),
            nn.SiLU(),
            nn.Linear(dit_dim, dit_dim),
        )

        # Timestep embedding
        self.time_embed = TimestepEmbedding(256, dit_dim)

        # Rotary position embedding
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len)

        # Transformer blocks (self-attention only)
        self.blocks = nn.ModuleList([
            DiTBlockSelfAttnOnly(dit_dim, heads, head_dim, ff_mult)
            for _ in range(depth)
        ])

        # Output: AdaLN → Linear → latent_dim
        self.norm_out = RMSNorm(dit_dim)
        self.proj_out = nn.Linear(dit_dim, latent_dim)
        self.out_scale_shift = nn.Parameter(torch.zeros(1, 2, dit_dim))

        # Initialize output projection to near-zero
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    @staticmethod
    def expand_text_to_frames(
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """
        Expand text tokens to frame length via nearest-neighbor repeat.

        Each valid token is repeated floor(T / L_valid) times.
        Remaining frames are filled by extending the last tokens.

        Args:
            text_kv:   (B, L, text_dim)
            text_mask: (B, L)  1=valid token, 0=pad
            T:         target frame count

        Returns:
            expanded:  (B, T, text_dim)
        """
        B, L, D = text_kv.shape
        device = text_kv.device

        expanded_list = []
        for b in range(B):
            valid_len = int(text_mask[b].sum().item())
            valid_len = max(valid_len, 1)
            valid_tokens = text_kv[b, :valid_len]  # (L_valid, D)

            repeat_base = T // valid_len
            remainder = T % valid_len

            repeats = torch.full((valid_len,), repeat_base, dtype=torch.long, device=device)
            if remainder > 0:
                repeats[-remainder:] += 1

            expanded = torch.repeat_interleave(valid_tokens, repeats, dim=0)  # (T, D)
            expanded_list.append(expanded)

        return torch.stack(expanded_list, dim=0)  # (B, T, D)

    def forward(
        self,
        x_t: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t:          (B, T, latent_dim) noisy latent sequence
            mask:         (B, T)             1=prompt, 0=target/pad
            timestep:     (B,)               diffusion timestep
            text_kv:      (B, L, text_dim)   text encoder output (mT5)
            text_mask:    (B, L)             text attention mask
            padding_mask: (B, T)             1=valid, 0=pad (for self-attention)
        """
        B, T, D = x_t.shape

        # --- 1. Expand text to frame length and project ---
        text_expanded = self.expand_text_to_frames(text_kv, text_mask, T)  # (B, T, text_dim)
        text_feat = self.text_proj(text_expanded)  # (B, T, dit_dim)

        # --- 2. Project audio input ---
        if mask.dim() == 2:
            mask_input = mask.unsqueeze(-1)
        else:
            mask_input = mask
        audio_feat = self.proj_in(torch.cat([x_t, mask_input], dim=-1))  # (B, T, dit_dim)

        # --- 3. Fuse: add text features to audio features ---
        x = audio_feat + text_feat

        # Timestep embedding
        time_emb = self.time_embed(timestep)  # (B, dit_dim)

        # RoPE
        rope_cos, rope_sin = self.rotary_emb(T, x.device)

        # --- 4. Transformer blocks (self-attention only) ---
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    block, x, time_emb,
                    rope_cos, rope_sin, padding_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, time_emb, rope_cos, rope_sin, padding_mask)

        # --- 5. Output projection with AdaLN ---
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

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

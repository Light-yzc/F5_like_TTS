"""
Duration Predictor for VAE-DiT TTS.

Predicts the number of latent frames to generate given text features.

Architecture (upgraded):
  - 1D ConvNeXt blocks for local pattern extraction (syllable rhythm)
  - Transformer encoder layers for global context (sentence prosody)
  - Multi-scale pooling → deeper MLP → scalar output

Why ConvNeXt + Transformer?
  - Duration depends on BOTH local patterns (word-level speaking rate)
    and global context (sentence-level prosody, pauses).
  - Conv captures "this word is typically 3 syllables long" efficiently.
  - Transformer captures "this is a question so the final word is longer."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """Lightweight 1D ConvNeXt block for local temporal patterns."""

    def __init__(self, dim: int, kernel_size: int = 7, mult: float = 2.0):
        super().__init__()
        inner_dim = int(dim * mult)
        padding = kernel_size // 2
        self.net = nn.Sequential(
            # Depthwise conv (group=dim, very parameter-efficient)
            nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim),
            nn.GroupNorm(1, dim),  # equivalent to LayerNorm for 1D
            # Pointwise expand + contract
            nn.Conv1d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Conv1d(inner_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L) — channel-first for Conv1d
        return x + self.net(x)


class DurationPredictor(nn.Module):
    """
    Predicts target audio duration (in latent frames) from text features.

    Architecture:
      1. ConvNeXt blocks for local rhythm patterns
      2. Transformer layers for global prosody context
      3. Multi-scale pooling (mean + max + attention) → MLP → scalar

    Config params (from yaml):
      - text_dim:    input feature dim (= dit_dim, from text encoder)
      - hidden_dim:  internal dim for MLP and conv blocks
      - num_layers:  number of transformer layers
    """

    def __init__(
        self,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        nhead: int = 8,
        num_conv_blocks: int = 3,
        conv_kernel: int = 7,
        latent_rate: int = 25,
    ):
        super().__init__()
        self.latent_rate = latent_rate

        # ── Project text features to hidden_dim ──
        self.proj_in = nn.Linear(text_dim, hidden_dim)

        # ── ConvNeXt blocks for local rhythm ──
        self.conv_blocks = nn.ModuleList([
            ConvNeXtBlock(hidden_dim, kernel_size=conv_kernel, mult=2.0)
            for _ in range(num_conv_blocks)
        ])
        self.conv_norm = nn.LayerNorm(hidden_dim)

        # ── Transformer for global prosody ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Multi-scale pooling ──
        # Attention pooling
        self.pool_weight = nn.Linear(hidden_dim, 1, bias=False)
        # Combine: attn_pool + mean_pool + max_pool → 3 * hidden_dim
        pool_dim = hidden_dim * 3

        # ── Prediction head (deeper MLP) ──
        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        target_text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            text_features: (B, L, D) — text encoder output
            text_mask:     (B, L)    — 1=valid, 0=pad
            target_text_mask: (B, L) - 1=target text, 0=prompt or pad (optional)

        Returns:
            predicted_frames: (B,) — predicted number of latent frames (float)
        """
        # If no target mask is provided, fallback to the standard mask
        if target_text_mask is None:
            target_text_mask = text_mask
        # Project to hidden dim
        x = self.proj_in(text_features)  # (B, L, hidden_dim)

        # ConvNeXt blocks (need channel-first)
        mask_bool = text_mask.bool()
        x_conv = x.transpose(1, 2)  # (B, hidden_dim, L)
        for conv_block in self.conv_blocks:
            x_conv = conv_block(x_conv)
        x = x_conv.transpose(1, 2)  # (B, L, hidden_dim)
        x = self.conv_norm(x)

        # Zero out padded positions before transformer
        x = x * text_mask.unsqueeze(-1)

        # Transformer with padding mask
        pad_mask = ~mask_bool  # True = pad
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # ── Multi-scale pooling ──
        # Use target_text_mask so it only pools target features
        pool_mask_bool = target_text_mask.bool()

        # 1. Attention pooling
        weights = self.pool_weight(x).squeeze(-1)  # (B, L)
        weights = weights.masked_fill(~pool_mask_bool, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (B, L)
        # Handle case where entire sequence is padded out (nan prevention)
        weights = torch.nan_to_num(weights, 0.0) 
        attn_pool = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # 2. Mean pooling (masked)
        x_masked = x * target_text_mask.unsqueeze(-1)
        lengths = target_text_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = x_masked.sum(dim=1) / lengths  # (B, D)

        # 3. Max pooling (masked)
        x_for_max = x.masked_fill(~pool_mask_bool.unsqueeze(-1), float("-inf"))
        max_pool = x_for_max.max(dim=1).values  # (B, D)
        # Handle -inf if whole sequence is masked
        max_pool = torch.nan_to_num(max_pool, 0.0)

        # Concatenate all pooling results
        pooled = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)  # (B, 3*D)

        # Predict frame count (always positive)
        predicted = self.head(pooled).squeeze(-1)  # (B,)
        return F.softplus(predicted)

    def predict_duration_sec(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        target_text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict duration in seconds."""
        frames = self.forward(text_features, text_mask, target_text_mask)
        return frames / self.latent_rate

    def loss(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        target_frames: torch.Tensor,
        target_text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSE loss between predicted and GT frame count."""
        pred = self.forward(text_features, text_mask, target_text_mask)
        return F.mse_loss(pred, target_frames.float())

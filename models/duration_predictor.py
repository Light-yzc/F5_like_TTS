"""
Duration Predictor for VAE-DiT TTS.

Predicts the number of latent frames to generate given text features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """
    Predicts target audio duration (in latent frames) from text features.

    Architecture:
      - Small Transformer encoder (2 layers) for contextualization
      - Weighted pooling → MLP → scalar output (frame count)

    Training:
      - Loss: MSE between predicted and ground-truth frame count
      - GT = number of latent frames in target audio
    """

    def __init__(
        self,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 2,
        nhead: int = 8,
        latent_rate: int = 25,
    ):
        super().__init__()
        self.latent_rate = latent_rate

        # Small transformer for text contextualization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learned pooling weight
        self.pool_weight = nn.Linear(text_dim, 1, bias=False)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_features: (B, L, D) — text encoder output (for TTS text only)
            text_mask:     (B, L)    — 1=valid, 0=pad

        Returns:
            predicted_frames: (B,) — predicted number of latent frames (float)
        """
        # Contextualize
        # Create src_key_padding_mask: True for PAD positions
        pad_mask = ~text_mask.bool()
        x = self.encoder(text_features, src_key_padding_mask=pad_mask)

        # Weighted pooling
        weights = self.pool_weight(x).squeeze(-1)  # (B, L)
        weights = weights.masked_fill(~text_mask.bool(), float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (B, L)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Predict frame count (always positive)
        predicted = self.head(pooled).squeeze(-1)  # (B,)
        return F.softplus(predicted)  # Ensure positive output

    def predict_duration_sec(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict duration in seconds."""
        frames = self.forward(text_features, text_mask)
        return frames / self.latent_rate

    def loss(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        target_frames: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and GT frame count."""
        pred = self.forward(text_features, text_mask)
        return F.mse_loss(pred, target_frames.float())

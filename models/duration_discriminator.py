"""
Duration Discriminator for VAE-DiT TTS (VITS2-style).

Conditions on text encoder hidden states and judges whether the
given duration is real (from dataset) or predicted (from DurationPredictor).

Architecture:
  - Input: text_features (B, L, text_dim) + log_duration (B,) scalar
  - Duration broadcast to all token positions → concat with text
  - Conv1d processing with Spectral Normalization
  - Masked mean pooling → MLP classifier → real/fake logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class DurationDiscriminator(nn.Module):
    """
    VITS2-style duration discriminator.

    Discriminates between (text, real_duration) and (text, predicted_duration).
    Uses Spectral Normalization for training stability.

    Args:
        text_dim:   dimension of text encoder features (dit_dim)
        hidden_dim: internal processing dimension
        num_layers: number of Conv1d layers
    """

    def __init__(
        self,
        text_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        # Project (text_features + duration) → hidden_dim
        self.input_proj = spectral_norm(nn.Linear(text_dim + 1, hidden_dim))

        # Conv1d stack for sequence processing (deeper, varied kernels)
        self.convs = nn.Sequential(
            spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global pooling after convs
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        log_duration: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_features: (B, L, text_dim) — text encoder output
            text_mask:     (B, L)           — 1=valid, 0=pad
            log_duration:  (B,)             — log-domain duration value

        Returns:
            logit: (B,) — discriminator logit (before sigmoid)
        """
        B, L, D = text_features.shape

        # Broadcast duration to all token positions
        dur_expanded = log_duration.view(B, 1, 1).expand(B, L, 1)

        # Concat text features + duration
        x = torch.cat([text_features, dur_expanded], dim=-1)  # (B, L, D+1)
        x = self.input_proj(x)  # (B, L, hidden_dim)

        # Conv processing (channel-first)
        x = x.transpose(1, 2)  # (B, hidden_dim, L)
        x = self.convs(x)

        # Global pooling
        pooled = self.pool(x).squeeze(-1)  # (B, hidden_dim)

        # Classify
        logit = self.classifier(pooled).squeeze(-1)  # (B,)
        return logit

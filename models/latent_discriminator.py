"""
Multi-Scale Latent Discriminator for VAE-DiT TTS.

Discriminates between real and generated VAE latent sequences.
Used with single-step denoising trick during flow matching training.

Architecture (inspired by HiFi-GAN MSD, adapted for latent space):
  - 3 sub-discriminators at different temporal scales (1x, 2x, 4x)
  - Each sub-D: Conv1d stack with Spectral Normalization
  - PatchGAN output: per-timestep real/fake scores
  - Feature maps exposed for feature matching loss

Reference: HiFi-GAN (Kong et al., 2020), VITS2 (Kong et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import List, Tuple


class SubDiscriminator(nn.Module):
    """Single-scale PatchGAN discriminator on latent sequences."""

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 256,
        scale: int = 1,
        num_layers: int = 6,
    ):
        super().__init__()
        self.scale = scale
        self.pool = nn.AvgPool1d(scale, scale) if scale > 1 else nn.Identity()

        # Conv stack: 6 layers with varied kernels and downsampling
        channels = [in_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim]
        kernels =  [7, 5, 5, 3, 3, 3]
        strides =  [1, 2, 2, 1, 2, 1]
        paddings = [3, 2, 2, 1, 1, 1]

        self.convs = nn.ModuleList()
        for i in range(min(num_layers, len(kernels))):
            self.convs.append(
                spectral_norm(nn.Conv1d(
                    channels[i], channels[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                ))
            )

        # Final: project to 1-channel score map
        self.final = spectral_norm(nn.Conv1d(channels[min(num_layers, len(kernels))], 1, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, D, T) latent sequence (channel-first)

        Returns:
            logit:  (B, 1, T') per-timestep discriminator scores
            fmaps:  list of intermediate feature maps for feature matching
        """
        x = self.pool(x)

        fmaps = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            fmaps.append(x)

        logit = self.final(x)  # (B, 1, T')
        return logit, fmaps


class MultiScaleLatentDiscriminator(nn.Module):
    """
    Multi-scale discriminator on VAE latent sequences.

    Uses 3 sub-discriminators at temporal scales 1x, 2x, 4x
    to capture both fine-grained and coarse patterns.

    Args:
        latent_dim:  VAE latent dimension (channels)
        hidden_dim:  internal Conv1d dimension
        num_scales:  number of temporal scales (default: 3)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_scales: int = 3,
    ):
        super().__init__()
        self.discs = nn.ModuleList([
            SubDiscriminator(
                in_dim=latent_dim,
                hidden_dim=hidden_dim,
                scale=2 ** i,
            )
            for i in range(num_scales)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x:    (B, T, D) latent sequence (time-first)
            mask: (B, T) optional mask (1=valid, 0=pad), applied via zeroing

        Returns:
            logits:    list of (B, 1, T'_i) per-scale score maps
            all_fmaps: list of [fmaps per scale] for feature matching
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # Convert to channel-first for Conv1d
        x = x.transpose(1, 2)  # (B, D, T)

        logits = []
        all_fmaps = []
        for disc in self.discs:
            logit, fmaps = disc(x)
            logits.append(logit)
            all_fmaps.append(fmaps)

        return logits, all_fmaps


# ── Loss utilities ──

def hinge_d_loss(
    d_real_logits: List[torch.Tensor],
    d_fake_logits: List[torch.Tensor],
) -> torch.Tensor:
    """Hinge loss for discriminator (multi-scale)."""
    loss = 0.0
    for d_real, d_fake in zip(d_real_logits, d_fake_logits):
        loss += F.relu(1.0 - d_real).mean()
        loss += F.relu(1.0 + d_fake).mean()
    return loss / len(d_real_logits)


def hinge_g_loss(
    d_fake_logits: List[torch.Tensor],
) -> torch.Tensor:
    """Hinge loss for generator (multi-scale)."""
    loss = 0.0
    for d_fake in d_fake_logits:
        loss += -d_fake.mean()
    return loss / len(d_fake_logits)


def feature_matching_loss(
    real_fmaps: List[List[torch.Tensor]],
    fake_fmaps: List[List[torch.Tensor]],
) -> torch.Tensor:
    """L1 feature matching loss across all scales and layers."""
    loss = 0.0
    count = 0
    for real_scale, fake_scale in zip(real_fmaps, fake_fmaps):
        for real_feat, fake_feat in zip(real_scale, fake_scale):
            loss += F.l1_loss(fake_feat, real_feat.detach())
            count += 1
    return loss / max(count, 1)

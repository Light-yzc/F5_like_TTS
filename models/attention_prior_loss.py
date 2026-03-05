"""
Attention Prior Loss for VAE-DiT TTS.

Applies a Gaussian diagonal penalty to cross-attention weights during training
to encourage monotonic text-audio alignment. Combined with CTC loss:
- CTC prevents word skipping (ensures all characters are covered)
- Attention Prior prevents repetition (penalizes attention staying on same token)

Reference:
    INTERSPEECH 2024: "Improving Robustness of LLM-based Speech Synthesis
    by Learning Monotonic Alignment"

Usage:
    ap_loss_fn = AttentionPriorLoss(sigma=0.4)
    loss = ap_loss_fn(attn_weights, text_mask, target_mask)
"""

import torch
import torch.nn as nn


class AttentionPriorLoss(nn.Module):
    """
    Guided Attention / Attention Prior loss using a Gaussian diagonal penalty.

    Generates a penalty matrix W where:
        W[t, l] = 1 - exp(-(t/T - l/L)^2 / (2 * sigma^2))

    Diagonal positions have W≈0 (no penalty), off-diagonal have W≈1 (heavy penalty).
    Loss = mean(attn_weights * W), masked to valid regions only.

    Args:
        sigma: Controls Gaussian width. Larger = more tolerant of speed variation.
               0.4 allows ~20% deviation from diagonal with minimal penalty.
    """

    def __init__(self, sigma: float = 0.4):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        attn_weights: torch.Tensor,
        text_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Attention Prior loss.

        Args:
            attn_weights: (B, H, T, L) cross-attention weights from one DiT layer
            text_mask:    (B, L) 1=valid text token, 0=pad
            target_mask:  (B, T) 1=target audio frame, 0=prompt/pad

        Returns:
            loss: scalar penalty (lower = more diagonal attention)
        """
        B, H, T, L = attn_weights.shape
        device = attn_weights.device
        dtype = attn_weights.dtype

        # Build penalty matrix W: (T, L)
        # W[t, l] = 1 - exp(-(t/T - l/L)^2 / 2σ^2)
        t_pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1) / max(T, 1)  # (T, 1)
        l_pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(0) / max(L, 1)  # (1, L)
        W = 1.0 - torch.exp(-((t_pos - l_pos) ** 2) / (2 * self.sigma ** 2))  # (T, L)

        # Apply attention weights: penalty = attn * W
        # attn_weights: (B, H, T, L), W: (T, L) → broadcast
        penalty = attn_weights * W.unsqueeze(0).unsqueeze(0)  # (B, H, T, L)

        # Mask: only penalize valid target frames × valid text tokens
        # target_mask: (B, T) → (B, 1, T, 1)
        # text_mask:   (B, L) → (B, 1, 1, L)
        mask = target_mask.unsqueeze(1).unsqueeze(-1) * text_mask.unsqueeze(1).unsqueeze(2)
        # mask shape: (B, 1, T, L)

        masked_penalty = penalty * mask
        num_valid = mask.sum() * H + 1e-8  # total valid elements across all heads

        loss = masked_penalty.sum() / num_valid

        return loss

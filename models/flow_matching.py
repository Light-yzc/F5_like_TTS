"""
Flow Matching training and inference for VAE-DiT TTS.

Training:
  - Construct noisy latent x_t = (1-t)*x0 + t*noise
  - Predict velocity v = noise - x0
  - Loss: MSE on generation region only (prompt region excluded)

Inference:
  - Euler ODE solver from noise → data
  - Classifier-Free Guidance (CFG)
  - Sway/cosine schedule for timesteps
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Callable
from tqdm import tqdm


class FlowMatching:
    """
    Flow Matching utilities for training and inference.

    Flow direction: x0 (data) → x1 (noise)
      - x_t = (1 - t) * x0 + t * x1
      - velocity field: v = x1 - x0
      - At inference, integrate from t=1 (noise) → t=0 (data)
    """

    def __init__(
        self,
        cfg_dropout_rate: float = 0.15,
        default_cfg_scale: float = 3.0,
        default_infer_steps: int = 30,
        sway_coef: float = -1.0,
    ):
        self.cfg_dropout_rate = cfg_dropout_rate
        self.default_cfg_scale = default_cfg_scale
        self.default_infer_steps = default_infer_steps
        self.sway_coef = sway_coef

    # =========================================================================
    # Training
    # =========================================================================

    def compute_loss(
        self,
        dit_model: torch.nn.Module,
        latent: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_mask: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        null_text_kv: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute flow matching training loss on packed latent.

        Args:
            dit_model:    DiT model
            latent:       (B, T, D) packed [prompt | target | pad] latent
            prompt_mask:  (B, T) 1=prompt frame, 0=other
            target_mask:  (B, T) 1=valid target frame, 0=other
            text_kv:      (B, L_text, dit_dim) text conditioning
            text_mask:    (B, L_text) text attention mask
            null_text_kv: (B, L_null, dit_dim) null condition for CFG training
            padding_mask: (B, T) 1=valid (prompt or target), 0=pad

        Returns:
            dict with 'loss' and 'mse' keys
        """
        B, T, D = latent.shape
        device = latent.device
        dtype = latent.dtype

        # Sample timestep t ~ U(0, 1) for each sample
        t = torch.rand(B, device=device, dtype=dtype)

        # Sample noise (same shape as full sequence)
        noise = torch.randn_like(latent)

        # Target velocity: v = noise - x0 (only meaningful where target_mask=1)
        v_target = noise - latent

        # Add noise ONLY to target region, keep prompt clean
        # x_t = prompt_data (where prompt) + interpolated (where target) + 0 (where pad)
        t_ = t.view(B, 1, 1)  # (B, 1, 1)
        target_mask_3d = target_mask.unsqueeze(-1)  # (B, T, 1)
        prompt_mask_3d = prompt_mask.unsqueeze(-1)   # (B, T, 1)

        x_t = (
            prompt_mask_3d * latent  # prompt frames: clean data
            + target_mask_3d * ((1 - t_) * latent + t_ * noise)  # target: noisy
            # padding frames: 0 (neither mask is 1)
        )

        # Mask channel for DiT input: prompt_mask (1=prompt, 0=other)
        mask_channel = prompt_mask

        # CFG training: randomly drop text condition
        if null_text_kv is not None and self.cfg_dropout_rate > 0:
            drop = torch.rand(B, device=device) < self.cfg_dropout_rate
            drop = drop.view(B, 1, 1)
            text_kv_train = torch.where(drop, null_text_kv.expand_as(text_kv), text_kv)
            null_mask = torch.ones(B, text_kv.shape[1], device=device, dtype=text_mask.dtype)
            text_mask_train = torch.where(
                drop.squeeze(-1), null_mask, text_mask
            )
        else:
            text_kv_train = text_kv
            text_mask_train = text_mask

        # Forward through DiT
        v_pred = dit_model(
            x_t, mask_channel, t, text_kv_train, text_mask_train,
            padding_mask=padding_mask,
        )

        # Loss: only on valid target frames (where target_mask=1)
        sq_error = (v_pred - v_target).pow(2)  # (B, T, D)
        masked_error = sq_error * target_mask_3d  # zero out non-target
        num_target_elements = target_mask.sum() * D  # total valid elements
        loss = masked_error.sum() / (num_target_elements + 1e-8)

        return {
            "loss": loss,
            "mse": loss.detach(),
        }

    # =========================================================================
    # Inference
    # =========================================================================

    def _get_time_schedule(
        self,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate timestep schedule from t=1 (noise) to t=0 (data)."""
        t_span = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)

        # Sway sampling: shift the schedule for better quality
        if self.sway_coef != 0:
            t_span = t_span + self.sway_coef * (
                torch.cos(math.pi / 2 * t_span) - 1 + t_span
            )
            t_span = t_span.clamp(0, 1)

        return t_span

    @torch.no_grad()
    def sample(
        self,
        dit_model: torch.nn.Module,
        prompt_latent: torch.Tensor,
        T_gen: int,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        null_text_kv: Optional[torch.Tensor] = None,
        null_text_mask: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate latent sequence via Euler ODE solver with CFG.

        Args:
            dit_model:      DiT model
            prompt_latent:  (B, T_prompt, D) clean prompt latent
            T_gen:          Number of frames to generate
            text_kv:        (B, L_text, dit_dim) text conditioning
            text_mask:      (B, L_text) text attention mask
            null_text_kv:   Text features for null condition (CFG)
            null_text_mask: Attention mask for null condition
            cfg_scale:      Classifier-free guidance scale
            n_steps:        Number of Euler steps
            seed:           Random seed for reproducibility

        Returns:
            gen_latent: (B, T_gen, D) generated latent
        """
        cfg_scale = cfg_scale or self.default_cfg_scale
        n_steps = n_steps or self.default_infer_steps

        B, T_prompt, D = prompt_latent.shape
        device = prompt_latent.device
        dtype = prompt_latent.dtype
        T_total = T_prompt + T_gen

        # Initialize noise
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(B, T_gen, D, generator=generator, device=device, dtype=dtype)
        else:
            noise = torch.randn(B, T_gen, D, device=device, dtype=dtype)

        # Construct initial sequence
        x = torch.cat([prompt_latent, noise], dim=1)  # (B, T_total, D)
        mask = torch.cat([
            torch.ones(B, T_prompt, device=device, dtype=dtype),
            torch.zeros(B, T_gen, device=device, dtype=dtype),
        ], dim=1)

        # Padding mask: all valid during inference (no batch padding)
        padding_mask = torch.ones(B, T_total, device=device, dtype=dtype)

        # Time schedule
        t_span = self._get_time_schedule(n_steps, device, dtype)

        # Euler integration
        steps = range(n_steps)
        if show_progress:
            steps = tqdm(steps, desc="Flow Matching", leave=False)

        for i in steps:
            t_cur = t_span[i].expand(B)
            dt = t_span[i + 1] - t_span[i]  # Negative (going from 1 → 0)

            # Conditional prediction
            v_cond = dit_model(x, mask, t_cur, text_kv, text_mask, padding_mask=padding_mask)

            if cfg_scale > 0 and null_text_kv is not None:
                # Unconditional prediction
                v_uncond = dit_model(x, mask, t_cur, null_text_kv, null_text_mask, padding_mask=padding_mask)
                # CFG: v = v_uncond + scale * (v_cond - v_uncond)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            # Euler step: only update generation region
            x_gen = x[:, T_prompt:] + dt * v[:, T_prompt:]
            x = torch.cat([prompt_latent, x_gen], dim=1)

        return x[:, T_prompt:]  # (B, T_gen, D)

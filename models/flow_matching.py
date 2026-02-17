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
        prompt_latent: torch.Tensor,
        target_latent: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        null_text_kv: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute flow matching training loss.

        Args:
            dit_model:      DiT model
            prompt_latent:  (B, T_prompt, D) clean prompt latent
            target_latent:  (B, T_gen, D) clean target latent
            text_kv:        (B, L_text, dit_dim) text conditioning
            text_mask:      (B, L_text) text attention mask
            null_text_kv:   (B, L_null, dit_dim) null condition for CFG training

        Returns:
            dict with 'loss' and 'mse' keys
        """
        B, T_gen, D = target_latent.shape
        T_prompt = prompt_latent.shape[1]
        device = target_latent.device
        dtype = target_latent.dtype

        # Sample timestep t ~ U(0, 1) for each sample
        t = torch.rand(B, device=device, dtype=dtype)

        # Sample noise
        noise = torch.randn_like(target_latent)

        # Interpolate: x_t = (1 - t) * x0 + t * noise
        t_ = t.view(B, 1, 1)
        x_t_gen = (1 - t_) * target_latent + t_ * noise

        # Construct full sequence
        x_t = torch.cat([prompt_latent, x_t_gen], dim=1)  # (B, T_total, D)
        mask = torch.cat([
            torch.ones(B, T_prompt, device=device, dtype=dtype),
            torch.zeros(B, T_gen, device=device, dtype=dtype),
        ], dim=1)  # (B, T_total)

        # CFG training: randomly drop text condition
        if null_text_kv is not None and self.cfg_dropout_rate > 0:
            drop = torch.rand(B, device=device) < self.cfg_dropout_rate
            drop = drop.view(B, 1, 1)
            text_kv_train = torch.where(drop, null_text_kv.expand_as(text_kv), text_kv)
            # Also mask the text attention when dropped
            null_mask = torch.ones(B, text_kv.shape[1], device=device, dtype=text_mask.dtype)
            text_mask_train = torch.where(
                drop.squeeze(-1), null_mask, text_mask
            )
        else:
            text_kv_train = text_kv
            text_mask_train = text_mask

        # Forward
        v_pred = dit_model(x_t, mask, t, text_kv_train, text_mask_train)

        # Target velocity: v = noise - x0
        v_target = noise - target_latent

        # Loss: only on generation region
        loss = F.mse_loss(v_pred[:, T_prompt:], v_target)

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
            v_cond = dit_model(x, mask, t_cur, text_kv, text_mask)

            if cfg_scale > 0 and null_text_kv is not None:
                # Unconditional prediction
                v_uncond = dit_model(x, mask, t_cur, null_text_kv, null_text_mask)
                # CFG: v = v_uncond + scale * (v_cond - v_uncond)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            # Euler step: only update generation region
            x_gen = x[:, T_prompt:] + dt * v[:, T_prompt:]
            x = torch.cat([prompt_latent, x_gen], dim=1)

        return x[:, T_prompt:]  # (B, T_gen, D)

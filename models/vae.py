from diffusers.models import AutoencoderOobleck


def load_vae(vae_path: str, device = 'auto', percision='bf16'):
    if not vae_path.is_dir():
        raise FileNotFoundError(f"VAE directory not found: {vae_path}")
    vae = AutoencoderOobleck.from_pretrained(vae_path)
    vae = vae.to(device, dtype=percision)
    vae.eval()
    return vae


def vae_encode(vae, waveform: torch.Tensor) -> torch.Tensor:
    """Encode waveform to latent."""
    return vae.encode(waveform).latent_dist.sample()


def vae_decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to waveform."""
    return vae.decode(latent).sample
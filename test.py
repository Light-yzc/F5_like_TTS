from models.vae import *
import torch
import torchaudio

vae = load_vae(r"D:\CODE\F5_like_TTS\model_files\vae").to(torch.float16)

with torch.no_grad():
    latent = torch.load("D:\CODE\F5_like_TTS\jvs_proceed\jvs001-normal-jvs001BASIC5000-0025.pt")
    print(latent.shape)
    audio = vae_decode(vae, latent.unsqueeze(0).to(vae.device, vae.dtype))
    wav = audio.squeeze(1).to(torch.float32).cpu()
    torchaudio.save("./test.wav", wav, 48000)
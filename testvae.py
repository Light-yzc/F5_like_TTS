# from models.vae import load_vae, vae_encode,vae_decode
# import torch
# import torchaudio
# device = "cuda" if torch.cuda.is_available() else "cpu"
# vae = load_vae(r'D:\CODE\F5_like_TTS\model_files\vae', device=device,precision='fp16')

# wav, sr = torchaudio.load(r'D:\CODE\F5_like_TTS\tts_dataset\test\wav\SSB0005\SSB00050353.wav')
# wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
# wav = torch.clamp(wav, -1.0, 1.0).to(device=vae.device, dtype=vae.dtype).unsqueeze(1).repeat(1, 2, 1)
# latent = vae_encode(vae, wav)
# decode = vae_decode(vae, latent).cpu().float()
# print(decode.shape)
# torchaudio.save('out_put.wav', decode.squeeze(), 48000)

from data.dataset import TTSDataset, collate_fn

data_set = TTSDataset(data_root = r'D:\CODE\F5_like_TTS\processed_dir\test')

print(len(data_set))
print(data_set[1])
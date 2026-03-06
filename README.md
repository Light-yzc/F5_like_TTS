---
language:
  - zh
  - ja
  - en
license: apache-2.0
tags:
  - text-to-speech
  - tts
  - speech-synthesis
  - diffusion
  - flow-matching
  - dit
  - multilingual
  - zero-shot
pipeline_tag: text-to-speech
library_name: transformers
---

# VAE-DiT TTS

A multilingual zero-shot Text-to-Speech model based on **Diffusion Transformer (DiT)** and **Flow Matching**, capable of cloning any voice with a short audio prompt.

## ✨ Features

- 🌏 **Multilingual**: Chinese (ZH), Japanese (JA), English (EN)
- 🎙️ **Zero-shot voice cloning**: 3-10s reference audio is enough
- 🏗️ **~500M parameters**: DiT backbone with cross-attention alignment
- ⚡ **30-step inference**: Sway sampling with classifier-free guidance
- 🎵 **48kHz output**: High-fidelity audio via Oobleck VAE

## 📊 Architecture

| Component | Details |
|---|---|
| **Backbone** | DiT (30 layers, 1024 dim, 16 heads) |
| **Text Encoder** | F5-style CharTokenizer + ConvNeXt (14 layers) |
| **Audio Codec** | AutoencoderOobleck (64-dim latent, 25fps) |
| **Duration Predictor** | ConvNeXt + Transformer + Multi-scale Pooling |
| **Sampling** | Flow Matching with Sway Sampling (coef=-1.0) |
| **Training** | CTC Alignment Loss + Flow Matching Loss |

## 🚀 Quick Start

### Install dependencies

```bash
pip install torch torchaudio transformers diffusers phonemizer pykakasi pypinyin rjieba
apt install espeak-ng  # Linux (required for G2P)
# brew install espeak-ng  # macOS
```

### Inference

```python
import torch
from huggingface_hub import hf_hub_download

# Download checkpoint
ckpt_path = hf_hub_download(
    repo_id="<HF_REPO_ID>",
    filename="checkpoint.pt"
)
vocab_path = hf_hub_download(
    repo_id="<HF_REPO_ID>",
    filename="char_vocab.json"
)

# Load models
from inference import load_checkpoint
from models.vae import load_vae, vae_encode, vae_decode

device = torch.device("cuda")
dit, text_encoder, dur_pred, flow, cfg, tokenizer = load_checkpoint(
    ckpt_path, device, vocab_path_override=vocab_path
)
vae = load_vae("stabilityai/stable-audio-open-1.0", device=device, precision="fp16")

# Generate speech
from inference import inference
inference(
    dit, text_encoder, dur_pred, flow, cfg,
    prompt_audio_path="prompt.wav",
    prompt_text="参考音频的文字",
    tts_text="你好，今天天气真好！",
    prompt_language="ZH",
    tts_language="ZH",
    char_tokenizer=tokenizer,
    vae_encode_fn=lambda wav: vae_encode(vae, wav),
    vae_decode_fn=lambda lat: vae_decode(vae, lat),
    output_path="output.wav",
    seed=42,
)
```

### Jupyter Notebook

See [`inference_notebook.ipynb`](./inference_notebook.ipynb) for an interactive demo with audio playback.

## 📁 Files

```
checkpoint.pt        # Model weights (DiT + TextEncoder + DurationPredictor)
char_vocab.json      # IPA character vocabulary (94 tokens)
inference.py         # Inference script
inference_notebook.ipynb  # Interactive Jupyter notebook
configs/model_medium.yaml # Model configuration
```

## ⚙️ Inference Parameters

| Parameter | Default | Description |
|---|---|---|
| `cfg_scale` | 3.0 | Classifier-free guidance scale |
| `n_steps` | 30 | Number of sampling steps |
| `duration` | auto | Override output duration (seconds) |
| `seed` | None | Random seed for reproducibility |

## 📝 Training Details

- **Data**: Multilingual speech corpus (ZH/JA/EN)
- **Steps**: 350K
- **Batch size**: 13
- **Optimizer**: AdamW 8-bit, lr=1.5e-4, cosine schedule
- **Audio**: 48kHz, Oobleck VAE latent (64-dim, 25fps)
- **Auxiliary losses**: CTC alignment loss (monotonic coverage)

## 🙏 Acknowledgements

- [F5-TTS](https://github.com/SWivid/F5-TTS) — Text encoder design inspiration
- [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) — Oobleck VAE
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — IPA G2P backend

## License

Apache 2.0

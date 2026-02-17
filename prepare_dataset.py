"""
Preprocess dataset for VAE-DiT TTS training.

Expects AISHELL-3 style directory:
  base_dir/
    train/
      wavs/
        SSB0001/
          SSB00010001.wav
          SSB00010002.wav
        SSB0002/
          ...
      content.txt   (lines like: "SSB00010001 pinyin1 汉字1 pinyin2 汉字2 ...")
    test/
      ...

Output:
  processed_dir/
    train/
      wavs/
        SSB0001_SSB00010001.pt   (VAE latent)
      content.txt                (lines like: "SSB0001_SSB00010001_汉字汉字汉字")
    test/
      ...
"""

import os
import argparse
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from models.vae import load_vae, vae_encode


def handle_wav(base_dir: str, processed_dir: str, split: str, vae):
    """Encode all wavs to VAE latents."""
    path_wav = os.path.join(base_dir, split, 'wavs')
    out_wav_dir = os.path.join(processed_dir, split, 'wavs')
    os.makedirs(out_wav_dir, exist_ok=True)

    folder = Path(path_wav)
    file_paths = [str(p) for p in folder.rglob('*.wav')]
    print(f"Found {len(file_paths)} wav files in {path_wav}")

    with torch.no_grad():
        for i in tqdm(file_paths, desc=f"Encoding {split}"):
            file = Path(i)
            wav, sr = torchaudio.load(file)

            # # Convert to mono
            # if wav.shape[0] > 1:
            #     wav = wav.mean(dim=0, keepdim=True)

            # Resample to 48kHz
            if sr != 48000:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
            wav = torch.clamp(wav, -1.0, 1.0).to(vae.device)
            # VAE encode: (1, 1, samples) → (1, T, D)
            latent = vae_encode(vae, wav)
            latent = latent.squeeze(0).cpu()  # (T, D)

            # Save: {speaker}_{utterance_id}.pt
            out_name = f"{file.parent.name}_{file.stem}.pt"
            torch.save(latent, os.path.join(out_wav_dir, out_name))


def handle_txt(base_dir: str, processed_dir: str, split: str):
    """
    Parse content.txt and extract Chinese characters only.

    AISHELL-3 format: "SSB00010001 pinyin1 汉字1 pinyin2 汉字2 ..."
    Output format:    "SSB0001_SSB00010001_汉字1汉字2..."
    """
    path_txt = os.path.join(base_dir, split, 'content.txt')
    out_txt = os.path.join(processed_dir, split, 'content.txt')
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    with open(path_txt, 'r', encoding='utf-8') as fin, \
         open(out_txt, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            utterance_id = parts[0]  # e.g. SSB00010001
            speaker = utterance_id[:7]  # e.g. SSB0001
            # Extract Chinese chars (every other token after the utterance ID)
            # AISHELL-3: "SSB00010001 pin1 汉 pin2 字 ..."
            only_text = ''.join(parts[2::2])  # skip id and pinyins
            fout.write(f"{speaker}_{utterance_id}_{only_text}\n")

    print(f"Processed text for {split}: {out_txt}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for VAE-DiT TTS")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to raw dataset (AISHELL-3)")
    parser.add_argument("--processed_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--vae_path", type=str, default="models/vae_model", help="Path to VAE model")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    if args.processed_dir is None:
        args.processed_dir = os.path.join(args.base_dir, 'processed')

    # Process text first (fast)
    for split in args.splits:
        handle_txt(args.base_dir, args.processed_dir, split)

    # Load VAE and encode audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(args.vae_path, device=device)
    for split in args.splits:
        handle_wav(args.base_dir, args.processed_dir, split, vae)

    print("Done!")


if __name__ == "__main__":
    main()

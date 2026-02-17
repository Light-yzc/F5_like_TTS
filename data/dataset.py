"""
Dataset and data loading for VAE-DiT TTS.

Expects pre-processed data where audio has been encoded to VAE latents offline.
Each sample in the dataset directory should contain:
  - latent.pt: (T, D) VAE latent tensor
  - text.txt: full transcription text

During training, each sample is randomly split into prompt + target.
"""

import os
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class TTSDataset(Dataset):
    """
    TTS dataset loading pre-computed VAE latents and text.

    Data directory structure:
      data_root/
        sample_000/
          latent.pt   — (T, D) tensor
          text.txt    — transcription
        sample_001/
          ...

    Each __getitem__ returns:
      - prompt_latent: (T_prompt, D)
      - target_latent: (T_gen, D)
      - text: str (full text, to be tokenized by collator)
    """

    def __init__(
        self,
        data_root: str,
        latent_rate: int = 25,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 30.0,
        prompt_ratio_min: float = 0.2,
        prompt_ratio_max: float = 0.5,
    ):
        super().__init__()
        self.data_root = data_root
        self.latent_rate = latent_rate
        self.min_frames = int(min_duration_sec * latent_rate)
        self.max_frames = int(max_duration_sec * latent_rate)
        self.prompt_ratio_min = prompt_ratio_min
        self.prompt_ratio_max = prompt_ratio_max

        # Discover samples
        self.samples = []
        if os.path.isdir(data_root):
            for name in sorted(os.listdir(data_root)):
                sample_dir = os.path.join(data_root, name)
                latent_path = os.path.join(sample_dir, "latent.pt")
                text_path = os.path.join(sample_dir, "text.txt")
                if os.path.isfile(latent_path) and os.path.isfile(text_path):
                    self.samples.append({
                        "latent_path": latent_path,
                        "text_path": text_path,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load latent
        latent = torch.load(sample["latent_path"], map_location="cpu", weights_only=True)
        # Clamp to max duration
        if latent.shape[0] > self.max_frames:
            start = random.randint(0, latent.shape[0] - self.max_frames)
            latent = latent[start : start + self.max_frames]

        T = latent.shape[0]
        if T < self.min_frames:
            # Skip too-short samples (should be filtered in preprocessing)
            # Pad with zeros as fallback
            pad = self.min_frames - T
            latent = F.pad(latent, (0, 0, 0, pad))
            T = self.min_frames

        # Random split into prompt + target
        ratio = random.uniform(self.prompt_ratio_min, self.prompt_ratio_max)
        split = max(1, int(T * ratio))
        split = min(split, T - 1)  # Ensure at least 1 frame for target

        prompt_latent = latent[:split]
        target_latent = latent[split:]

        # Load text
        with open(sample["text_path"], "r", encoding="utf-8") as f:
            text = f.read().strip()

        return {
            "prompt_latent": prompt_latent,     # (T_prompt, D)
            "target_latent": target_latent,     # (T_gen, D)
            "full_text": text,                  # str
            "total_frames": T,                  # int
            "target_frames": T - split,         # int (for duration predictor GT)
        }


def collate_fn(batch: list[dict], tokenizer=None, max_text_len: int = 512) -> dict:
    """
    Collate function that pads latents and tokenizes text.

    Args:
        batch: list of dataset items
        tokenizer: T5 tokenizer for text encoding
        max_text_len: maximum text token length

    Returns:
        Collated batch dict with padded tensors
    """
    B = len(batch)

    # Find max lengths
    max_prompt = max(item["prompt_latent"].shape[0] for item in batch)
    max_target = max(item["target_latent"].shape[0] for item in batch)
    D = batch[0]["prompt_latent"].shape[-1]

    # Pad latents
    prompt_latents = torch.zeros(B, max_prompt, D)
    target_latents = torch.zeros(B, max_target, D)
    prompt_masks = torch.zeros(B, max_prompt)
    target_masks = torch.zeros(B, max_target)
    target_frames = torch.zeros(B)

    for i, item in enumerate(batch):
        t_p = item["prompt_latent"].shape[0]
        t_g = item["target_latent"].shape[0]

        prompt_latents[i, :t_p] = item["prompt_latent"]
        target_latents[i, :t_g] = item["target_latent"]
        prompt_masks[i, :t_p] = 1.0
        target_masks[i, :t_g] = 1.0
        target_frames[i] = item["target_frames"]

    result = {
        "prompt_latent": prompt_latents,
        "target_latent": target_latents,
        "prompt_mask": prompt_masks,
        "target_mask": target_masks,
        "target_frames": target_frames,
    }

    # Tokenize text
    if tokenizer is not None:
        texts = [item["full_text"] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        result["input_ids"] = encoded["input_ids"]
        result["attention_mask"] = encoded["attention_mask"]
    else:
        result["texts"] = [item["full_text"] for item in batch]

    return result

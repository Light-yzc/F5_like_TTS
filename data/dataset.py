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
import tqdm
from pathlib import Path

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
        # content.txt format: "speaker_utteranceId_text"
        # e.g. "SSB0001_SSB00010001_今天天气真好"
        self.samples = []
        self.speaker_to_indices = {}  # speaker_id → [sample indices]
        folder = Path(data_root)
        with open(os.path.join(folder, 'content.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                speaker, utt_id, text = line.split('_', 2)  # maxsplit=2
                sample_idx = len(self.samples)
                latent_name = f"{speaker}_{utt_id}.pt"
                self.samples.append({
                    "latent_path": str(folder / 'wavs' / latent_name),
                    "text": text,
                    "speaker": speaker,
                })
                self.speaker_to_indices.setdefault(speaker, []).append(sample_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_latent(self, path: str) -> torch.Tensor:
        """Load latent and clamp to valid length range."""
        latent = torch.load(path, map_location="cpu", weights_only=True)
        if latent.shape[0] > self.max_frames:
            start = random.randint(0, latent.shape[0] - self.max_frames)
            latent = latent[start : start + self.max_frames]
        if latent.shape[0] < self.min_frames:
            pad = self.min_frames - latent.shape[0]
            latent = F.pad(latent, (0, 0, 0, pad))
        return latent

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        speaker = sample["speaker"]

        # Target: current sample
        target_latent = self._load_latent(sample["latent_path"])

        # Prompt: pick a different utterance from the same speaker
        same_speaker_indices = self.speaker_to_indices[speaker]
        if len(same_speaker_indices) > 1:
            prompt_idx = idx
            while prompt_idx == idx:
                prompt_idx = random.choice(same_speaker_indices)
            prompt_latent = self._load_latent(self.samples[prompt_idx]["latent_path"])
            prompt_text = self.samples[prompt_idx]["text"]
        else:
            # Fallback: same-utterance split
            ratio = random.uniform(self.prompt_ratio_min, self.prompt_ratio_max)
            split = max(1, min(int(target_latent.shape[0] * ratio), target_latent.shape[0] - 1))
            prompt_latent = target_latent[:split]
            target_latent = target_latent[split:]
            prompt_text = sample["text"]  # full text as approximation

        # Combined text for T5: "prompt_text [SEP] target_text"
        target_text = sample["text"]
        full_text = f"{prompt_text} [SEP] {target_text}"

        return {
            "prompt_latent": prompt_latent,
            "target_latent": target_latent,
            "full_text": full_text,
            "total_frames": prompt_latent.shape[0] + target_latent.shape[0],
            "target_frames": target_latent.shape[0],
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

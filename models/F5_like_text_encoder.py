"""
F5-like Text Encoder for TTS.

Character-level embedding + ConvNeXt blocks for local context modeling.
Replaces mT5 (580M frozen params) with a lightweight trainable encoder (~5M params).

Architecture:
  input text → char tokenization → nn.Embedding → ConvNeXt blocks → (B, L, dit_dim)

Reference: F5-TTS (https://arxiv.org/abs/2410.06885)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Character Tokenizer (no dependency, supports CJK + ASCII + punctuation)
# =============================================================================

class CharTokenizer:
    """
    Simple character-level tokenizer.
    Vocab: PAD=0, UNK=1, all unique chars seen during build_vocab().
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        self.pad_id = 0
        self.unk_id = 1
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.id_to_char = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def build_vocab(self, texts: list[str]):
        """Build vocab from a list of texts. Call once before training."""
        for text in texts:
            for ch in text:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)
        self.id_to_char = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(ch, self.unk_id) for ch in text]

    def batch_encode(
        self,
        texts: list[str],
        max_len: int | None = None,
        return_tensors: bool = True,
    ) -> dict:
        """
        Encode a batch of texts, pad to max length.

        Returns:
            input_ids:      (B, L) long tensor
            attention_mask: (B, L) float tensor  1=valid, 0=pad
        """
        encoded = [self.encode(t) for t in texts]
        if max_len is None:
            max_len = max(len(e) for e in encoded)
        else:
            encoded = [e[:max_len] for e in encoded]

        input_ids = []
        attention_mask = []
        for e in encoded:
            pad_len = max_len - len(e)
            input_ids.append(e + [self.pad_id] * pad_len)
            attention_mask.append([1.0] * len(e) + [0.0] * pad_len)

        if return_tensors:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __call__(self, texts, padding=True, truncation=True, max_length=512,
                 return_tensors="pt", **kwargs):
        """HuggingFace-compatible interface for collate_fn."""
        return self.batch_encode(texts, max_len=max_length if truncation else None,
                                 return_tensors=(return_tensors == "pt"))

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        import json
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)


# =============================================================================
# ConvNeXt Block (1D, for sequence modeling)
# =============================================================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt V2 block (1D variant).

    Structure:
      x → DepthwiseConv1d(k=7) → LayerNorm → Linear(dim*4) → GELU → Linear(dim) → + x
    """

    def __init__(self, dim: int, mult: int = 4, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        residual = x
        # Conv operates on (B, D, T)
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.ffn(x)
        return x + residual


# =============================================================================
# F5-like Text Encoder
# =============================================================================

class F5TextEncoder(nn.Module):
    """
    Character-level text encoder with ConvNeXt context modeling.

    Args:
        vocab_size:   number of unique characters (from CharTokenizer)
        dim:          output dimension (should match dit_dim)
        depth:        number of ConvNeXt blocks
        kernel_size:  ConvNeXt depthwise conv kernel size
        ff_mult:      ConvNeXt FFN expansion factor
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        dim: int = 768,
        depth: int = 4,
        kernel_size: int = 7,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)  # PAD=0
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim, mult=ff_mult, kernel_size=kernel_size)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, L) character token IDs
            attention_mask: (B, L) 1=valid, 0=pad

        Returns:
            text_features:  (B, L, dim) contextualized character embeddings
            attention_mask: (B, L) passed through unchanged
        """
        x = self.embedding(input_ids)  # (B, L, dim)

        # Mask padding positions to zero before conv
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        for block in self.blocks:
            x = block(x)
            # Re-mask after each block to prevent pad positions leaking
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)

        x = self.out_norm(x)
        return x, attention_mask

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

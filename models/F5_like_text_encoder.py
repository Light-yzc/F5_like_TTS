"""
F5-like Text Encoder for TTS.

Character-level embedding + ConvNeXt blocks for local context modeling
+ optional Transformer blocks for global context modeling.

Architecture:
  input text → char tokenization → nn.Embedding
             → ConvNeXt blocks (local rhythm/phonetics)
             → Transformer blocks (global semantics/prosody)
             → (B, L, text_enc_dim)

Reference: F5-TTS (https://arxiv.org/abs/2410.06885)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Character Tokenizer (no dependency, supports CJK + ASCII + punctuation)
# =============================================================================

class CharTokenizer:
    """
    Character-level tokenizer with optional multi-character special token support.

    If the vocab contains entries such as "<PAD>", "<UNK>", or "[EROGE]",
    encode() will greedily match the longest special token first, and fall back
    to single-character lookup for the rest.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        self.pad_id = 0
        self.unk_id = 1
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        self.special_tokens = sorted(
            [token for token in self.vocab if len(token) > 1],
            key=len,
            reverse=True,
        )

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
        self.special_tokens = sorted(
            [token for token in self.vocab if len(token) > 1],
            key=len,
            reverse=True,
        )

    def encode(self, text: str) -> list[int]:
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for token in self.special_tokens:
                if text.startswith(token, i):
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue
            ids.append(self.vocab.get(text[i], self.unk_id))
            i += 1
        return ids

    def encoded_length(self, text: str) -> int:
        return len(self.encode(text))

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
        if isinstance(texts, str):
            texts = [texts]
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
# Transformer Building Blocks for Text Encoder
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class TextSelfAttention(nn.Module):
    """Multi-head self-attention for text encoder (no RoPE, uses absolute pos emb)."""

    def __init__(self, dim: int, heads: int, head_dim: int = 64):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim

        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))
            attn_mask = attn_mask.to(q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SiLUGatedFFN(nn.Module):
    """SiLU-gated Feed-Forward Network (same as LLaMA MLP)."""

    def __init__(self, dim: int, mult: float = 2.5):
        super().__init__()
        hidden = int(dim * mult)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextTransformerBlock(nn.Module):
    """
    Pre-norm Transformer block for text encoder.

    Structure:
      x → RMSNorm → Self-Attention → + x → RMSNorm → SiLU-gated FFN → + x
    """

    def __init__(self, dim: int, heads: int, head_dim: int = 64, ff_mult: float = 2.5):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = TextSelfAttention(dim, heads, head_dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = SiLUGatedFFN(dim, ff_mult)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ff(self.ff_norm(x))
        return x


# =============================================================================
# F5-like Text Encoder
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """Absolute Positional Embeddings to replace Cross-Attention RoPE dependencies."""
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_seq_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class F5TextEncoder(nn.Module):
    """
    Character-level text encoder with ConvNeXt + Transformer context modeling.

    Args:
        vocab_size:          number of unique characters (from CharTokenizer)
        dim:                 encoder hidden dimension (can differ from dit_dim)
        depth:               number of ConvNeXt blocks (local context)
        kernel_size:         ConvNeXt depthwise conv kernel size
        ff_mult:             ConvNeXt FFN expansion factor
        transformer_depth:   number of Transformer blocks (global context), 0 to disable
        transformer_heads:   number of attention heads for Transformer blocks
        transformer_ff_mult: FFN expansion factor for Transformer blocks
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        dim: int = 512,
        depth: int = 4,
        kernel_size: int = 7,
        ff_mult: int = 4,
        transformer_depth: int = 0,
        transformer_heads: int = 8,
        transformer_ff_mult: float = 2.5,
    ):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)  # PAD=0
        self.pos_emb = SinusoidalPositionalEmbedding(dim)

        # ConvNeXt blocks for local context (phonetics, rhythm)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim, mult=ff_mult, kernel_size=kernel_size)
            for _ in range(depth)
        ])

        # Transformer blocks for global context (semantics, prosody)
        if transformer_depth > 0:
            head_dim = dim // transformer_heads
            self.transformer_blocks = nn.ModuleList([
                TextTransformerBlock(dim, transformer_heads, head_dim, transformer_ff_mult)
                for _ in range(transformer_depth)
            ])
        else:
            self.transformer_blocks = None

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
        x = self.pos_emb(x)            # (B, L, dim) - Absolute position for cross-attention
        
        # Mask padding positions to zero before conv
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        # ConvNeXt blocks (local context)
        for block in self.blocks:
            x = block(x)
            # Re-mask after each block to prevent pad positions leaking
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)

        # Transformer blocks (global context)
        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                x = block(x, mask=attention_mask)
                # Re-mask after each block
                if attention_mask is not None:
                    x = x * attention_mask.unsqueeze(-1)

        x = self.out_norm(x)
        return x, attention_mask

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

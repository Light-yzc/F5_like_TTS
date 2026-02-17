"""
Text Encoder wrapper for VAE-DiT TTS.

Uses a frozen mT5-large encoder with a trainable linear projector.
"""

import torch
import torch.nn as nn
from typing import Optional


class TextConditioner(nn.Module):
    """
    Frozen mT5 encoder + trainable projector for text conditioning.

    Usage:
        conditioner = TextConditioner(dit_dim=1024)
        text_kv, text_mask = conditioner(input_ids, attention_mask)
        # text_kv: (B, L, dit_dim) → feed to DiT cross-attention
    """

    def __init__(
        self,
        model_name: str = "google/mt5-large",
        text_dim: int = 1024,
        dit_dim: int = 1024,
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import T5EncoderModel

        self.encoder = T5EncoderModel.from_pretrained(model_name, weights_only=False)
        self.text_dim = text_dim
        self.dit_dim = dit_dim

        if freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        # Trainable projector: text_dim → dit_dim
        self.projector = nn.Linear(text_dim, dit_dim, bias=False)

    @torch.no_grad()
    def encode_text(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text with frozen T5 encoder."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, text_dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, L) tokenized text
            attention_mask: (B, L) 1=valid, 0=pad

        Returns:
            text_kv:   (B, L, dit_dim) projected text features
            text_mask: (B, L) attention mask (same as input)
        """
        text_features = self.encode_text(input_ids, attention_mask)
        text_kv = self.projector(text_features)  # (B, L, dit_dim)
        return text_kv, attention_mask

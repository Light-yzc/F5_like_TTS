"""
CTC Alignment Head for VAE-DiT TTS.

Provides auxiliary CTC supervision to encourage monotonic text-audio alignment
in the DiT's cross-attention layers. Applied to the last DiT block's hidden
states so gradients flow back through all layers via residual connections.

Usage:
    ctc_head = CTCAlignmentHead(dit_dim=1024, vocab_size=94)
    loss = ctc_head.loss(hidden_states, target_mask, ctc_targets, ctc_target_lengths)

Reference:
    - A-DMA (arXiv 2412.09563): CTC alignment in diffusion TTS
    - CTC-TTS: CTC-based alignment for LLM-based TTS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCAlignmentHead(nn.Module):
    """
    CTC projection head for alignment supervision.

    Takes the last DiT block's hidden states and projects them to
    per-frame character predictions. CTC loss enforces monotonic
    alignment without requiring frame-level time annotations.

    Args:
        dit_dim:    DiT hidden dimension (e.g., 1024)
        vocab_size: Number of characters in char_vocab (excluding blank)
        blank_id:   CTC blank token ID (appended after vocab)
    """

    def __init__(self, dit_dim: int, vocab_size: int, blank_id: int = None):
        super().__init__()
        self.vocab_size = vocab_size
        # blank_id defaults to vocab_size (last class)
        self.blank_id = blank_id if blank_id is not None else vocab_size
        self.num_classes = vocab_size + 1  # +1 for CTC blank

        # Single linear projection: hidden → per-frame character logits
        self.proj = nn.Linear(dit_dim, self.num_classes)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to log probabilities.

        Args:
            hidden_states: (B, T, dit_dim) last DiT block output

        Returns:
            log_probs: (T, B, num_classes) — CTC requires time-first
        """
        logits = self.proj(hidden_states)  # (B, T, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, num_classes)
        log_probs = log_probs.permute(1, 0, 2)  # (T, B, num_classes)
        return log_probs

    @staticmethod
    def pack_target_hidden_states(
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pack target frames to the front for each sample.

        CTCLoss assumes the valid time axis occupies the first input_length
        steps of each sequence. Our acoustic layout is [prompt | target | pad],
        so we need to gather only target frames and make them contiguous.
        """
        B, _, D = hidden_states.shape
        target_mask_bool = target_mask.bool()
        input_lengths = target_mask_bool.sum(dim=1).long()
        max_target_len = int(input_lengths.max().item()) if B > 0 else 0

        packed_hidden = hidden_states.new_zeros(B, max_target_len, D)
        for i in range(B):
            cur_hidden = hidden_states[i][target_mask_bool[i]]
            packed_hidden[i, :cur_hidden.shape[0]] = cur_hidden

        return packed_hidden, input_lengths

    def loss(
        self,
        hidden_states: torch.Tensor,
        target_mask: torch.Tensor,
        ctc_targets: torch.Tensor,
        ctc_target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC alignment loss on target frames only.

        Args:
            hidden_states:       (B, T, dit_dim) last DiT block output
            target_mask:         (B, T) 1=target frame, 0=other
            ctc_targets:         (sum(target_lengths),) flattened target IDs
            ctc_target_lengths:  (B,) number of text chars per sample

        Returns:
            ctc_loss: scalar loss value
        """
        packed_hidden, input_lengths = self.pack_target_hidden_states(hidden_states, target_mask)
        if packed_hidden.shape[1] == 0:
            return hidden_states.new_zeros(())

        log_probs = self.forward(packed_hidden)  # (T_target_max, B, num_classes)

        # CTC loss — input_lengths tells CTC where valid frames are
        loss = self.ctc_loss(
            log_probs,           # (T, B, C)
            ctc_targets,         # (sum(target_lengths),)
            input_lengths,       # (B,) number of target frames
            ctc_target_lengths,  # (B,) number of text chars
        )

        return loss

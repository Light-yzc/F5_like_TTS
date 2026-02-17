"""
Training script for VAE-DiT TTS.

Usage:
    python train.py --config configs/model_medium.yaml --data_root data/processed/
"""

import os
import math
import argparse
from functools import partial

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models.dit import DiT
from models.text_encoder import TextConditioner
from models.duration_predictor import DurationPredictor
from models.flow_matching import FlowMatching
from data.dataset import TTSDataset, collate_fn


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_models(cfg: dict, device: torch.device):
    model_cfg = cfg["model"]

    # DiT
    dit = DiT(
        latent_dim=model_cfg["latent_dim"],
        dit_dim=model_cfg["dit_dim"],
        depth=model_cfg["depth"],
        heads=model_cfg["heads"],
        head_dim=model_cfg["head_dim"],
        ff_mult=model_cfg["ff_mult"],
    ).to(device)

    # Text Conditioner (T5 encoder frozen + projector trainable)
    text_cond = TextConditioner(
        model_name=model_cfg["text_encoder_name"],
        text_dim=model_cfg["text_encoder_dim"],
        dit_dim=model_cfg["dit_dim"],
        freeze=model_cfg["freeze_text_encoder"],
    ).to(device)

    # Duration Predictor
    dur_pred = DurationPredictor(
        text_dim=model_cfg["text_encoder_dim"],
        hidden_dim=model_cfg["duration_hidden_dim"],
        num_layers=model_cfg["duration_num_layers"],
        latent_rate=cfg["audio"]["latent_rate"],
    ).to(device)

    # Flow Matching (not a nn.Module, just utility)
    flow = FlowMatching(
        cfg_dropout_rate=model_cfg["cfg_dropout_rate"],
        default_cfg_scale=model_cfg["default_cfg_scale"],
        default_infer_steps=model_cfg["default_infer_steps"],
    )

    return dit, text_cond, dur_pred, flow


def train(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["training"]
    audio_cfg = cfg["audio"]

    print(f"Device: {device}")

    # Build models
    dit, text_cond, dur_pred, flow = build_models(cfg, device)
    print(f"DiT parameters: {dit.num_params / 1e6:.1f}M (trainable: {dit.num_trainable_params / 1e6:.1f}M)")

    # Tokenizer (for text encoding)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder_name"])

    # Dataset
    dataset = TTSDataset(
        data_root=args.data_root,
        latent_rate=audio_cfg["latent_rate"],
        min_duration_sec=audio_cfg["min_duration_sec"],
        max_duration_sec=audio_cfg["max_duration_sec"],
        prompt_ratio_min=audio_cfg["prompt_ratio_min"],
        prompt_ratio_max=audio_cfg["prompt_ratio_max"],
    )
    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_with_tokenizer,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples")

    # Optimizer (only trainable params)
    trainable_params = (
        list(dit.parameters())
        + list(text_cond.projector.parameters())
        + list(dur_pred.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    scaler = GradScaler(enabled=train_cfg.get("fp16", True))

    # Null condition for CFG training
    null_text_kv = torch.zeros(1, 1, cfg["model"]["dit_dim"], device=device)

    # Training loop
    global_step = 0
    dit.train()
    text_cond.projector.train()
    dur_pred.train()

    print("Starting training...")
    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            # Move to device
            prompt_latent = batch["prompt_latent"].to(device)
            target_latent = batch["target_latent"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_frames = batch["target_frames"].to(device)

            with autocast(enabled=train_cfg.get("fp16", True)):
                # Text encoding
                text_kv, text_mask = text_cond(input_ids, attention_mask)

                # Expand null condition to batch size
                null_kv = null_text_kv.expand(prompt_latent.shape[0], -1, -1)

                # Flow matching loss
                fm_losses = flow.compute_loss(
                    dit, prompt_latent, target_latent,
                    text_kv, text_mask, null_kv,
                )

                # Duration predictor loss (on frozen text features)
                with torch.no_grad():
                    text_features_for_dur = text_cond.encode_text(input_ids, attention_mask)
                dur_loss = dur_pred.loss(text_features_for_dur, attention_mask, target_frames)

                # Total loss
                loss = fm_losses["loss"] + 0.1 * dur_loss

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # Logging
            if global_step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {global_step}/{max_steps} | "
                    f"loss={loss.item():.4f} | "
                    f"fm={fm_losses['mse'].item():.4f} | "
                    f"dur={dur_loss.item():.4f} | "
                    f"lr={lr:.2e}"
                )

            # Save checkpoint
            if global_step % 10000 == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "dit": dit.state_dict(),
                    "text_projector": text_cond.projector.state_dict(),
                    "dur_pred": dur_pred.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "config": cfg,
                }, os.path.join(ckpt_dir, "checkpoint.pt"))
                print(f"Saved checkpoint at step {global_step}")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE-DiT TTS")
    parser.add_argument("--config", type=str, default="configs/model_medium.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)

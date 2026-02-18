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
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models.dit_only_self_attn import DiTSelfAttnOnly as DiT
from models.text_encoder import TextConditioner
from models.duration_predictor import DurationPredictor
from models.flow_matching import FlowMatching
from data.dataset import TTSDataset, collate_fn
from inference import inference
from models.vae import load_vae, vae_encode, vae_decode

import gc
import bitsandbytes as bnb
import wandb


def load_config(path: str) -> dict:
    with open(path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_models(cfg: dict, device: torch.device):
    model_cfg = cfg["model"]

    # DiT
    dit = DiT(
        latent_dim=model_cfg["latent_dim"],
        dit_dim=model_cfg["dit_dim"],
        text_dim=model_cfg["text_encoder_dim"],
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
    wandb.login()
    # Use a stable run_id derived from config path so resume always continues the same run
    import hashlib
    _run_id = hashlib.md5(os.path.abspath(args.config).encode()).hexdigest()[:8]
    wandb.init(
        project="vae_dit_tts",
        id=_run_id,
        resume="allow",
        config=cfg,
    )
    print(f"Device: {device}")

    # Build models
    dit, text_cond, dur_pred, flow = build_models(cfg, device)
    if train_cfg.get("gradient_checkpointing", False):
        dit.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")
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
    # optimizer = torch.optim.AdamW(
    #     trainable_params,
    #     lr=train_cfg["learning_rate"],
    #     weight_decay=train_cfg["weight_decay"],
    # )
    optimizer = bnb.optim.AdamW8bit(
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

    # Resume from checkpoint
    global_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        dit.load_state_dict(ckpt["dit"])
        text_cond.projector.load_state_dict(ckpt["text_projector"])
        dur_pred.load_state_dict(ckpt["dur_pred"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt["global_step"]
        print(f"Resumed at step {global_step}")
        del ckpt
        torch.cuda.empty_cache()

    # Training loop
    dit.train()
    text_cond.projector.train()
    dur_pred.train()

    print("Starting training...")
    progress_bar = tqdm(total=max_steps, initial=global_step, desc="Training")
    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            # Move to device
            latent = batch["latent"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_frames = batch["target_frames"].to(device)

            # Check for NaN/Inf in input data
            if torch.isnan(latent).any() or torch.isinf(latent).any():
                print(f"[Step {global_step}] WARNING: NaN/Inf in latent, skipping batch")
                continue

            with autocast(enabled=train_cfg.get("fp16", True)):
                # Text encoding
                text_kv, text_mask = text_cond(input_ids, attention_mask)

                # Expand null condition to batch size
                null_kv = null_text_kv.expand(latent.shape[0], -1, -1)

                # Flow matching loss
                fm_losses = flow.compute_loss(
                    dit, latent, prompt_mask, target_mask,
                    text_kv, text_mask, null_kv,
                    padding_mask=padding_mask,
                )

                # Duration predictor loss (on frozen text features)
                with torch.no_grad():
                    text_features_for_dur = text_cond.encode_text(input_ids, attention_mask)
                dur_loss = dur_pred.loss(text_features_for_dur, attention_mask, target_frames)

                # Total loss (dur_weight decays linearly: 0.1 → 0.01 over steps 2000~5000)
                dur_decay_start, dur_decay_end = 1800, 3500
                dur_weight_start, dur_weight_end = 0.1, 0.05
                if global_step < dur_decay_start:
                    dur_weight = dur_weight_start
                elif global_step > dur_decay_end:
                    dur_weight = dur_weight_end
                else:
                    progress = (global_step - dur_decay_start) / (dur_decay_end - dur_decay_start)
                    dur_weight = dur_weight_start + (dur_weight_end - dur_weight_start) * progress

                loss = fm_losses["loss"] + dur_weight * dur_loss

                # NaN check on loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Step {global_step}] WARNING: NaN/Inf loss detected!")
                    print(f"  fm_loss={fm_losses['loss'].item()}, dur_loss={dur_loss.item()}")
                    print(f"  latent stats: mean={latent.mean():.4f}, std={latent.std():.4f}, max={latent.abs().max():.4f}")
                    optimizer.zero_grad()
                    continue

                wandb.log({
                    "train/loss": loss.item(),
                    "train/fm_loss": fm_losses["loss"].item(),
                    "train/dur_loss": dur_loss.item(),
                    "train/dur_weight": dur_weight,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            progress_bar.update(1)

            # # Logging
            # if global_step % 100 == 0:
            #     lr = scheduler.get_last_lr()[0]
            #     progress_bar.set_postfix({
            #         "loss": f"{loss.item():.4f}",
            #         "fm": f"{fm_losses['mse'].item():.4f}",
            #         "dur": f"{dur_loss.item():.4f}",
            #         "lr": f"{lr:.2e}"
            #     })
            # Periodic inference with on-demand VAE
            if global_step % 500 == 0:
                try:
                    dit.eval()
                    # Load VAE → infer → unload
                    vae_cfg = cfg["vae"]
                    vae = load_vae(vae_cfg["model_path"], device=str(device), precision=vae_cfg.get("precision", "fp16"))
                    output_path = f"outputs/infer_step_{global_step}.wav"
                    os.makedirs("outputs", exist_ok=True)
                    inference(
                        dit, text_cond, dur_pred, flow, cfg,
                        prompt_audio_path="ref_audio.wav",
                        prompt_text="八点十分",
                        tts_text="春天有野草",
                        vae_encode_fn=lambda wav: vae_encode(vae, wav),
                        vae_decode_fn=lambda lat: vae_decode(vae, lat),
                        output_path=output_path,
                    )
                    # Log audio to wandb
                    if os.path.exists(output_path):
                        wandb.log({
                            "infer/audio": wandb.Audio(
                                output_path,
                                sample_rate=audio_cfg["sample_rate"],
                                caption=f"step_{global_step}",
                            ),
                        }, step=global_step)
                    # Unload VAE to free VRAM
                    del vae
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"[Step {global_step}] Inference failed: {e}")
                finally:
                    dit.train()

            # Save checkpoint
            if global_step % 2500 == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "dit": dit.state_dict(),
                    "text_projector": text_cond.projector.state_dict(),
                    "dur_pred": dur_pred.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "global_step": global_step,
                    "config": cfg,
                }, os.path.join(ckpt_dir, "checkpoint.pt"))
                # Save checkpoint
                print(f"Saved checkpoint at step {global_step}")
    
    progress_bar.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE-DiT TTS")
    parser.add_argument("--config", type=str, default="configs/model_medium.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume from")
    args = parser.parse_args()
    train(args)

"""
Training script for VAE-DiT TTS.

Usage:
    python train.py --config configs/model_medium.yaml --data_root data/processed/
"""

import warnings
import logging
# Suppress phonemizer's flood of "words count mismatch" warnings
warnings.filterwarnings("ignore", module="phonemizer")
logging.getLogger("phonemizer").setLevel(logging.ERROR)

import os
import math
import argparse
from functools import partial

import yaml
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.dit import DiT
from models.F5_like_text_encoder import F5TextEncoder, CharTokenizer
from models.duration_predictor import DurationPredictor
from models.duration_discriminator import DurationDiscriminator
from models.latent_discriminator import (
    MultiScaleLatentDiscriminator, hinge_d_loss, hinge_g_loss, feature_matching_loss,
)
from models.flow_matching import FlowMatching
from models.ctc_head import CTCAlignmentHead
from data.dataset import TTSDataset, collate_fn
from inference import inference
from models.vae import load_vae, vae_encode, vae_decode

import gc
import bitsandbytes as bnb
import wandb


def load_config(path: str) -> dict:
    with open(path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_models(cfg: dict, device: torch.device, char_tokenizer: CharTokenizer = None):
    model_cfg = cfg["model"]
    dit_dim = model_cfg["dit_dim"]

    # DiT (with cross-attention + RoPE + AdaLN gate)
    dit = DiT(
        latent_dim=model_cfg["latent_dim"],
        dit_dim=dit_dim,
        depth=model_cfg["depth"],
        heads=model_cfg["heads"],
        head_dim=model_cfg["head_dim"],
        ff_mult=model_cfg["ff_mult"],
    ).to(device)

    # F5-like Text Encoder (character-level, fully trainable)
    vocab_size = model_cfg.get("text_encoder_vocab_size", 16384)
    text_encoder = F5TextEncoder(
        vocab_size=max(vocab_size, char_tokenizer.vocab_size) if char_tokenizer else vocab_size,
        dim=dit_dim,
        depth=model_cfg.get("text_conv_depth", 4),
        kernel_size=model_cfg.get("text_conv_kernel", 7),
        ff_mult=model_cfg.get("text_conv_ff_mult", 4),
    ).to(device)

    # Duration Predictor (input dim = dit_dim, same as F5TextEncoder output)
    dur_pred = DurationPredictor(
        text_dim=dit_dim,
        hidden_dim=model_cfg["duration_hidden_dim"],
        num_layers=model_cfg["duration_num_layers"],
        nhead=model_cfg.get("duration_nhead", 8),
        num_conv_blocks=model_cfg.get("duration_conv_blocks", 3),
        conv_kernel=model_cfg.get("duration_conv_kernel", 7),
        latent_rate=cfg["audio"]["latent_rate"],
        noise_dim=model_cfg.get("duration_noise_dim", 64),
    ).to(device)

    # Flow Matching (not a nn.Module, just utility)
    flow = FlowMatching(
        cfg_dropout_rate=model_cfg["cfg_dropout_rate"],
        default_cfg_scale=model_cfg["default_cfg_scale"],
        default_infer_steps=model_cfg["default_infer_steps"],
    )

    return dit, text_encoder, dur_pred, flow


    
def train(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["training"]
    audio_cfg = cfg["audio"]
    wandb.login()
    # Only resume wandb run when resuming training from checkpoint
    wandb_kwargs = {"project": "vae_dit_tts_f5_text_enc_v3_fix_ctc_gan", "config": cfg}
    wandb.init(**wandb_kwargs)
    print(f"Device: {device}")

    # Load character vocabulary
    vocab_path = args.vocab or os.path.join(args.data_root, "char_vocab.json")
    if os.path.exists(vocab_path):
        char_tokenizer = CharTokenizer.load(vocab_path)
        print(f"Loaded char vocab from {vocab_path} ({char_tokenizer.vocab_size} chars)")
    else:
        print(f"Vocab not found at {vocab_path}, building from dataset...")
        from data.build_char_vocab import build_vocab
        vocab = build_vocab(args.data_root)
        import json
        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        char_tokenizer = CharTokenizer(vocab)
        print(f"Built and saved vocab ({char_tokenizer.vocab_size} chars)")

    # Build models
    dit, text_encoder, dur_pred, flow = build_models(cfg, device, char_tokenizer)
    if train_cfg.get("gradient_checkpointing", False):
        dit.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")
    print(f"DiT parameters: {dit.num_params / 1e6:.1f}M (trainable: {dit.num_trainable_params / 1e6:.1f}M)")

    # CTC Alignment Head
    ctc_head = CTCAlignmentHead(
        dit_dim=cfg["model"]["dit_dim"],
        vocab_size=char_tokenizer.vocab_size,
    ).to(device)
    print(f"CTC Head parameters: {sum(p.numel() for p in ctc_head.parameters()) / 1e3:.1f}K")
    print(f"TextEncoder parameters: {text_encoder.num_params / 1e6:.1f}M")

    # Duration Discriminator (VITS2-style adversarial training)
    dur_disc = DurationDiscriminator(
        text_dim=cfg["model"]["dit_dim"],
        hidden_dim=cfg["model"].get("duration_disc_hidden", 256),
    ).to(device)
    disc_optimizer = bnb.optim.AdamW8bit(
        dur_disc.parameters(), lr=2e-4, weight_decay=0.01,
    )
    dur_adv_weight = cfg["model"].get("duration_adv_weight", 0.1)
    dur_noise_dim = cfg["model"].get("duration_noise_dim", 64)
    print(f"DurationDiscriminator parameters: {sum(p.numel() for p in dur_disc.parameters()) / 1e3:.1f}K")
    print(f"Duration adversarial weight: {dur_adv_weight}, noise_dim: {dur_noise_dim}")

    # Latent Discriminator (Multi-Scale, for flow matching output)
    latent_disc = MultiScaleLatentDiscriminator(
        latent_dim=cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"].get("latent_disc_hidden", 256),
        num_scales=cfg["model"].get("latent_disc_scales", 3),
    ).to(device)
    latent_disc_optimizer = bnb.optim.AdamW8bit(
        latent_disc.parameters(), lr=2e-4, weight_decay=0.01,
    )
    latent_adv_weight = cfg["model"].get("latent_adv_weight", 0.1)
    latent_fm_weight = cfg["model"].get("latent_fm_weight", 2.0)
    latent_gan_t_threshold = cfg["model"].get("latent_gan_t_threshold", 0.5)
    print(f"LatentDiscriminator parameters: {sum(p.numel() for p in latent_disc.parameters()) / 1e3:.1f}K")
    print(f"Latent GAN: adv_weight={latent_adv_weight}, fm_weight={latent_fm_weight}, t_threshold={latent_gan_t_threshold}")

    # Dataset
    dataset = TTSDataset(
        data_root=args.data_root,
        latent_rate=audio_cfg["latent_rate"],
        min_duration_sec=audio_cfg["min_duration_sec"],
        max_duration_sec=audio_cfg["max_duration_sec"],
        prompt_ratio_min=audio_cfg["prompt_ratio_min"],
        prompt_ratio_max=audio_cfg["prompt_ratio_max"],
    )
    collate_with_tokenizer = partial(collate_fn, tokenizer=char_tokenizer)
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

    # Optimizer (all trainable: DiT + TextEncoder + DurationPredictor + CTC Head)
    trainable_params = (
        list(dit.parameters())
        + list(text_encoder.parameters())
        + list(dur_pred.parameters())
        + list(ctc_head.parameters())
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
    scaler = GradScaler('cuda', enabled=train_cfg.get("fp16", True))

    # Null condition for CFG training
    null_text_kv = torch.zeros(1, 1, cfg["model"]["dit_dim"], device=device)

    # Resume from checkpoint
    global_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        dit.load_state_dict(ckpt["dit"], strict=False)
        if "ctc_head" in ckpt:
            ctc_head.load_state_dict(ckpt["ctc_head"])
        if "text_encoder" in ckpt:
            text_encoder.load_state_dict(ckpt["text_encoder"])
        # Load dur_pred with shape-safe filtering (handles kernel/arch changes)
        if "dur_pred" in ckpt:
            saved = ckpt["dur_pred"]
            current = dur_pred.state_dict()
            compatible = {k: v for k, v in saved.items() if k in current and current[k].shape == v.shape}
            skipped = [k for k in saved if k not in compatible]
            dur_pred.load_state_dict(compatible, strict=False)
            if skipped:
                print(f"dur_pred: loaded {len(compatible)}, skipped {len(skipped)} (shape mismatch): {skipped}")
            else:
                print(f"dur_pred: loaded all {len(compatible)} params")
        if "dur_disc" in ckpt:
            try:
                dur_disc.load_state_dict(ckpt["dur_disc"])
                print("Loaded duration discriminator")
            except Exception as e:
                print(f"WARNING: Could not load dur_disc state: {e}")
        if "disc_optimizer" in ckpt:
            try:
                disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
            except Exception as e:
                print(f"WARNING: Could not load disc_optimizer state: {e}")
        if "latent_disc" in ckpt:
            try:
                latent_disc.load_state_dict(ckpt["latent_disc"])
                print("Loaded latent discriminator")
            except Exception as e:
                print(f"WARNING: Could not load latent_disc state: {e}")
        if "latent_disc_optimizer" in ckpt:
            try:
                latent_disc_optimizer.load_state_dict(ckpt["latent_disc_optimizer"])
            except Exception as e:
                print(f"WARNING: Could not load latent_disc_optimizer state: {e}")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as e:
            print(f"WARNING: Could not load optimizer state (likely due to new parameters): {e}")
            print("Continuing with fresh optimizer state for new parameters.")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt["global_step"]

        # Override LR from config (in case it changed)
        new_lr = train_cfg["learning_rate"]
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
            pg["initial_lr"] = new_lr
        # Rebuild scheduler with new base LR, fast-forward to current step
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        for _ in range(global_step):
            scheduler.step()
        print(f"Resumed at step {global_step}, LR overridden to {new_lr}")
        del ckpt
        torch.cuda.empty_cache()

    # Training loop
    dit.train()
    text_encoder.train()
    dur_pred.train()
    ctc_head.train()
    dur_disc.train()
    latent_disc.train()

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
            target_text_mask = batch.get("target_text_mask", attention_mask).to(device)
            target_frames = batch["target_frames"].to(device)

            # Check for NaN/Inf in input data
            if torch.isnan(latent).any() or torch.isinf(latent).any():
                print(f"[Step {global_step}] WARNING: NaN/Inf in latent, skipping batch")
                continue

            with autocast('cuda', enabled=train_cfg.get("fp16", True)):
                # Text encoding (F5-like, fully trainable)
                text_kv, text_mask = text_encoder(input_ids, attention_mask)

                # Expand null condition to batch size
                null_kv = null_text_kv.expand(latent.shape[0], -1, -1)

                # Decide which auxiliary losses to compute this step
                use_ctc = (global_step % 25 == 0)

                # Flow matching loss (with hidden states for CTC)
                fm_losses = flow.compute_loss(
                    dit, latent, prompt_mask, target_mask,
                    text_kv, text_mask, null_kv,
                    padding_mask=padding_mask,
                    return_hidden=True,
                )

                # ── Duration predictor (VITS2-style: log-domain + noise + GAN) ──
                B_cur = latent.shape[0]
                dur_noise = torch.randn(B_cur, dur_noise_dim, device=device)
                text_kv_det = text_kv.detach()

                # Duration predictor forward
                dur_pred_out = dur_pred(
                    text_kv_det, attention_mask, target_text_mask, noise=dur_noise,
                )
                dur_real = target_frames.float()

                # Duration MSE loss
                dur_mse_loss = F.mse_loss(dur_pred_out, dur_real)

                # Duration weight: decreased because raw frames MSE is much larger than log space
                dur_weight = 1e-3

            # ── Train Duration Discriminator ──
            disc_optimizer.zero_grad()
            with autocast('cuda', enabled=train_cfg.get("fp16", True)):
                d_real = dur_disc(text_kv_det, attention_mask, dur_real)
                d_fake = dur_disc(text_kv_det, attention_mask, dur_pred_out.detach())
                d_loss = (
                    F.relu(1.0 - d_real).mean()
                    + F.relu(1.0 + d_fake).mean()
                )
            scaler.scale(d_loss).backward()
            scaler.unscale_(disc_optimizer)
            torch.nn.utils.clip_grad_norm_(dur_disc.parameters(), 1.0)
            scaler.step(disc_optimizer)

            # ── Generator adversarial loss (dur_pred wants to fool discriminator) ──
            with autocast('cuda', enabled=train_cfg.get("fp16", True)):
                g_fake = dur_disc(text_kv_det, attention_mask, dur_pred_out)
                dur_adv_loss = -g_fake.mean()

                # Combined duration loss
                dur_loss = dur_mse_loss + dur_adv_weight * dur_adv_loss

                # CTC alignment loss (every 25 steps, decaying weight)
                # 278k~300k: 0.02 (stable), 300k~340k: 0.02→0.005, 340k+: 0.005
                ctc_decay_start, ctc_decay_end = 300000, 1240000
                ctc_weight_start, ctc_weight_end = 0.02, 0.005
                if global_step < ctc_decay_start:
                    ctc_weight = ctc_weight_start
                elif global_step > ctc_decay_end:
                    ctc_weight = 0.0
                else:
                    progress = (global_step - ctc_decay_start) / (ctc_decay_end - ctc_decay_start)
                    ctc_weight = ctc_weight_start + (ctc_weight_end - ctc_weight_start) * progress

                if use_ctc and ctc_weight > 0:
                    ctc_targets = batch["ctc_targets"].to(device)
                    ctc_target_lengths = batch["ctc_target_lengths"].to(device)
                    ctc_loss = ctc_head.loss(
                        fm_losses["hidden_states"],
                        target_mask,
                        ctc_targets,
                        ctc_target_lengths,
                    )
                else:
                    ctc_loss = torch.tensor(0.0, device=device)

                loss = fm_losses["loss"] + dur_weight * dur_loss + ctc_weight * ctc_loss

            # ── Train Latent Discriminator (only when t > threshold) ──
            latent_t = fm_losses["t"]
            gan_mask = latent_t > latent_gan_t_threshold
            latent_d_loss = torch.tensor(0.0, device=device)
            latent_g_adv = torch.tensor(0.0, device=device)
            latent_g_fm = torch.tensor(0.0, device=device)

            if gan_mask.any():
                x_real = fm_losses["x_0_real"][gan_mask]
                x_fake = fm_losses["x_0_hat"][gan_mask]
                tgt_mask_gan = target_mask[gan_mask]

                # Train D
                latent_disc_optimizer.zero_grad()
                with autocast('cuda', enabled=train_cfg.get("fp16", True)):
                    d_real_logits, d_real_fmaps = latent_disc(x_real.detach(), tgt_mask_gan)
                    d_fake_logits, d_fake_fmaps = latent_disc(x_fake.detach(), tgt_mask_gan)
                    latent_d_loss = hinge_d_loss(d_real_logits, d_fake_logits)
                scaler.scale(latent_d_loss).backward()
                scaler.unscale_(latent_disc_optimizer)
                torch.nn.utils.clip_grad_norm_(latent_disc.parameters(), 1.0)
                scaler.step(latent_disc_optimizer)

                # G adversarial + feature matching loss
                with autocast('cuda', enabled=train_cfg.get("fp16", True)):
                    g_fake_logits, g_fake_fmaps = latent_disc(
                        fm_losses["x_0_hat"][gan_mask], tgt_mask_gan,
                    )
                    latent_g_adv = hinge_g_loss(g_fake_logits)
                    # Feature matching: L1 between D's intermediate features
                    with torch.no_grad():
                        _, d_real_fmaps_fg = latent_disc(x_real.detach(), tgt_mask_gan)
                    latent_g_fm = feature_matching_loss(d_real_fmaps_fg, g_fake_fmaps)

                    latent_gan_loss = (
                        latent_adv_weight * latent_g_adv
                        + latent_fm_weight * latent_g_fm
                    )
                    loss = loss + latent_gan_loss

                # NaN check on loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Step {global_step}] WARNING: NaN/Inf loss detected!")
                    print(f"  fm_loss={fm_losses['loss'].item()}, dur_loss={dur_loss.item()}")
                    print(f"  latent stats: mean={latent.mean():.4f}, std={latent.std():.4f}, max={latent.abs().max():.4f}")
                    optimizer.zero_grad()
                    scaler.update() # NEED TO UPDATE SCALER EVEN ON SKIP!
                    continue

                wandb.log({
                    "train/loss": loss.item(),
                    "train/fm_loss": fm_losses["loss"].item(),
                    "train/dur_loss": dur_loss.item(),
                    "train/dur_mse": dur_mse_loss.item(),
                    "train/dur_adv": dur_adv_loss.item(),
                    "train/d_loss": d_loss.item(),
                    "train/dur_weight": dur_weight,
                    "train/ctc_loss": ctc_loss.item(),
                    "train/ctc_weight": ctc_weight,
                    "train/latent_d_loss": latent_d_loss.item(),
                    "train/latent_g_adv": latent_g_adv.item(),
                    "train/latent_g_fm": latent_g_fm.item(),
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
                    tts_texts = [
                        'ZH_杀死我的责任，你打算怎么负责呢？”纯白的吸血姬这么说着。',
                        'JA_ありがとうございます！なんだか、すごく嬉しいです、先輩！わたし、今日この時の気持ちをずっと忘れません。',
                        'EN_We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.'
                        ]
                    dit.eval()
                    text_encoder.eval()
                    # Load VAE → infer → unload
                    vae_cfg = cfg["vae"]
                    vae = load_vae(vae_cfg["model_path"], device=str(device), precision=vae_cfg.get("precision", "fp16"))
                    os.makedirs("outputs", exist_ok=True)
                    with torch.no_grad():
                        for text in tts_texts:
                            language, text = text.split('_', 1)
                            output_path = f"outputs/infer_step_{global_step}_{language}.wav"
                            inference(
                                dit, text_encoder, dur_pred, flow, cfg,
                                prompt_audio_path="ref_audio.mp3",
                                prompt_text="カルデア式ですね。わかります。",
                                tts_text=text,
                                prompt_language="JA",
                                tts_language=language,
                                char_tokenizer=char_tokenizer,
                                vae_encode_fn=lambda wav: vae_encode(vae, wav),
                                vae_decode_fn=lambda lat: vae_decode(vae, lat),
                                output_path=output_path,
                            )
                            # Log audio to wandb
                            if os.path.exists(output_path):
                                wandb.log({
                                    f"infer/audio_{language}": wandb.Audio(
                                        output_path,
                                        sample_rate=audio_cfg["sample_rate"],
                                        caption=f"step_{global_step}_{language}",
                                    ),
                                }, step=global_step)
                except Exception as e:
                    print(f"[Step {global_step}] Inference failed: {e}")
                finally:
                    # Aggressively free VRAM
                    if 'vae' in locals():
                        del vae
                    gc.collect()
                    torch.cuda.empty_cache()
                    dit.train()
                    text_encoder.train()

            # Save checkpoint
            if global_step % 2000 == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "dit": dit.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "dur_pred": dur_pred.state_dict(),
                    "dur_disc": dur_disc.state_dict(),
                    "latent_disc": latent_disc.state_dict(),
                    "ctc_head": ctc_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "disc_optimizer": disc_optimizer.state_dict(),
                    "latent_disc_optimizer": latent_disc_optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "global_step": global_step,
                    "config": cfg,
                    "vocab_path": vocab_path,
                }, os.path.join(ckpt_dir, "checkpoint.pt"))
                print(f"Saved checkpoint at step {global_step}")

                # Keep only the latest 2 checkpoints
                if not hasattr(args, 'saved_ckpts'):
                    args.saved_ckpts = []
                args.saved_ckpts.append(ckpt_dir)
                if len(args.saved_ckpts) > 1:
                    old_ckpt_dir = args.saved_ckpts.pop(0)
                    import shutil
                    if os.path.exists(old_ckpt_dir):
                        try:
                            shutil.rmtree(old_ckpt_dir)
                            print(f"Removed old checkpoint: {old_ckpt_dir}")
                        except Exception as e:
                            print(f"Failed to remove old checkpoint {old_ckpt_dir}: {e}")
    
    progress_bar.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE-DiT TTS")
    parser.add_argument("--config", type=str, default="configs/model_medium.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume from")
    parser.add_argument("--vocab", type=str, default=None, help="Path to char_vocab.json")
    args = parser.parse_args()
    train(args)

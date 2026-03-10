"""
LoRA fine-tuning script for VAE-DiT TTS.

Uses HuggingFace PEFT to inject LoRA adapters into DiT attention layers.
Only LoRA parameters are trained; base model weights are frozen.

Usage:
    python train_lora.py \
        --config configs/config_lora.yaml \
        --data_root data/processed/ \
        --base_ckpt checkpoints/checkpoint.pt
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
from torch.amp import GradScaler, autocast

from peft import LoraConfig, get_peft_model, PeftModel

from models.dit import DiT
from models.F5_like_text_encoder import F5TextEncoder, CharTokenizer
from models.duration_predictor import DurationPredictor
from models.flow_matching import FlowMatching
from data.dataset import TTSDatasetLoRA, collate_fn
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

    dit = DiT(
        latent_dim=model_cfg["latent_dim"],
        dit_dim=dit_dim,
        depth=model_cfg["depth"],
        heads=model_cfg["heads"],
        head_dim=model_cfg["head_dim"],
        ff_mult=model_cfg["ff_mult"],
    ).to(device)

    vocab_size = model_cfg.get("text_encoder_vocab_size", 16384)
    text_encoder = F5TextEncoder(
        vocab_size=max(vocab_size, char_tokenizer.vocab_size) if char_tokenizer else vocab_size,
        dim=dit_dim,
        depth=model_cfg.get("text_conv_depth", 4),
        kernel_size=model_cfg.get("text_conv_kernel", 7),
        ff_mult=model_cfg.get("text_conv_ff_mult", 4),
    ).to(device)

    dur_pred = DurationPredictor(
        text_dim=dit_dim,
        hidden_dim=model_cfg["duration_hidden_dim"],
        num_layers=model_cfg["duration_num_layers"],
        nhead=model_cfg.get("duration_nhead", 8),
        num_conv_blocks=model_cfg.get("duration_conv_blocks", 3),
        conv_kernel=model_cfg.get("duration_conv_kernel", 7),
        latent_rate=cfg["audio"]["latent_rate"],
    ).to(device)

    flow = FlowMatching(
        cfg_dropout_rate=model_cfg.get("cfg_dropout_rate", 0.15),
        default_cfg_scale=model_cfg["default_cfg_scale"],
        default_infer_steps=model_cfg["default_infer_steps"],
    )

    return dit, text_encoder, dur_pred, flow


def train_lora(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["training"]
    audio_cfg = cfg["audio"]
    lora_cfg = cfg["lora"]

    # --- WandB ---
    wandb.login()
    wandb.init(project="vae_dit_tts_lora", config=cfg)
    print(f"Device: {device}")

    # --- Load vocabulary ---
    vocab_path = args.vocab or os.path.join(args.data_root, "char_vocab.json")
    if os.path.exists(vocab_path):
        char_tokenizer = CharTokenizer.load(vocab_path)
        print(f"Loaded char vocab from {vocab_path} ({char_tokenizer.vocab_size} chars)")
    else:
        raise FileNotFoundError(f"Vocab not found at {vocab_path}. Provide --vocab.")

    # --- Build models & load base checkpoint ---
    dit, text_encoder, dur_pred, flow = build_models(cfg, device, char_tokenizer)

    print(f"Loading base checkpoint: {args.base_ckpt}")
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    dit.load_state_dict(ckpt["dit"], strict=False)
    if "text_encoder" in ckpt:
        text_encoder.load_state_dict(ckpt["text_encoder"])
    dur_pred.load_state_dict(ckpt["dur_pred"], strict=False)
    del ckpt
    torch.cuda.empty_cache()

    # --- Apply LoRA to DiT ---
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
    )
    dit = get_peft_model(dit, lora_config)
    dit.print_trainable_parameters()

    if train_cfg.get("gradient_checkpointing", False):
        dit.base_model.model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # --- Freeze TextEncoder & DurationPredictor ---
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in dur_pred.parameters():
        p.requires_grad = False
    text_encoder.eval()
    dur_pred.eval()
    print("TextEncoder and DurationPredictor frozen")

    # --- Dataset ---
    dataset = TTSDatasetLoRA(
        data_root=args.data_root,
        language=args.language,
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

    # --- Optimizer (only LoRA params) ---
    trainable_params = [p for p in dit.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # --- Scheduler ---
    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- AMP ---
    scaler = GradScaler('cuda', enabled=train_cfg.get("fp16", True))

    # --- Null condition for CFG ---
    null_text_kv = torch.zeros(1, 1, cfg["model"]["dit_dim"], device=device)

    # --- Resume from LoRA checkpoint ---
    global_step = 0
    if args.resume:
        print(f"Resuming LoRA from: {args.resume}")
        dit = PeftModel.from_pretrained(dit.base_model.model, args.resume)
        if os.path.exists(os.path.join(args.resume, "training_state.pt")):
            state = torch.load(os.path.join(args.resume, "training_state.pt"), map_location=device)
            global_step = state.get("global_step", 0)
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception as e:
                print(f"WARNING: Could not load optimizer state: {e}")
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            del state
        print(f"Resumed at step {global_step}")
        torch.cuda.empty_cache()

    # --- Training loop ---
    dit.train()
    print("Starting LoRA training...")
    progress_bar = tqdm(total=max_steps, initial=global_step, desc="LoRA Training")

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

            # Check for NaN/Inf
            if torch.isnan(latent).any() or torch.isinf(latent).any():
                print(f"[Step {global_step}] WARNING: NaN/Inf in latent, skipping")
                continue

            with autocast('cuda', enabled=train_cfg.get("fp16", True)):
                # Text encoding (frozen)
                with torch.no_grad():
                    text_kv, text_mask = text_encoder(input_ids, attention_mask)

                # Expand null condition
                null_kv = null_text_kv.expand(latent.shape[0], -1, -1)

                # Flow matching loss
                fm_losses = flow.compute_loss(
                    dit, latent, prompt_mask, target_mask,
                    text_kv, text_mask, null_kv,
                    padding_mask=padding_mask,
                )
                loss = fm_losses["loss"]

                # NaN check
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Step {global_step}] WARNING: NaN/Inf loss!")
                    optimizer.zero_grad()
                    continue

                wandb.log({
                    "train/loss": loss.item(),
                    "train/fm_loss": fm_losses["loss"].item(),
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

            # --- Periodic inference ---
            if global_step % train_cfg.get("infer_every", 500) == 0:
                try:
                    tts_texts = [
                        'JA_口を吸うたびに見つめ合い、たまらずにまた重なる',
                    ]
                    dit.eval()
                    vae_cfg = cfg["vae"]
                    vae = load_vae(vae_cfg["model_path"], device=str(device), precision=vae_cfg.get("precision", "fp16"))
                    os.makedirs("outputs_lora", exist_ok=True)
                    with torch.no_grad():
                        for text in tts_texts:
                            language, text = text.split('_', 1)
                            output_path = f"outputs_lora/lora_step_{global_step}_{language}.wav"
                            inference(
                                dit, text_encoder, dur_pred, flow, cfg,
                                prompt_audio_path="/content/F5_like_TTS/lora_ref_audio.wav",
                                prompt_text="リラックスせんと、眠れんよ?",
                                tts_text=text,
                                prompt_language="JA",
                                tts_language=language,
                                char_tokenizer=char_tokenizer,
                                vae_encode_fn=lambda wav: vae_encode(vae, wav),
                                vae_decode_fn=lambda lat: vae_decode(vae, lat),
                                output_path=output_path,
                            )
                            if os.path.exists(output_path):
                                wandb.log({
                                    f"infer/audio_{language}": wandb.Audio(
                                        output_path,
                                        sample_rate=audio_cfg["sample_rate"],
                                        caption=f"lora_step_{global_step}_{language}",
                                    ),
                                }, step=global_step)
                except Exception as e:
                    print(f"[Step {global_step}] Inference failed: {e}")
                finally:
                    if 'vae' in locals():
                        del vae
                    gc.collect()
                    torch.cuda.empty_cache()
                    dit.train()

            # --- Save LoRA checkpoint ---
            if global_step % train_cfg.get("save_every", 2000) == 0:
                save_dir = os.path.join(args.output_dir, f"lora_step_{global_step}")
                dit.save_pretrained(save_dir)
                # Save training state separately
                torch.save({
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }, os.path.join(save_dir, "training_state.pt"))
                print(f"Saved LoRA checkpoint at step {global_step}")

                # Keep only latest N checkpoints
                if not hasattr(args, 'saved_ckpts'):
                    args.saved_ckpts = []
                args.saved_ckpts.append(save_dir)
                if len(args.saved_ckpts) > 2:
                    old_dir = args.saved_ckpts.pop(0)
                    import shutil
                    if os.path.exists(old_dir):
                        try:
                            shutil.rmtree(old_dir)
                            print(f"Removed old checkpoint: {old_dir}")
                        except Exception as e:
                            print(f"Failed to remove {old_dir}: {e}")

    progress_bar.close()
    # Final save
    final_dir = os.path.join(args.output_dir, "lora_final")
    dit.save_pretrained(final_dir)
    torch.save({
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, os.path.join(final_dir, "training_state.pt"))
    print(f"LoRA training complete! Final weights saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for VAE-DiT TTS")
    parser.add_argument("--config", type=str, default="configs/config_lora.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, required=True, help="Path to base model checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints_lora")
    parser.add_argument("--resume", type=str, default=None, help="Path to LoRA checkpoint dir to resume from")
    parser.add_argument("--vocab", type=str, default=None, help="Path to char_vocab.json")
    parser.add_argument("--language", type=str, default="JA", help="Language tag for G2P (ZH, JA, EN)")
    args = parser.parse_args()
    train_lora(args)

"""
Inference script for VAE-DiT TTS.

Usage:
    python inference.py \
        --checkpoint checkpoints/step_500000/checkpoint.pt \
        --prompt_audio prompt.wav \
        --prompt_text "参考音频的文字" \
        --tts_text "你好，今天天气真好" \
        --output output.wav
"""

import argparse
import importlib
import math
import os
import re
import torch
import torchaudio
from torch.amp import autocast

from models.dit import DiT
from models.F5_like_text_encoder import F5TextEncoder, CharTokenizer
from models.duration_predictor import DurationPredictor
from models.flow_matching import FlowMatching
# from utils.g2p import text_to_phonemes
from utils.g2p_ipa import text_to_phonemes_ipa as text_to_phonemes


# ─── Sentence splitting ─────────────────────────────────────────────
# Chinese/Japanese punctuation + English punctuation
_SPLIT_PATTERN = re.compile(r'(?<=[。！？；…，、,.!?;])')


def split_text(text: str, min_len: int = 8) -> list[str]:
    """
    Split long text by punctuation marks.

    Args:
        text:    input text, e.g. "你好，今天天气真好。我们出去玩吧！"
        min_len: minimum segment length; shorter segments are merged with
                 the previous one to avoid overly short fragments.

    Returns:
        list of text segments, e.g. ["你好，今天天气真好。", "我们出去玩吧！"]
    """
    raw_parts = _SPLIT_PATTERN.split(text)
    raw_parts = [p for p in raw_parts if p.strip()]

    if len(raw_parts) <= 1:
        return [text.strip()] if text.strip() else []

    # Merge short segments with the previous one
    merged = [raw_parts[0]]
    for part in raw_parts[1:]:
        if len(merged[-1]) < min_len:
            merged[-1] += part
        else:
            merged.append(part)

    # If the last segment is too short, merge it back
    if len(merged) > 1 and len(merged[-1]) < min_len:
        merged[-2] += merged[-1]
        merged.pop()

    return merged


def _ensure_torch_utils_loaded() -> None:
    """
    Work around environments where torch serialization is available but
    torch._utils has not been attached to the torch package yet.
    """
    if hasattr(torch, "_utils"):
        return
    try:
        importlib.import_module("torch._utils")
    except Exception as exc:
        raise RuntimeError(
            "PyTorch environment is incomplete: failed to import `torch._utils` before "
            "loading the checkpoint. In Colab, restart the runtime or reinstall matching "
            "`torch`, `torchvision`, and `torchaudio` versions."
        ) from exc


def load_checkpoint(
    ckpt_path: str,
    device: torch.device,
    vocab_path: str = None,
    vocab_path_override: str = None,
    lora_path: str = None,
):
    """Load checkpoint and reconstruct models.
    
    Args:
        vocab_path: path to char vocab JSON.
        vocab_path_override: backward-compatible alias for vocab_path.
        lora_path: optional path to PEFT LoRA adapter directory.
                   If provided, loads LoRA weights on top of the base DiT.
    """
    _ensure_torch_utils_loaded()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    dit_dim = model_cfg["dit_dim"]

    # Build models
    dit = DiT(
        latent_dim=model_cfg["latent_dim"],
        dit_dim=dit_dim,
        depth=model_cfg["depth"],
        heads=model_cfg["heads"],
        head_dim=model_cfg["head_dim"],
        ff_mult=model_cfg["ff_mult"],
        use_text_expand=model_cfg.get("use_text_expand", False),
        use_text_expand_pos_emb=model_cfg.get("use_text_expand_pos_emb", False),
        text_expand_pos_emb_scale=model_cfg.get("text_expand_pos_emb_scale", 1.0),
    ).to(device)
    dit.load_state_dict(ckpt["dit"], strict=False)

    # Apply LoRA if provided
    if lora_path is not None:
        from peft import PeftModel
        dit = PeftModel.from_pretrained(dit, lora_path)
        print(f"Loaded LoRA adapter from: {lora_path}")

    dit.eval()

    # Load char vocab
    resolved_vocab_path = vocab_path_override or vocab_path
    if resolved_vocab_path and os.path.exists(resolved_vocab_path):
        char_tokenizer = CharTokenizer.load(resolved_vocab_path)
    else:
        # Fallback: try common paths
        for p in ["data/char_vocab.json", "char_vocab.json"]:
            if os.path.exists(p):
                resolved_vocab_path = p
                break
        char_tokenizer = CharTokenizer.load(resolved_vocab_path) if resolved_vocab_path else CharTokenizer()

    text_encoder = F5TextEncoder(
        vocab_size=max(model_cfg.get("text_encoder_vocab_size", 16384),
                       char_tokenizer.vocab_size),
        dim=dit_dim,
        depth=model_cfg.get("text_conv_depth", 4),
        kernel_size=model_cfg.get("text_conv_kernel", 7),
        ff_mult=model_cfg.get("text_conv_ff_mult", 4),
        transformer_depth=model_cfg.get("text_transformer_depth", 0),
        transformer_heads=model_cfg.get("text_transformer_heads", 8),
        transformer_ff_mult=model_cfg.get("text_transformer_ff_mult", 2.5),
    ).to(device)
    if "text_encoder" in ckpt:
        incompatible = text_encoder.load_state_dict(ckpt["text_encoder"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "Loaded text encoder with compatibility mode: "
                f"missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
    text_encoder.eval()

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
    dur_pred.load_state_dict(ckpt["dur_pred"], strict=False)
    dur_pred.eval()

    flow = FlowMatching(
        default_cfg_scale=model_cfg["default_cfg_scale"],
        default_infer_steps=model_cfg["default_infer_steps"],
        sway_coef=model_cfg.get("sway_coef", -1.0),
    )

    return dit, text_encoder, dur_pred, flow, cfg, char_tokenizer


def _resolve_duration_controls(
    cfg: dict,
    duration_scale: float | None = None,
    duration_bias_sec: float | None = None,
) -> tuple[float, float]:
    model_cfg = cfg["model"]
    scale = duration_scale if duration_scale is not None else model_cfg.get("duration_infer_scale", 1.0)
    bias_sec = duration_bias_sec if duration_bias_sec is not None else model_cfg.get("duration_infer_bias_sec", 0.0)

    if scale <= 0:
        raise ValueError(f"duration_scale must be > 0, got {scale}")
    return float(scale), float(bias_sec)


def _resolve_infer_attn_controls(
    infer_attn_guidance: bool = False,
    infer_attn_strength: float = 1.5,
    infer_attn_sigma: float = 0.4,
    infer_attn_decay_power: float = 2.0,
    infer_attn_early_stop_ratio: float = 0.7,
) -> dict:
    if infer_attn_sigma <= 0:
        raise ValueError(f"infer_attn_sigma must be > 0, got {infer_attn_sigma}")
    if infer_attn_strength < 0:
        raise ValueError(f"infer_attn_strength must be >= 0, got {infer_attn_strength}")
    if infer_attn_decay_power < 0:
        raise ValueError(f"infer_attn_decay_power must be >= 0, got {infer_attn_decay_power}")
    if not (0.0 <= infer_attn_early_stop_ratio <= 1.0):
        raise ValueError(
            f"infer_attn_early_stop_ratio must be in [0, 1], got {infer_attn_early_stop_ratio}"
        )

    return {
        "infer_attn_guidance": bool(infer_attn_guidance),
        "infer_attn_strength": float(infer_attn_strength),
        "infer_attn_sigma": float(infer_attn_sigma),
        "infer_attn_decay_power": float(infer_attn_decay_power),
        "infer_attn_early_stop_ratio": float(infer_attn_early_stop_ratio),
    }


@torch.no_grad()
def inference(
    dit, text_encoder, dur_pred, flow, cfg,
    prompt_audio_path: str,
    prompt_text: str,
    tts_text: str,
    prompt_language: str = "ZH",
    tts_language: str = "ZH",
    char_tokenizer: CharTokenizer = None,
    vae_encode_fn=None,
    vae_decode_fn=None,
    output_path: str = "output.wav",
    duration: float = None,
    duration_scale: float = None,
    duration_bias_sec: float = None,
    cfg_scale: float = None,
    n_steps: int = None,
    seed: int = None,
    infer_attn_guidance: bool = False,
    infer_attn_strength: float = 1.5,
    infer_attn_sigma: float = 0.4,
    infer_attn_decay_power: float = 2.0,
    infer_attn_early_stop_ratio: float = 0.7,
):
    """
    Run TTS inference.

    Args:
        dit, text_encoder, dur_pred, flow: loaded models
        cfg: config dict
        prompt_audio_path: path to reference audio
        prompt_text: transcription of reference audio
        tts_text: text to synthesize
        vae_encode_fn: function(waveform) → latent
        vae_decode_fn: function(latent) → waveform
        output_path: where to save output audio
        duration: override duration in seconds (None = auto predict)
        cfg_scale: override CFG scale
        n_steps: override number of inference steps
        seed: random seed
    """
    device = next(dit.parameters()).device
    audio_cfg = cfg["audio"]
    latent_rate = audio_cfg["latent_rate"]
    sample_rate = audio_cfg["sample_rate"]
    infer_attn_kwargs = _resolve_infer_attn_controls(
        infer_attn_guidance=infer_attn_guidance,
        infer_attn_strength=infer_attn_strength,
        infer_attn_sigma=infer_attn_sigma,
        infer_attn_decay_power=infer_attn_decay_power,
        infer_attn_early_stop_ratio=infer_attn_early_stop_ratio,
    )

    if infer_attn_kwargs["infer_attn_guidance"]:
        print(
            "Inference attention guidance enabled: "
            f"kv_strength={infer_attn_kwargs['infer_attn_strength']:.3f}, "
            f"sigma={infer_attn_kwargs['infer_attn_sigma']:.3f}, "
            f"decay_power={infer_attn_kwargs['infer_attn_decay_power']:.3f}, "
            f"guided_ratio={infer_attn_kwargs['infer_attn_early_stop_ratio']:.3f}"
        )
        if getattr(dit, "use_text_expand", False):
            print("WARNING: infer_attn_guidance is ignored when use_text_expand=true.")

    with autocast('cuda', dtype=torch.float16):
        # --- 1. Encode prompt audio ---
        if vae_encode_fn is not None:
            wav, sr = torchaudio.load(prompt_audio_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            prompt_latent = vae_encode_fn(wav.unsqueeze(0).to(device))  # (1, T_prompt, D)
        else:
            # Placeholder: create dummy prompt latent for testing
            print("WARNING: No VAE encode function provided, using random prompt latent")
            prompt_latent = torch.randn(1, 3 * latent_rate, cfg["model"]["latent_dim"], device=device)

        # --- 2. Encode text (character-level) ---
        mapped_prompt_text = text_to_phonemes(prompt_text, prompt_language)
        mapped_tts_text = text_to_phonemes(tts_text, tts_language)
        combined_text = f"{mapped_prompt_text} [SEP] {mapped_tts_text}"
        
        if char_tokenizer is not None:
            tokens = char_tokenizer(combined_text, max_length=512)
        else:
            # Fallback: inline char tokenization
            fallback_tokenizer = CharTokenizer()
            tokens = fallback_tokenizer.batch_encode([combined_text], max_len=512)
            
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        # Calculate target_text_mask for inference
        target_text_mask = torch.zeros_like(attention_mask)
        prefix_text = f"{mapped_prompt_text} [SEP] "
        start_idx = char_tokenizer.encoded_length(prefix_text) if char_tokenizer is not None and hasattr(char_tokenizer, "encoded_length") else len((char_tokenizer or fallback_tokenizer).encode(prefix_text))
        target_text_mask[0, start_idx:] = attention_mask[0, start_idx:]

        text_kv, text_mask = text_encoder(input_ids, attention_mask)

        # Null condition for CFG
        null_text_kv = torch.zeros(1, 1, cfg["model"]["dit_dim"], device=device)
        null_text_mask = torch.ones(1, 1, device=device)

        # --- 3. Determine generation length ---
        if duration is not None:
            T_gen = int(duration * latent_rate)
        else:
            pred_frames = float(dur_pred.predict_frames(text_kv, attention_mask, target_text_mask).item())
            dur_scale, dur_bias_sec = _resolve_duration_controls(cfg, duration_scale, duration_bias_sec)
            bias_frames = int(round(dur_bias_sec * latent_rate))
            T_gen = int(math.ceil(pred_frames * dur_scale)) + bias_frames
            T_gen = max(latent_rate, T_gen)  # At least 1 second
            print(
                f"Predicted duration: {pred_frames / latent_rate:.2f}s ({pred_frames:.1f} frames)"
            )
            if dur_scale != 1.0 or bias_frames != 0:
                print(
                    f"Adjusted duration: {T_gen / latent_rate:.2f}s ({T_gen} frames) "
                    f"[scale={dur_scale:.3f}, bias_sec={dur_bias_sec:.3f}]"
                )

        # --- 4. Flow Matching sampling ---
        gen_latent = flow.sample(
            dit_model=dit,
            prompt_latent=prompt_latent,
            T_gen=T_gen,
            text_kv=text_kv,
            text_mask=text_mask,
            null_text_kv=null_text_kv,
            null_text_mask=null_text_mask,
            cfg_scale=cfg_scale,
            n_steps=n_steps,
            seed=seed,
            show_progress=True,
            target_text_mask=target_text_mask,
            **infer_attn_kwargs,
        )
        print(f"Generated latent shape: {gen_latent.shape}")

        # --- 5. Decode to waveform ---
        if vae_decode_fn is not None:
            waveform = vae_decode_fn(gen_latent)  # (1, 1, num_samples)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            torchaudio.save(output_path, waveform.cpu().float(), sample_rate)
            print(f"Saved output to: {output_path}")
        else:
            # Save raw latent for inspection
            torch.save(gen_latent.cpu(), output_path.replace(".wav", "_latent.pt"))
            print(f"WARNING: No VAE decode function. Saved raw latent to {output_path.replace('.wav', '_latent.pt')}")

    return gen_latent

@torch.no_grad()
def inference_long(
    dit, text_encoder, dur_pred, flow, cfg,
    prompt_audio_path: str,
    prompt_text: str,
    tts_text: str,
    prompt_language: str = "ZH",
    tts_language: str = "ZH",
    char_tokenizer: CharTokenizer = None,
    vae_encode_fn=None,
    vae_decode_fn=None,
    output_path: str = "output.wav",
    duration: float = None,
    duration_scale: float = None,
    duration_bias_sec: float = None,
    cfg_scale: float = None,
    n_steps: int = None,
    seed: int = None,
    min_split_len: int = 8,
    infer_attn_guidance: bool = False,
    infer_attn_strength: float = 1.5,
    infer_attn_sigma: float = 0.4,
    infer_attn_decay_power: float = 2.0,
    infer_attn_early_stop_ratio: float = 0.7,
):
    """
    Long-text TTS inference with automatic sentence splitting.

    Splits tts_text by punctuation, runs each segment independently
    (sharing the same prompt audio), and concatenates the output waveforms.

    If the text has no splittable punctuation or is short enough,
    falls back to single-segment inference().
    """
    segments = split_text(tts_text, min_len=min_split_len)

    if len(segments) <= 1:
        # No splitting needed
        return inference(
            dit, text_encoder, dur_pred, flow, cfg,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            tts_text=tts_text,
            prompt_language=prompt_language,
            tts_language=tts_language,
            char_tokenizer=char_tokenizer,
            vae_encode_fn=vae_encode_fn,
            vae_decode_fn=vae_decode_fn,
            output_path=output_path,
            duration=duration,
            duration_scale=duration_scale,
            duration_bias_sec=duration_bias_sec,
            cfg_scale=cfg_scale,
            n_steps=n_steps,
            seed=seed,
            infer_attn_guidance=infer_attn_guidance,
            infer_attn_strength=infer_attn_strength,
            infer_attn_sigma=infer_attn_sigma,
            infer_attn_decay_power=infer_attn_decay_power,
            infer_attn_early_stop_ratio=infer_attn_early_stop_ratio,
        )

    print(f"Split into {len(segments)} segments:")
    for i, seg in enumerate(segments):
        print(f"  [{i+1}] {seg}")

    device = next(dit.parameters()).device
    audio_cfg = cfg["audio"]
    sample_rate = audio_cfg["sample_rate"]

    # Encode prompt audio ONCE (shared across all segments)
    with autocast('cuda', dtype=torch.float16):
        if vae_encode_fn is not None:
            wav, sr = torchaudio.load(prompt_audio_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            prompt_latent = vae_encode_fn(wav.unsqueeze(0).to(device))
        else:
            latent_rate = audio_cfg["latent_rate"]
            prompt_latent = torch.randn(1, 3 * latent_rate, cfg["model"]["latent_dim"], device=device)

    # Generate each segment and collect waveforms
    waveforms = []
    all_latents = []

    for i, seg_text in enumerate(segments):
        print(f"\n--- Segment [{i+1}/{len(segments)}]: {seg_text} ---")

        # Per-segment duration: proportionally split if total duration specified
        seg_duration = None
        if duration is not None:
            # Distribute total duration by character count ratio
            ratio = len(seg_text) / len(tts_text)
            seg_duration = duration * ratio

        seg_latent = _inference_core(
            dit, text_encoder, dur_pred, flow, cfg,
            prompt_latent=prompt_latent,
            prompt_text=prompt_text,
            tts_text=seg_text,
            prompt_language=prompt_language,
            tts_language=tts_language,
            char_tokenizer=char_tokenizer,
            duration=seg_duration,
            duration_scale=duration_scale,
            duration_bias_sec=duration_bias_sec,
            cfg_scale=cfg_scale,
            n_steps=n_steps,
            seed=seed,
            infer_attn_guidance=infer_attn_guidance,
            infer_attn_strength=infer_attn_strength,
            infer_attn_sigma=infer_attn_sigma,
            infer_attn_decay_power=infer_attn_decay_power,
            infer_attn_early_stop_ratio=infer_attn_early_stop_ratio,
        )
        all_latents.append(seg_latent)

        if vae_decode_fn is not None:
            seg_wav = vae_decode_fn(seg_latent)
            if seg_wav.dim() == 3:
                seg_wav = seg_wav.squeeze(0)
            waveforms.append(seg_wav)

    # Concatenate waveforms
    if waveforms:
        full_wav = torch.cat(waveforms, dim=-1)  # concat along time
        torchaudio.save(output_path, full_wav.cpu().float(), sample_rate)
        print(f"\nSaved concatenated output ({len(segments)} segments) to: {output_path}")

    full_latent = torch.cat(all_latents, dim=1)
    return full_latent


def _inference_core(
    dit, text_encoder, dur_pred, flow, cfg,
    prompt_latent: torch.Tensor,
    prompt_text: str,
    tts_text: str,
    prompt_language: str = "ZH",
    tts_language: str = "ZH",
    char_tokenizer: CharTokenizer = None,
    duration: float = None,
    duration_scale: float = None,
    duration_bias_sec: float = None,
    cfg_scale: float = None,
    n_steps: int = None,
    seed: int = None,
    infer_attn_guidance: bool = False,
    infer_attn_strength: float = 1.5,
    infer_attn_sigma: float = 0.4,
    infer_attn_decay_power: float = 2.0,
    infer_attn_early_stop_ratio: float = 0.7,
) -> torch.Tensor:
    """Core inference logic (text encode → duration → sample). Returns gen_latent."""
    device = next(dit.parameters()).device
    audio_cfg = cfg["audio"]
    latent_rate = audio_cfg["latent_rate"]
    infer_attn_kwargs = _resolve_infer_attn_controls(
        infer_attn_guidance=infer_attn_guidance,
        infer_attn_strength=infer_attn_strength,
        infer_attn_sigma=infer_attn_sigma,
        infer_attn_decay_power=infer_attn_decay_power,
        infer_attn_early_stop_ratio=infer_attn_early_stop_ratio,
    )

    with autocast('cuda', dtype=torch.float16):
        mapped_prompt_text = text_to_phonemes(prompt_text, prompt_language)
        mapped_tts_text = text_to_phonemes(tts_text, tts_language)
        combined_text = f"{mapped_prompt_text} [SEP] {mapped_tts_text}"

        if char_tokenizer is not None:
            tokens = char_tokenizer(combined_text, max_length=512)
        else:
            fallback_tokenizer = CharTokenizer()
            tokens = fallback_tokenizer.batch_encode([combined_text], max_len=512)

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        target_text_mask = torch.zeros_like(attention_mask)
        prefix_text = f"{mapped_prompt_text} [SEP] "
        start_idx = char_tokenizer.encoded_length(prefix_text) if char_tokenizer is not None and hasattr(char_tokenizer, "encoded_length") else len((char_tokenizer or fallback_tokenizer).encode(prefix_text))
        target_text_mask[0, start_idx:] = attention_mask[0, start_idx:]

        text_kv, text_mask = text_encoder(input_ids, attention_mask)

        null_text_kv = torch.zeros(1, 1, cfg["model"]["dit_dim"], device=device)
        null_text_mask = torch.ones(1, 1, device=device)

        if duration is not None:
            T_gen = int(duration * latent_rate)
        else:
            pred_frames = float(dur_pred.predict_frames(text_kv, attention_mask, target_text_mask).item())
            dur_scale, dur_bias_sec = _resolve_duration_controls(cfg, duration_scale, duration_bias_sec)
            bias_frames = int(round(dur_bias_sec * latent_rate))
            T_gen = int(math.ceil(pred_frames * dur_scale)) + bias_frames
            T_gen = max(latent_rate, T_gen)
            print(
                f"Predicted duration: {pred_frames / latent_rate:.2f}s ({pred_frames:.1f} frames)"
            )
            if dur_scale != 1.0 or bias_frames != 0:
                print(
                    f"Adjusted duration: {T_gen / latent_rate:.2f}s ({T_gen} frames) "
                    f"[scale={dur_scale:.3f}, bias_sec={dur_bias_sec:.3f}]"
                )

        gen_latent = flow.sample(
            dit_model=dit,
            prompt_latent=prompt_latent,
            T_gen=T_gen,
            text_kv=text_kv,
            text_mask=text_mask,
            null_text_kv=null_text_kv,
            null_text_mask=null_text_mask,
            cfg_scale=cfg_scale,
            n_steps=n_steps,
            seed=seed,
            show_progress=True,
            target_text_mask=target_text_mask,
            **infer_attn_kwargs,
        )
        print(f"Generated latent shape: {gen_latent.shape}")

    return gen_latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE-DiT TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt_audio", type=str, required=True)
    parser.add_argument("--prompt_text", type=str, required=True)
    parser.add_argument("--prompt_language", type=str, default="ZH", help="Language of the prompt text (ZH, JA, EN)")
    parser.add_argument("--tts_text", type=str, required=True)
    parser.add_argument("--tts_language", type=str, default="ZH", help="Language of the TTS text (ZH, JA, EN)")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in seconds")
    parser.add_argument("--duration_scale", type=float, default=None, help="Multiply predicted duration before sampling")
    parser.add_argument("--duration_bias_sec", type=float, default=None, help="Add a fixed duration margin in seconds after scaling")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vocab", type=str, default=None, help="Path to char_vocab.json")
    parser.add_argument("--split", action="store_true", help="Split long text by punctuation")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--infer_attn_guidance", action="store_true", help="Enable inference-only target-text attention guidance")
    parser.add_argument("--infer_attn_strength", type=float, default=1.5, help="Max K/V boost of the sliding target-text spotlight")
    parser.add_argument("--infer_attn_sigma", type=float, default=0.4, help="Width of the sliding spotlight on the target-text axis")
    parser.add_argument("--infer_attn_decay_power", type=float, default=2.0, help="How quickly the K/V boost fades during the guided phase")
    parser.add_argument("--infer_attn_early_stop_ratio", type=float, default=0.7, help="Fraction of sampling steps that use the sliding text spotlight")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dit, text_encoder, dur_pred, flow, cfg, char_tokenizer = load_checkpoint(
        args.checkpoint, device, vocab_path=args.vocab, lora_path=args.lora,
    )

    infer_fn = inference_long if args.split else inference
    infer_fn(
        dit, text_encoder, dur_pred, flow, cfg,
        prompt_audio_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        tts_text=args.tts_text,
        prompt_language=args.prompt_language,
        tts_language=args.tts_language,
        char_tokenizer=char_tokenizer,
        output_path=args.output,
        duration=args.duration,
        duration_scale=args.duration_scale,
        duration_bias_sec=args.duration_bias_sec,
        cfg_scale=args.cfg_scale,
        n_steps=args.n_steps,
        seed=args.seed,
        infer_attn_guidance=args.infer_attn_guidance,
        infer_attn_strength=args.infer_attn_strength,
        infer_attn_sigma=args.infer_attn_sigma,
        infer_attn_decay_power=args.infer_attn_decay_power,
        infer_attn_early_stop_ratio=args.infer_attn_early_stop_ratio,
    )

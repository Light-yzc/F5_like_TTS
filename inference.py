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
import torch
import torchaudio
from torch.amp import autocast

from models.dit import DiT
from models.F5_like_text_encoder import F5TextEncoder, CharTokenizer
from models.duration_predictor import DurationPredictor
from models.flow_matching import FlowMatching
# from utils.g2p import text_to_phonemes
from utils.g2p_ipa import text_to_phonemes_ipa as text_to_phonemes

def load_checkpoint(ckpt_path: str, device: torch.device, vocab_path_override: str = None):
    """Load checkpoint and reconstruct models."""
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
    ).to(device)
    dit.load_state_dict(ckpt["dit"], strict=False)
    dit.eval()

    # Load char vocab
    import os
    vocab_path = ckpt.get("vocab_path", vocab_path_override)
    if vocab_path and os.path.exists(vocab_path):
        char_tokenizer = CharTokenizer.load(vocab_path)
    else:
        # Fallback: try common paths
        for p in ["data/char_vocab.json", "char_vocab.json"]:
            if os.path.exists(p):
                vocab_path = p
                break
        char_tokenizer = CharTokenizer.load(vocab_path) if vocab_path else CharTokenizer()

    text_encoder = F5TextEncoder(
        vocab_size=max(model_cfg.get("text_encoder_vocab_size", 16384),
                       char_tokenizer.vocab_size),
        dim=dit_dim,
        depth=model_cfg.get("text_conv_depth", 4),
        kernel_size=model_cfg.get("text_conv_kernel", 7),
        ff_mult=model_cfg.get("text_conv_ff_mult", 4),
    ).to(device)
    if "text_encoder" in ckpt:
        text_encoder.load_state_dict(ckpt["text_encoder"])
    text_encoder.eval()

    dur_pred = DurationPredictor(
        text_dim=dit_dim,
        hidden_dim=model_cfg["duration_hidden_dim"],
        num_layers=model_cfg["duration_num_layers"],
        nhead=model_cfg.get("duration_nhead", 8),
        num_conv_blocks=model_cfg.get("duration_conv_blocks", 3),
        conv_kernel=model_cfg.get("duration_conv_kernel", 7),
        latent_rate=cfg["audio"]["latent_rate"],
    ).to(device)
    dur_pred.load_state_dict(ckpt["dur_pred"], strict=False)
    dur_pred.eval()

    flow = FlowMatching(
        default_cfg_scale=model_cfg["default_cfg_scale"],
        default_infer_steps=model_cfg["default_infer_steps"],
        sway_coef=model_cfg.get("sway_coef", -1.0),
    )

    return dit, text_encoder, dur_pred, flow, cfg, char_tokenizer


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
    cfg_scale: float = None,
    n_steps: int = None,
    seed: int = None,
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

    with autocast('cuda', dtype=torch.float16):
        # --- 1. Encode prompt audio ---
        if vae_encode_fn is not None:
            wav, sr = torchaudio.load(prompt_audio_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            # Ensure mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
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
        # Because CharTokenizer is 1 char = 1 token, the target starts exactly
        # after mapped_prompt_text (len) + " [SEP] " (7 chars)
        start_idx = len(mapped_prompt_text) + 7
        target_text_mask[0, start_idx:] = attention_mask[0, start_idx:]

        text_kv, text_mask = text_encoder(input_ids, attention_mask)

        # Null condition for CFG
        null_text_kv = torch.zeros(1, 1, cfg["model"]["dit_dim"], device=device)
        null_text_mask = torch.ones(1, 1, device=device)

        # --- 3. Determine generation length ---
        if duration is not None:
            T_gen = int(duration * latent_rate)
        else:
            # Use duration predictor
            T_gen = int(dur_pred(text_kv, attention_mask, target_text_mask).item())
            T_gen = max(latent_rate, T_gen)  # At least 1 second
            print(f"Predicted duration: {T_gen / latent_rate:.2f}s ({T_gen} frames)")

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
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vocab", type=str, default=None, help="Path to char_vocab.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dit, text_encoder, dur_pred, flow, cfg, char_tokenizer = load_checkpoint(
        args.checkpoint, device, vocab_path_override=args.vocab,
    )

    inference(
        dit, text_encoder, dur_pred, flow, cfg,
        prompt_audio_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        tts_text=args.tts_text,
        prompt_language=args.prompt_language,
        tts_language=args.tts_language,
        char_tokenizer=char_tokenizer,
        output_path=args.output,
        duration=args.duration,
        cfg_scale=args.cfg_scale,
        n_steps=args.n_steps,
        seed=args.seed,
    )

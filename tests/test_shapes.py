"""
Shape verification test for VAE-DiT TTS models.

Tests that all model components produce correct output shapes
and that the full forward/backward pass works without errors.

Run:
    python tests/test_shapes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.dit import DiT, DiTBlock, RMSNorm, TimestepEmbedding, RotaryEmbedding
from models.duration_predictor import DurationPredictor
from models.flow_matching import FlowMatching


def test_rmsnorm():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64), f"RMSNorm shape mismatch: {out.shape}"
    print("✓ RMSNorm")


def test_rotary_embedding():
    rope = RotaryEmbedding(dim=64)
    cos, sin = rope(seq_len=100, device=torch.device("cpu"))
    assert cos.shape == (100, 32), f"RoPE cos shape mismatch: {cos.shape}"
    assert sin.shape == (100, 32), f"RoPE sin shape mismatch: {sin.shape}"
    print("✓ RotaryEmbedding")


def test_timestep_embedding():
    te = TimestepEmbedding(freq_dim=256, embed_dim=512)
    t = torch.rand(4)
    emb = te(t)
    assert emb.shape == (4, 512), f"TimestepEmbedding shape mismatch: {emb.shape}"
    print("✓ TimestepEmbedding")


def test_dit_block():
    block = DiTBlock(dim=256, heads=4, head_dim=64, ff_mult=2.0)
    B, T = 2, 50
    x = torch.randn(B, T, 256)
    time_emb = torch.randn(B, 256)
    text_kv = torch.randn(B, 20, 256)
    text_mask = torch.ones(B, 20)
    rope = RotaryEmbedding(64)
    cos, sin = rope(T, torch.device("cpu"))

    text_rope = RotaryEmbedding(64)
    text_cos, text_sin = text_rope(20, torch.device("cpu"))
    out = block(x, time_emb, text_kv, text_mask, cos, sin, None, text_cos, text_sin)
    assert out.shape == (B, T, 256), f"DiTBlock shape mismatch: {out.shape}"
    print("✓ DiTBlock")


def test_dit_full():
    latent_dim = 32
    dit_dim = 256
    depth = 4
    heads = 4
    head_dim = 64

    dit = DiT(
        latent_dim=latent_dim,
        dit_dim=dit_dim,
        depth=depth,
        heads=heads,
        head_dim=head_dim,
        ff_mult=2.0,
    )

    B = 2
    T_prompt = 20
    T_gen = 30
    T_total = T_prompt + T_gen
    L_text = 15

    x_t = torch.randn(B, T_total, latent_dim)
    mask = torch.cat([
        torch.ones(B, T_prompt),
        torch.zeros(B, T_gen),
    ], dim=1)
    timestep = torch.rand(B)
    text_kv = torch.randn(B, L_text, dit_dim)
    text_mask = torch.ones(B, L_text)
    padding_mask = torch.ones(B, T_total)  # no padding in test

    velocity = dit(x_t, mask, timestep, text_kv, text_mask, padding_mask=padding_mask)
    assert velocity.shape == (B, T_total, latent_dim), \
        f"DiT output shape mismatch: {velocity.shape}"
    print(f"✓ DiT (params: {dit.num_params / 1e6:.1f}M)")


def test_dit_backward():
    dit = DiT(latent_dim=16, dit_dim=128, depth=2, heads=2, head_dim=64, ff_mult=2.0)

    x_t = torch.randn(2, 30, 16)
    mask = torch.cat([torch.ones(2, 10), torch.zeros(2, 20)], dim=1)
    t = torch.rand(2)
    text_kv = torch.randn(2, 10, 128)
    text_mask = torch.ones(2, 10)
    padding_mask = torch.ones(2, 30)

    out = dit(x_t, mask, t, text_kv, text_mask, padding_mask=padding_mask)
    loss = out[:, 10:].pow(2).mean()  # Loss on generation region
    loss.backward()

    # Check gradients exist
    has_grad = sum(1 for p in dit.parameters() if p.grad is not None)
    total = sum(1 for p in dit.parameters())
    assert has_grad > 0, "No gradients found!"
    print(f"✓ DiT backward ({has_grad}/{total} params have gradients)")


def test_duration_predictor():
    dp = DurationPredictor(text_dim=128, hidden_dim=64, num_layers=1, nhead=4)
    text_feat = torch.randn(2, 15, 128)
    text_mask = torch.ones(2, 15)

    pred = dp(text_feat, text_mask)
    assert pred.shape == (2,), f"DurationPredictor shape mismatch: {pred.shape}"
    assert (pred > 0).all(), "DurationPredictor output should be positive"

    # Test loss
    gt_frames = torch.tensor([100.0, 75.0])
    loss = dp.loss(text_feat, text_mask, gt_frames)
    loss.backward()
    print(f"✓ DurationPredictor (pred={pred.detach().tolist()}, loss={loss.item():.4f})")


def test_flow_matching_loss():
    dit = DiT(latent_dim=16, dit_dim=128, depth=2, heads=2, head_dim=64, ff_mult=2.0)
    flow = FlowMatching(cfg_dropout_rate=0.5)

    B = 2
    T_prompt, T_target = 10, 20
    T_total = T_prompt + T_target

    # Build packed latent: [prompt | target]
    latent = torch.randn(B, T_total, 16)
    prompt_mask = torch.zeros(B, T_total)
    target_mask = torch.zeros(B, T_total)
    padding_mask = torch.ones(B, T_total)
    prompt_mask[:, :T_prompt] = 1.0
    target_mask[:, T_prompt:] = 1.0

    text_kv = torch.randn(B, 8, 128)
    text_mask = torch.ones(B, 8)
    null_kv = torch.zeros(B, 1, 128)

    losses = flow.compute_loss(
        dit, latent, prompt_mask, target_mask,
        text_kv, text_mask, null_kv,
        padding_mask=padding_mask,
    )
    assert "loss" in losses
    losses["loss"].backward()
    print(f"✓ FlowMatching.compute_loss (loss={losses['loss'].item():.4f})")


def test_flow_matching_sample():
    dit = DiT(latent_dim=16, dit_dim=128, depth=2, heads=2, head_dim=64, ff_mult=2.0)
    dit.eval()
    flow = FlowMatching(default_infer_steps=5, default_cfg_scale=2.0)

    prompt = torch.randn(1, 10, 16)
    text_kv = torch.randn(1, 8, 128)
    text_mask = torch.ones(1, 8)
    null_kv = torch.zeros(1, 1, 128)
    null_mask = torch.ones(1, 1)

    gen = flow.sample(
        dit, prompt, T_gen=20,
        text_kv=text_kv, text_mask=text_mask,
        null_text_kv=null_kv, null_text_mask=null_mask,
        n_steps=5, seed=42, show_progress=False,
    )
    assert gen.shape == (1, 20, 16), f"Sample shape mismatch: {gen.shape}"
    print("✓ FlowMatching.sample")


def test_end_to_end_pipeline():
    """Test the full pipeline: text_encode → dit → flow_sample."""
    latent_dim = 16
    dit_dim = 128

    dit = DiT(latent_dim=latent_dim, dit_dim=dit_dim, depth=2, heads=2,
              head_dim=64, ff_mult=2.0)
    dit.eval()
    flow = FlowMatching(default_infer_steps=3, default_cfg_scale=2.0)
    dur_pred = DurationPredictor(text_dim=dit_dim, hidden_dim=64, num_layers=1, nhead=4)

    # Simulate text encoder output
    text_kv = torch.randn(1, 10, dit_dim)
    text_mask = torch.ones(1, 10)
    null_kv = torch.zeros(1, 1, dit_dim)
    null_mask = torch.ones(1, 1)

    # Simulate prompt latent (3 seconds @ 25Hz = 75 frames)
    prompt_latent = torch.randn(1, 75, latent_dim)

    # Predict duration
    T_gen = int(dur_pred(text_kv, text_mask).item())
    T_gen = max(25, T_gen)  # At least 1 second

    # Generate
    gen_latent = flow.sample(
        dit, prompt_latent, T_gen=T_gen,
        text_kv=text_kv, text_mask=text_mask,
        null_text_kv=null_kv, null_text_mask=null_mask,
        n_steps=3, show_progress=False,
    )
    assert gen_latent.shape == (1, T_gen, latent_dim)
    print(f"✓ End-to-end pipeline (prompt={75}f + gen={T_gen}f = {75 + T_gen}f)")


if __name__ == "__main__":
    print("=" * 60)
    print("VAE-DiT TTS — Shape Verification Tests")
    print("=" * 60)

    test_rmsnorm()
    test_rotary_embedding()
    test_timestep_embedding()
    test_dit_block()
    test_dit_full()
    test_dit_backward()
    test_duration_predictor()
    test_flow_matching_loss()
    test_flow_matching_sample()
    test_end_to_end_pipeline()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

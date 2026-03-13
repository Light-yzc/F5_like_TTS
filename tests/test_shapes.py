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
from models.duration_discriminator import DurationDiscriminator
from models.latent_discriminator import (
    MultiScaleLatentDiscriminator, hinge_d_loss, hinge_g_loss, feature_matching_loss,
)
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
    dp = DurationPredictor(text_dim=128, hidden_dim=64, num_layers=1, nhead=4, noise_dim=32)
    text_feat = torch.randn(2, 15, 128)
    text_mask = torch.ones(2, 15)

    # Test log-domain output
    log_dur = dp(text_feat, text_mask)
    assert log_dur.shape == (2,), f"DurationPredictor shape mismatch: {log_dur.shape}"

    # Test with noise
    noise = torch.randn(2, 32)
    log_dur_noisy = dp(text_feat, text_mask, noise=noise)
    assert log_dur_noisy.shape == (2,), f"Noisy output shape mismatch: {log_dur_noisy.shape}"

    # Test predict_frames (should be positive after exp)
    frames = dp.predict_frames(text_feat, text_mask)
    assert (frames > 0).all(), f"predict_frames should be positive, got {frames}"

    # Test loss (log-domain MSE)
    gt_frames = torch.tensor([100.0, 75.0])
    loss = dp.loss(text_feat, text_mask, gt_frames, noise=noise)
    loss.backward()
    print(f"✓ DurationPredictor (log_dur={log_dur.detach().tolist()}, frames={frames.detach().tolist()}, loss={loss.item():.4f})")


def test_duration_discriminator():
    disc = DurationDiscriminator(text_dim=128, hidden_dim=64, num_layers=2)
    text_feat = torch.randn(2, 15, 128)
    text_mask = torch.ones(2, 15)
    log_duration = torch.tensor([4.5, 3.8])  # log(frames+1)

    logit = disc(text_feat, text_mask, log_duration)
    assert logit.shape == (2,), f"Discriminator output shape mismatch: {logit.shape}"

    # Test backward
    loss = F.binary_cross_entropy_with_logits(logit, torch.ones(2))
    loss.backward()
    has_grad = sum(1 for p in disc.parameters() if p.grad is not None)
    assert has_grad > 0, "No gradients in discriminator"
    print(f"✓ DurationDiscriminator (logit={logit.detach().tolist()}, loss={loss.item():.4f})")


def test_duration_gan_step():
    """Test a full GAN training step: D update + G adversarial loss."""
    dp = DurationPredictor(text_dim=128, hidden_dim=64, num_layers=1, nhead=4, noise_dim=32)
    disc = DurationDiscriminator(text_dim=128, hidden_dim=64, num_layers=2)

    text_feat = torch.randn(2, 15, 128)
    text_mask = torch.ones(2, 15)
    gt_frames = torch.tensor([100.0, 75.0])
    noise = torch.randn(2, 32)

    # Forward
    log_pred = dp(text_feat, text_mask, noise=noise)
    log_real = torch.log(gt_frames + 1)

    # Discriminator step
    d_real = disc(text_feat, text_mask, log_real)
    d_fake = disc(text_feat, text_mask, log_pred.detach())
    d_loss = (
        F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    )
    d_loss.backward()

    # Generator adversarial step
    g_fake = disc(text_feat, text_mask, log_pred)
    g_adv = F.binary_cross_entropy_with_logits(g_fake, torch.ones_like(g_fake))
    dur_mse = F.mse_loss(log_pred, log_real)
    g_loss = dur_mse + 0.1 * g_adv
    g_loss.backward()

    # Both should have gradients
    dp_grads = sum(1 for p in dp.parameters() if p.grad is not None)
    disc_grads = sum(1 for p in disc.parameters() if p.grad is not None)
    assert dp_grads > 0, "No gradients in duration predictor"
    assert disc_grads > 0, "No gradients in discriminator"
    print(f"✓ DurationGAN step (d_loss={d_loss.item():.4f}, g_loss={g_loss.item():.4f})")


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
    dur_pred = DurationPredictor(text_dim=dit_dim, hidden_dim=64, num_layers=1, nhead=4, noise_dim=32)

    # Simulate text encoder output
    text_kv = torch.randn(1, 10, dit_dim)
    text_mask = torch.ones(1, 10)
    null_kv = torch.zeros(1, 1, dit_dim)
    null_mask = torch.ones(1, 1)

    # Simulate prompt latent (3 seconds @ 25Hz = 75 frames)
    prompt_latent = torch.randn(1, 75, latent_dim)

    # Predict duration
    T_gen = int(dur_pred.predict_frames(text_kv, text_mask).item())
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


def test_ctc_head_forward():
    from models.ctc_head import CTCAlignmentHead
    ctc = CTCAlignmentHead(dit_dim=128, vocab_size=94)
    hidden = torch.randn(2, 50, 128)
    log_probs = ctc(hidden)
    assert log_probs.shape == (50, 2, 95), f"CTC log_probs shape mismatch: {log_probs.shape}"
    # Verify log_softmax outputs (should sum to ~1 in probability space)
    probs = log_probs.exp()
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "CTC probs don't sum to 1"
    print("✓ CTCAlignmentHead forward")


def test_ctc_head_loss():
    from models.ctc_head import CTCAlignmentHead
    ctc = CTCAlignmentHead(dit_dim=128, vocab_size=94)

    B, T = 2, 50
    hidden = torch.randn(B, T, 128, requires_grad=True)
    target_mask = torch.zeros(B, T)
    target_mask[:, 20:] = 1.0  # 30 target frames

    # CTC targets: 2 samples, 10 chars each
    ctc_targets = torch.randint(1, 94, (20,))  # avoid PAD=0
    ctc_target_lengths = torch.tensor([10, 10])

    loss = ctc.loss(hidden, target_mask, ctc_targets, ctc_target_lengths)
    assert loss.shape == (), f"CTC loss should be scalar, got {loss.shape}"
    assert not torch.isnan(loss), "CTC loss is NaN"
    loss.backward()
    assert hidden.grad is not None, "No gradient on hidden states"
    print(f"✓ CTCAlignmentHead loss (loss={loss.item():.4f})")


def test_dit_return_hidden():
    dit = DiT(latent_dim=16, dit_dim=128, depth=2, heads=2, head_dim=64, ff_mult=2.0)
    B, T_total, L_text = 2, 30, 10

    x_t = torch.randn(B, T_total, 16)
    mask = torch.cat([torch.ones(B, 10), torch.zeros(B, 20)], dim=1)
    t = torch.rand(B)
    text_kv = torch.randn(B, L_text, 128)
    text_mask = torch.ones(B, L_text)
    padding_mask = torch.ones(B, T_total)

    # Without return_hidden
    out = dit(x_t, mask, t, text_kv, text_mask, padding_mask=padding_mask)
    assert isinstance(out, torch.Tensor), "Without return_hidden should return tensor"

    # With return_hidden
    velocity, hidden, attn_w = dit(x_t, mask, t, text_kv, text_mask, padding_mask=padding_mask, return_hidden=True)
    assert velocity.shape == (B, T_total, 16), f"velocity shape: {velocity.shape}"
    assert hidden.shape == (B, T_total, 128), f"hidden shape: {hidden.shape}"
    assert attn_w is None, "attn_weights should be None when ap_layer_idx not set"
    print("✓ DiT return_hidden")


def test_flow_matching_with_ctc():
    dit = DiT(latent_dim=16, dit_dim=128, depth=2, heads=2, head_dim=64, ff_mult=2.0)
    flow = FlowMatching(cfg_dropout_rate=0.5)

    B, T_prompt, T_target = 2, 10, 20
    T_total = T_prompt + T_target

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
        return_hidden=True,
    )
    assert "hidden_states" in losses, "Missing hidden_states in return"
    assert losses["hidden_states"].shape == (B, T_total, 128), \
        f"hidden_states shape: {losses['hidden_states'].shape}"
    print("✓ FlowMatching.compute_loss with return_hidden")

def test_attention_prior_loss():
    from models.attention_prior_loss import AttentionPriorLoss
    ap = AttentionPriorLoss(sigma=0.4)
    B, H, T, L = 2, 4, 30, 10
    attn = torch.softmax(torch.randn(B, H, T, L), dim=-1)
    text_mask = torch.ones(B, L)
    target_mask = torch.cat([torch.zeros(B, 10), torch.ones(B, 20)], dim=1)

    loss = ap(attn, text_mask, target_mask)
    assert loss.shape == (), f"AP loss should be scalar, got {loss.shape}"
    assert loss.item() >= 0, "AP loss should be non-negative"
    # Diagonal attention should give lower loss than random
    diag_attn = torch.zeros(B, H, T, L)
    for t in range(T):
        l = min(int(t / T * L), L - 1)
        diag_attn[:, :, t, l] = 1.0
    diag_loss = ap(diag_attn, text_mask, target_mask)
    assert diag_loss.item() < loss.item(), "Diagonal attention should have lower loss"
    print(f"✓ AttentionPriorLoss (random={loss.item():.4f}, diag={diag_loss.item():.4f})")


def test_dit_attn_weights():
    dit = DiT(latent_dim=16, dit_dim=128, depth=4, heads=2, head_dim=64, ff_mult=2.0)
    B, T_total, L_text = 2, 30, 10

    x_t = torch.randn(B, T_total, 16)
    mask = torch.cat([torch.ones(B, 10), torch.zeros(B, 20)], dim=1)
    t = torch.rand(B)
    text_kv = torch.randn(B, L_text, 128)
    text_mask = torch.ones(B, L_text)
    padding_mask = torch.ones(B, T_total)

    # With ap_layer_idx=2 (middle of 4 layers)
    velocity, hidden, attn_weights = dit(
        x_t, mask, t, text_kv, text_mask,
        padding_mask=padding_mask,
        return_hidden=True,
        ap_layer_idx=2,
    )
    assert attn_weights is not None, "attn_weights should not be None"
    assert attn_weights.shape == (B, 2, T_total, L_text), f"attn_weights shape: {attn_weights.shape}"
    # Without ap_layer_idx
    velocity2 = dit(x_t, mask, t, text_kv, text_mask, padding_mask=padding_mask)
    assert isinstance(velocity2, torch.Tensor), "Without return_hidden should return tensor"
    print("✓ DiT ap_layer_idx returns attn_weights")


def test_flow_matching_with_ap():
    dit = DiT(latent_dim=16, dit_dim=128, depth=4, heads=2, head_dim=64, ff_mult=2.0)
    flow = FlowMatching(cfg_dropout_rate=0.5)

    B, T_prompt, T_target = 2, 10, 20
    T_total = T_prompt + T_target

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
        return_hidden=True,
        ap_layer_idx=2,
    )
    assert "attn_weights" in losses, "Missing attn_weights in return"
    assert losses["attn_weights"].shape == (B, 2, T_total, 8), \
        f"attn_weights shape: {losses['attn_weights'].shape}"
    print("✓ FlowMatching.compute_loss with ap_layer_idx")


def test_flow_matching_returns_x0_hat():
    dit = DiT(latent_dim=16, dit_dim=128, depth=2, heads=2, head_dim=64, ff_mult=2.0)
    flow = FlowMatching(cfg_dropout_rate=0.5)

    B, T_prompt, T_target = 2, 10, 20
    T_total = T_prompt + T_target

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
    assert "x_0_hat" in losses, "Missing x_0_hat"
    assert "x_0_real" in losses, "Missing x_0_real"
    assert "t" in losses, "Missing t"
    assert losses["x_0_hat"].shape == (B, T_total, 16)
    assert losses["x_0_real"].shape == (B, T_total, 16)
    assert losses["t"].shape == (B,)
    print("✓ FlowMatching returns x_0_hat, x_0_real, t")


def test_latent_discriminator():
    disc = MultiScaleLatentDiscriminator(latent_dim=16, hidden_dim=64, num_scales=3)
    x = torch.randn(2, 30, 16)  # (B, T, D)
    mask = torch.ones(2, 30)

    logits, fmaps = disc(x, mask)
    assert len(logits) == 3, f"Expected 3 scales, got {len(logits)}"
    assert len(fmaps) == 3, f"Expected 3 fmap lists, got {len(fmaps)}"
    for i, logit in enumerate(logits):
        assert logit.shape[0] == 2 and logit.shape[1] == 1, f"Scale {i} logit shape: {logit.shape}"
    print(f"✓ MultiScaleLatentDiscriminator (scales={len(logits)}, logit_shapes={[l.shape for l in logits]})")


def test_latent_gan_step():
    """Test full latent GAN step: D hinge loss + G adversarial + feature matching."""
    disc = MultiScaleLatentDiscriminator(latent_dim=16, hidden_dim=64, num_scales=2)
    x_real = torch.randn(2, 30, 16)
    x_fake = torch.randn(2, 30, 16)
    mask = torch.ones(2, 30)

    # D step
    d_real_logits, d_real_fmaps = disc(x_real, mask)
    d_fake_logits, d_fake_fmaps = disc(x_fake.detach(), mask)
    d_loss = hinge_d_loss(d_real_logits, d_fake_logits)
    d_loss.backward()

    # G step
    disc.zero_grad()
    g_logits, g_fmaps = disc(x_fake, mask)
    g_adv = hinge_g_loss(g_logits)
    with torch.no_grad():
        _, real_fmaps_ref = disc(x_real, mask)
    g_fm = feature_matching_loss(real_fmaps_ref, g_fmaps)
    g_loss = 0.1 * g_adv + 2.0 * g_fm
    g_loss.backward()

    assert not torch.isnan(d_loss), "D loss is NaN"
    assert not torch.isnan(g_loss), "G loss is NaN"
    print(f"✓ LatentGAN step (d_loss={d_loss.item():.4f}, g_adv={g_adv.item():.4f}, g_fm={g_fm.item():.4f})")


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
    test_duration_discriminator()
    test_duration_gan_step()
    test_flow_matching_loss()
    test_flow_matching_sample()
    test_end_to_end_pipeline()

    print("-" * 60)
    print("CTC Alignment Tests")
    print("-" * 60)
    test_ctc_head_forward()
    test_ctc_head_loss()
    test_dit_return_hidden()
    test_flow_matching_with_ctc()

    print("-" * 60)
    print("Attention Prior Tests")
    print("-" * 60)
    test_attention_prior_loss()
    test_dit_attn_weights()
    test_flow_matching_with_ap()

    print("-" * 60)
    print("Latent GAN Tests")
    print("-" * 60)
    test_flow_matching_returns_x0_hat()
    test_latent_discriminator()
    test_latent_gan_step()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

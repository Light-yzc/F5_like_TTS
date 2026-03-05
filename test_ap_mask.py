"""
Test: Verify Attention Prior Loss mask behavior.

Simulates a realistic scenario:
- prompt: 50 audio frames, 30 text tokens
- target: 100 audio frames, 70 text tokens
- Tests that AP loss penalizes wrong attention patterns correctly.
"""
import torch
from models.attention_prior_loss import AttentionPriorLoss

ap = AttentionPriorLoss(sigma=0.4)

B, H = 1, 1
T_prompt, T_target = 50, 100
L_prompt, L_target = 30, 70
T = T_prompt + T_target  # 150
L = L_prompt + L_target   # 100

# Masks
target_mask = torch.zeros(B, T)
target_mask[:, T_prompt:] = 1.0  # target frames: 50~149

text_mask = torch.ones(B, L)  # all text valid

print("=" * 60)
print("Attention Prior Loss — Mask Behavior Test")
print("=" * 60)
print(f"Audio: {T_prompt} prompt + {T_target} target = {T} total")
print(f"Text:  {L_prompt} prompt + {L_target} target = {L} total")
print()

# ================================================================
# Case 1: Target frames attend to TARGET text (diagonal) — GOOD
# ================================================================
attn_good = torch.zeros(B, H, T, L)
for t in range(T_prompt, T):
    # Map target frame t to target text position
    progress = (t - T_prompt) / T_target  # 0.0 ~ 1.0
    l = L_prompt + int(progress * L_target)
    l = min(l, L - 1)
    attn_good[:, :, t, l] = 1.0

loss_good = ap(attn_good, text_mask, target_mask)
print(f"Case 1 — Target帧 → Target文字 (对角线):  loss = {loss_good.item():.6f}")

# ================================================================
# Case 2: Target frames attend to PROMPT text — BAD (should be penalized)
# ================================================================
attn_bad_prompt = torch.zeros(B, H, T, L)
for t in range(T_prompt, T):
    # Target frames all attend to prompt text position 0
    attn_bad_prompt[:, :, t, 0] = 1.0

loss_bad_prompt = ap(attn_bad_prompt, text_mask, target_mask)
print(f"Case 2 — Target帧 → Prompt文字 (重复):    loss = {loss_bad_prompt.item():.6f}")

# ================================================================
# Case 3: Target frames stuck on SAME target text — BAD (repetition)
# ================================================================
attn_repeat = torch.zeros(B, H, T, L)
for t in range(T_prompt, T):
    # All target frames attend to the first target text token
    attn_repeat[:, :, t, L_prompt] = 1.0

loss_repeat = ap(attn_repeat, text_mask, target_mask)
print(f"Case 3 — Target帧 → 同一个Target字 (重复): loss = {loss_repeat.item():.6f}")

# ================================================================
# Case 4: Target frames skip middle text — BAD (skipping)
# ================================================================
attn_skip = torch.zeros(B, H, T, L)
for t in range(T_prompt, T):
    progress = (t - T_prompt) / T_target
    if progress < 0.3:
        l = L_prompt + int(progress / 0.3 * 10)  # first 10 tokens fast
    else:
        l = L_prompt + 60 + int((progress - 0.3) / 0.7 * 10)  # skip 50 tokens, last 10
    l = min(l, L - 1)
    attn_skip[:, :, t, l] = 1.0

loss_skip = ap(attn_skip, text_mask, target_mask)
print(f"Case 4 — 跳过中间50个字:                  loss = {loss_skip.item():.6f}")

# ================================================================
# Case 5: Prompt frames should not be penalized at all
# ================================================================
attn_prompt_only = torch.zeros(B, H, T, L)
attn_prompt_only[:, :, :T_prompt, :] = 1.0 / L  # prompt frames attend everywhere

loss_prompt = ap(attn_prompt_only, text_mask, target_mask)
print(f"Case 5 — Prompt帧乱关注 (应该=0):          loss = {loss_prompt.item():.6f}")

# ================================================================
# Summary
# ================================================================
print()
print("-" * 60)
print("Summary:")
print(f"  ✅ Good (diagonal):    {loss_good.item():.6f}")
print(f"  ❌ Bad (prompt text):  {loss_bad_prompt.item():.6f}  ({loss_bad_prompt.item()/max(loss_good.item(),1e-8):.1f}x worse)")
print(f"  ❌ Bad (repetition):   {loss_repeat.item():.6f}  ({loss_repeat.item()/max(loss_good.item(),1e-8):.1f}x worse)")
print(f"  ❌ Bad (skipping):     {loss_skip.item():.6f}  ({loss_skip.item()/max(loss_good.item(),1e-8):.1f}x worse)")
print(f"  ✅ Prompt frames:     {loss_prompt.item():.6f}  (should be 0)")

all_ok = (
    loss_good.item() < loss_bad_prompt.item()
    and loss_good.item() < loss_repeat.item()
    and loss_good.item() < loss_skip.item()
    and loss_prompt.item() == 0.0
)
print()
print("✅ All cases correct!" if all_ok else "❌ Some cases failed!")

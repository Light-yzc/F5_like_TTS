"""Quick smoke test for the new architecture changes."""
import sys
sys.path.insert(0, ".")

import torch
import yaml

def test_new_arch():
    # Load config
    with open("configs/model_tiny.yaml") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    dit_dim = model_cfg["dit_dim"]
    text_enc_dim = model_cfg.get("text_enc_dim", dit_dim)

    print(f"dit_dim={dit_dim}, text_enc_dim={text_enc_dim}")
    print(f"depth={model_cfg['depth']}, num_cross_attn={model_cfg.get('num_cross_attn', 1)}")

    # 1. Test F5TextEncoder
    from models.F5_like_text_encoder import F5TextEncoder

    text_enc = F5TextEncoder(
        vocab_size=128,
        dim=text_enc_dim,
        depth=model_cfg.get("text_conv_depth", 4),
        kernel_size=model_cfg.get("text_conv_kernel", 7),
        ff_mult=model_cfg.get("text_conv_ff_mult", 4),
        transformer_depth=model_cfg.get("text_transformer_depth", 0),
        transformer_heads=model_cfg.get("text_transformer_heads", 8),
        transformer_ff_mult=model_cfg.get("text_transformer_ff_mult", 2.5),
    )
    text_enc_params = sum(p.numel() for p in text_enc.parameters()) / 1e6
    print(f"\n✓ F5TextEncoder: {text_enc_params:.1f}M params, dim={text_enc.dim}")

    # Test forward
    B, L = 2, 20
    input_ids = torch.randint(0, 128, (B, L))
    attention_mask = torch.ones(B, L)
    text_kv, text_mask = text_enc(input_ids, attention_mask)
    assert text_kv.shape == (B, L, text_enc_dim), f"Expected {(B, L, text_enc_dim)}, got {text_kv.shape}"
    print(f"✓ F5TextEncoder forward: input={input_ids.shape} → output={text_kv.shape}")

    # 2. Test DiT
    from models.dit import DiT

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
        text_enc_dim=text_enc_dim,
        num_cross_attn=model_cfg.get("num_cross_attn", 1),
    )
    dit_params = sum(p.numel() for p in dit.parameters()) / 1e6
    print(f"\n✓ DiT: {dit_params:.1f}M params, depth={dit.depth}")
    print(f"  text_kv_proj: {'Yes' if dit.text_kv_proj is not None else 'No'}")
    print(f"  num_cross_attn per block: {dit.blocks[0].num_cross_attn}")

    # Test forward
    T = 50
    x_t = torch.randn(B, T, model_cfg["latent_dim"])
    mask = torch.ones(B, T)
    timestep = torch.rand(B)
    padding_mask = torch.ones(B, T)
    
    velocity = dit(x_t, mask, timestep, text_kv, text_mask, padding_mask=padding_mask)
    assert velocity.shape == (B, T, model_cfg["latent_dim"]), f"Expected {(B, T, model_cfg['latent_dim'])}, got {velocity.shape}"
    print(f"✓ DiT forward: x_t={x_t.shape}, text_kv={text_kv.shape} → velocity={velocity.shape}")

    # Test with return_hidden + ap_layer_indices
    vel, hidden, attn_w = dit(
        x_t, mask, timestep, text_kv, text_mask,
        padding_mask=padding_mask,
        return_hidden=True,
        ap_layer_indices=[7, 8],
    )
    print(f"✓ DiT forward (with hidden+attn): hidden={hidden.shape}, attn_weights={len(attn_w)} layers")
    for i, aw in enumerate(attn_w):
        print(f"  attn_weights[{i}]={aw.shape}")

    # 3. Summary
    total_params = dit_params + text_enc_params
    print(f"\n{'='*50}")
    print(f"TOTAL: DiT={dit_params:.1f}M + TextEnc={text_enc_params:.1f}M = {total_params:.1f}M params")
    print(f"Text conditioning passes per forward: {model_cfg['depth'] * model_cfg.get('num_cross_attn', 1)}")
    print(f"{'='*50}")
    print("\n✅ All shape checks passed!")


if __name__ == "__main__":
    test_new_arch()

import torch
from torch.amp import autocast
import time
from models.dit import DiT
import logging
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiT().to(device)

def test_speed(use_reentrant, preserve_rng_state):
    # Hack the checkpoint arguments in forward
    original_forward = DiT.forward
    
    # We patch the DiT.forward
    pass

model.enable_gradient_checkpointing()
model.train()
x_t = torch.randn(2, 512, 64, device=device)
mask = torch.ones(2, 512, device=device)
timestep = torch.rand(2, device=device)
text_kv = torch.randn(2, 128, 1024, device=device, requires_grad=True)

# Warmup
with autocast(device_type="cuda" if device=="cuda" else "cpu"):
    out = model(x_t, mask, timestep, text_kv)
    loss = out.sum()
loss.backward()

# measure
torch.cuda.synchronize() if device=="cuda" else None
t0 = time.time()
for _ in range(5):
    with autocast(device_type="cuda" if device=="cuda" else "cpu"):
        out = model(x_t, mask, timestep, text_kv)
        loss = out.sum()
    loss.backward()
torch.cuda.synchronize() if device=="cuda" else None
t1 = time.time()

print(f"Time: {t1-t0:.4f}s")

import sys, logging
sys.path.insert(0, sys.argv[1])
import numpy as np, scipy.sparse as sp, torch
from fedspam_common import SpamMLP, DPConfig, _dp_linear_layers, _dp_sgd_step, train_local_dp, calibrate_noise_multiplier, dp_steps_per_epoch
from opacus.accountants import RDPAccountant

torch.manual_seed(0)
m = SpamMLP(20, 8, dropout=0.0)
layers = _dp_linear_layers(m)
xb = torch.randn(16, 20) * 3; yb = (torch.rand(16) > 0.5).float()
for C in (0.05, 1.0, 1e6):
    dp = DPConfig(0.0, C, 1e-5)
    _dp_sgd_step(m, layers, xb, yb, dp, 16, torch.Generator().manual_seed(1))
    fast = torch.cat([p.grad.flatten() for p in m.parameters()])
    total = torch.zeros_like(fast); clipped_count = 0
    for i in range(16):
        m.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(m(xb[i:i+1]), yb[i:i+1], reduction="sum").backward()
        g = torch.cat([p.grad.flatten() for p in m.parameters()])
        clipped_count += int(g.norm() > C)
        total += g * min(1.0, C / (g.norm().item() + 1e-6))
    print(f"C={C:g}: max abs diff vs brute force {((fast - total/16).abs().max()).item():.2e}, emails clipped {clipped_count}/16")

# empty Poisson batch
dp = DPConfig(1.0, 1.0, 1e-5)
_dp_sgd_step(m, layers, torch.zeros(0, 20), torch.zeros(0), dp, 16, torch.Generator().manual_seed(1))
print("empty batch grad std (expect ~1/16=0.0625):", torch.cat([p.grad.flatten() for p in m.parameters()]).std().item())

# end-to-end small training run + accountant vs calibration
rng = np.random.default_rng(0)
X = sp.random(2000, 20, density=0.3, format="csr", dtype=np.float32, random_state=0)
w = rng.normal(size=20); y = (X @ w > 0).astype(np.int64)
log = logging.getLogger("t"); logging.basicConfig(level=logging.INFO, format="%(message)s")
sigma = calibrate_noise_multiplier(3.0, 1e-5, 64/2000, 3*dp_steps_per_epoch(2000, 64))
acc = RDPAccountant(); m2 = SpamMLP(20, 16, dropout=0.3)
train_local_dp(m2, X, y, epochs=3, batch_size=64, lr=1e-2, device="cpu", seed=1, log=log,
               dp=DPConfig(sigma, 1.0, 1e-5), accountant=acc)
print(f"calibrated sigma {sigma:.3f} -> spent epsilon {acc.get_epsilon(1e-5):.3f} (target 3.0)")

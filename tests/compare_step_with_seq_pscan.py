import torch

import sys

from poetry.console.commands import self

sys.path.append('..')
from mambafied.mamba import Mamba, MambaConfig

Bs, L, D, N = 2, 64, 7, 16
n_layers = 2

config = MambaConfig(d_model=D, n_layers=n_layers, use_cuda=False)
model = Mamba(config).to("mps")

# API for selective_scan() and selective_scan_seq()
# x : (Bs, L, ED)

# y : (Bs, L, ED)

x1 = torch.randn(Bs, L//2, D).to("mps") # x.requieres_grad = True
x2 = torch.randn(Bs, L//2, D).to("mps")


# Optimized Pscan Mode -----------------------------------------------------------------------------

cache_pscan = None
y_pscan_1, cache_pscan = model(x1, caches=cache_pscan)
y_pscan_2, cache_pscan = model(x2, caches=cache_pscan)
y_pscan = torch.cat([y_pscan_1, y_pscan_2], dim=1)



# Python Loop with Step ----------------------------------------------------------------------------

ED = config.d_inner
N = config.d_state
d_conv = config.d_conv
layer_cache = (torch.zeros(Bs, ED, N, dtype=x1.dtype, device=x1.device),
               torch.zeros(Bs, ED, d_conv - 1, dtype=x1.dtype, device=x1.device))
cache_seq = [layer_cache for _ in range(n_layers)]

y_seq = []
x_complete = torch.cat([x1, x2], dim=1)
for t in range(L):
    x_t = x_complete[:, t]
    y_t, cache_seq = model.step(x_t, caches=cache_seq)
    y_seq.append(y_t)
y_seq = torch.stack(y_seq, dim=1)

# Compare Results ----------------------------------------------------------------------------------

#print(y_pscan)
#print(y_seq)

for c_pscan, c_seq in zip(cache_pscan, cache_seq):
    h_pscan, inputs_pscan = c_pscan[0]
    h_seq, inputs_seq = c_seq[0]
    print(torch.allclose(h_pscan, h_seq, rtol=0.01))
    print(torch.allclose(inputs_pscan, inputs_seq, rtol=0.01))

print(torch.allclose(y_seq, y_pscan, rtol=0.01))

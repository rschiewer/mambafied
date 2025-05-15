import torch

import sys

from poetry.console.commands import self

sys.path.append('..')
from mambafied.mamba import Mamba, MambaConfig

Bs, L, D, N = 2, 64, 7, 16
n_layers = 2

config = MambaConfig(d_model=D, n_layers=n_layers, use_cuda=False)
model = Mamba(config).to("mps")
optim = torch.optim.Adam(model.parameters())

# API for selective_scan() and selective_scan_seq()
# x : (Bs, L, ED)

# y : (Bs, L, ED)

x1 = torch.randn(Bs, L//4, D).to("mps")
x2 = torch.randn(Bs, L//4, D).to("mps")
x3 = torch.randn(Bs, L//4, D).to("mps")
x4 = torch.randn(Bs, L//4, D).to("mps")
y = torch.randn(Bs, L, D).to("mps")

# Optimized Pscan Mode -----------------------------------------------------------------------------

# forward
cache_pscan = None
y_pscan_1, cache_pscan = model(x1, caches=cache_pscan)
y_pscan_2, cache_pscan = model(x2, caches=cache_pscan)
y_pscan_3, cache_pscan = model(x3, caches=cache_pscan)
y_pscan_4, cache_pscan = model(x4, caches=cache_pscan)
y_pscan = torch.cat([y_pscan_1, y_pscan_2, y_pscan_3, y_pscan_4], dim=1)

# backward
loss = torch.nn.functional.mse_loss(y_pscan, y, reduction='mean')
optim.zero_grad()
loss.backward()

# record gradients
grads_pscan = [p.grad.data.clone() for p in model.parameters()]

# Python Loop with Step ----------------------------------------------------------------------------

ED = config.d_inner
N = config.d_state
d_conv = config.d_conv
layer_cache = (torch.zeros(Bs, ED, N, dtype=x1.dtype, device=x1.device),
               torch.zeros(Bs, ED, d_conv - 1, dtype=x1.dtype, device=x1.device))
cache_seq = [layer_cache for _ in range(n_layers)]

x_complete = torch.cat([x1, x2, x3, x4], dim=1)
y_seq = []
for t in range(L):
    x_t = x_complete[:, t]
    y_t, cache_seq = model.step(x_t, caches=cache_seq)
    y_seq.append(y_t)
y_seq = torch.stack(y_seq, dim=1)

# backward
loss = torch.nn.functional.mse_loss(y_seq, y, reduction='mean')
optim.zero_grad()
loss.backward()

# record gradients
grads_seq = [p.grad.data.clone() for p in model.parameters()]

# Compare Results ----------------------------------------------------------------------------------

#print(y_pscan)
#print(y_seq)

for c_pscan, c_seq in zip(cache_pscan, cache_seq):
    h_pscan, inputs_pscan = c_pscan[0]
    h_seq, inputs_seq = c_seq[0]
    print(torch.allclose(h_pscan, h_seq, rtol=0.01))
    print(torch.allclose(inputs_pscan, inputs_seq, rtol=0.01))

print(torch.allclose(y_seq, y_pscan, rtol=0.01))

for grad_pscan, grad_seq in zip(grads_pscan, grads_seq):
    print(torch.allclose(grad_pscan, grad_seq, rtol=0.01))

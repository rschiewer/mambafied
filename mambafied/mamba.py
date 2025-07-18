import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from mambafied.pscan import pscan

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    use_asni: bool = False
    asni_time_independent: bool = False

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, caches=None):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        if caches is None:
            caches = [None for _ in self.layers]

        hs_last_layer = None
        for i, layer in enumerate(self.layers):
            x, hs_last_layer, caches[i] = layer(x, caches[i])

        return x, hs_last_layer, caches
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.use_asni = config.use_asni
        self.asni = ASNI(config.asni_time_independent) if self.use_asni else None

    def forward(self, x, cache=None):
        # x : (B, L, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, ED, d_conv-1)

        # output : (B, L, D)

        x_norm = self.norm(x)

        if self.use_asni:
            x_norm = self.asni(x_norm)

        output, hs, cache = self.mixer(x_norm, cache)
        output = output + x
        return output, hs, cache
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x, cache=None):
        # x : (B, L, D)
        
        # y : (B, L, D)

        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        if self.config.use_cuda:
            assert cache is None, 'CUDA implementation can\'t be used with cache'

        B, L, D = x.shape
        d_conv = self.config.d_conv

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        if cache is None:
            # first get the last d_conv input tokens, then process x
            ED = self.config.d_inner
            inputs = torch.zeros(B, ED, d_conv - 1, device=x.device, dtype=x.dtype)
            # don't take more than the available number of time steps from input to cache
            max_time_steps = min(L, d_conv - 1)
            inputs[:, :, -max_time_steps:] = x[:, :, -max_time_steps:]

            # now process x
            x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        else:
            # first get the last d_conv input tokens, then process x
            inputs = torch.cat([cache[1], x], dim=2)[:, :, -d_conv+1:]

            # now process x with cache
            x = self.conv1d(torch.cat([cache[1], x], dim=2))[:, :, d_conv-1:L+d_conv-1]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y, hs = self.ssm(x, z, cache=cache)

        if self.config.use_cuda:
            output = self.out_proj(y) # (B, L, D)
            return output # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        cache = (hs[:, -1], inputs)

        return output, hs, cache
    
    def ssm(self, x, z, cache=None):
        # x : (B, L, ED)

        # y : (B, L, ED)

        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        if self.config.use_cuda:
            assert cache is None, 'CUDA implementation can\'t be used with cache'

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y, hs = self.selective_scan(x, delta, A, B, C, D, cache=cache)
            else:
                y, hs = self.selective_scan_seq(x, delta, A, B, C, D, cache=cache)

        return y, hs
    
    def selective_scan(self, x, delta, A, B, C, D, cache=None):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        if cache is not None:
            # inject last invocation's h
            # prepend one time step of zeros to deltaA
            #deltaA = torch.nn.functional.pad(deltaA, (0, 0, 0, 0, 1, 0, 0, 0), mode='constant',
            #                                 value=0)
            # note that the contents of A_0 don't matter, since h_0 = 0
            A_0 = torch.zeros_like(deltaA[:, 0])
            # handcrafted padding that avoids the use of nn.functional.pad
            deltaA = torch.cat([A_0.unsqueeze(1), deltaA], dim=1)

            # prepend last invocation's h to BX
            h = cache[0]
            BX = torch.cat([h.unsqueeze(1), BX], dim=1)

            deltaA = deltaA.contiguous()
            BX = BX.contiguous()
        
        hs = pscan(deltaA, BX)
        # remove first padding time step
        if cache is not None:
            hs = hs[:, 1:]

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y, hs
    
    def selective_scan_seq(self, x, delta, A, B, C, D, cache=None):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        if cache is not None:
            h = cache[0]  # easily inject old h
        else:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y, hs
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output

# see https://arxiv.org/abs/1909.09819
# and https://github.com/BeyremKh/ASNI/blob/master/Simul_struc_drop_fast.ipynb
class ASNI(nn.Module):

    _sqrt_1 = torch.tensor(1.0).sqrt().requires_grad_(False)

    def __init__(self, time_independent: bool):
        super().__init__()
        self.time_independent = time_independent

    def forward(self, x: torch.Tensor):
        if self.time_independent:
            return self._forward_time_independent(x)
        else:
            return self._forward_time_dependent(x)

    # treat the complete time series that represent a batch item as a single data point
    def _forward_time_dependent(self, x: torch.Tensor):
        B, T, D = x.shape
        d_cov = T * D
        flat = x.reshape(B, d_cov)
        mean = flat.mean(dim=0, keepdim=True)
        flat_center = (flat - mean)
        cov = flat_center.t() @ flat_center / B
        cov += 1e-2 * torch.eye(d_cov, device=x.device)
        sample = MultivariateNormal(torch.zeros(d_cov, dtype=x.dtype, device=x.device), cov).sample()
        sample = sample.unsqueeze(0).expand(B, -1)
        ret = flat * (self._sqrt_1.to(x) * sample + torch.ones_like(flat))
        ret = ret.reshape(B, T, D)
        return ret

    # treat time as another batch dimension
    def _forward_time_independent(self, x: torch.Tensor):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        mean = flat.mean(dim=0, keepdim=True)
        flat_center = (flat - mean)
        cov = flat_center.t() @ flat_center / (B * T)
        cov += 1e-2 * torch.eye(D, device=x.device)
        sample = MultivariateNormal(torch.zeros(D, dtype=x.dtype, device=x.device), cov).sample()
        sample = sample.unsqueeze(0).expand(B * T, -1)
        ret = flat * (self._sqrt_1.to(x) * sample + torch.ones_like(flat))
        ret = ret.reshape(B, T, D)
        return ret


class GNI(nn.Module):
    def __init__(self, ghost_bs=32, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.ghost_bs = ghost_bs

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(B, -1)                    # [B, F], F = T*D

        with torch.no_grad():
            μ = flat.mean(dim=1, keepdim=True)  # [B, 1]
            σ2 = flat.var(dim=1, keepdim=True)

            idx = torch.randint(0, B, (self.ghost_bs, B), device=x.device)
            ghosts = flat[idx, torch.arange(B)] # [ghost_bs, B, F]

            # Collapse ghost-bs and feature dims to derive per-sample scalar stats:
            μg = ghosts.reshape(self.ghost_bs, B, -1).mean(dim=(0, 2), keepdim=True)  # [1,1] → broadcastable
            σg2 = ghosts.reshape(self.ghost_bs, B, -1).var(dim=(0, 2), unbiased=False, keepdim=True)

        shift = μg - μ                           # [B, F] due to broadcasting
        scale = torch.sqrt((σg2 + self.eps) / (σ2 + self.eps))

        out = (flat - shift) / scale            # [B, F]
        return out.reshape(B, T, D).contiguous()
from __future__ import annotations

import numpy as np
import torch

from .physics_satisfaction import _ExactHeatFEM


class HeatFEMTorchResidual:
    def __init__(self, z_mean, z_scale):
        fem = _ExactHeatFEM()
        self.elem_int = torch.as_tensor(fem.elem_int, dtype=torch.long)
        self.grads = torch.as_tensor(fem.elem_grads, dtype=torch.float32)
        self.quad_N = torch.as_tensor(fem.quad_N, dtype=torch.float32)
        self.quad_w = torch.as_tensor(fem.quad_w, dtype=torch.float32)
        x = fem.quad_xy
        modes = fem.modes
        basis = np.stack([np.sin(a*np.pi*x[...,0])*np.sin(b*np.pi*x[...,1])
                          for a,b in modes], axis=-1)
        self.basis = torch.as_tensor(basis, dtype=torch.float32)
        self.z_mean = torch.as_tensor(z_mean, dtype=torch.float32)
        self.z_scale = torch.as_tensor(z_scale, dtype=torch.float32)

    def __call__(self, y_standardized, q_physical):
        device = y_standardized.device
        z = y_standardized * self.z_scale.to(device) + self.z_mean.to(device)
        q = q_physical.to(device=device, dtype=y_standardized.dtype)
        elem_int = self.elem_int.to(device)
        safe = elem_int.clamp_min(0)
        tv = z[:, safe]
        tv = torch.where((elem_int >= 0)[None, :, :], tv, torch.zeros_like(tv))
        dtype = y_standardized.dtype
        grads = self.grads.to(device=device, dtype=dtype); N = self.quad_N.to(device=device, dtype=dtype); w = self.quad_w.to(device=device, dtype=dtype)
        gradT = torch.einsum("eai,bea->bei", grads, tv)
        Tg = torch.einsum("ega,bea->beg", N, tv)
        basis = self.basis.to(device=device, dtype=dtype)
        source = 10.0 * (1.0 + .05 * torch.einsum("egk,bk->beg", basis, q))
        gradterm = torch.einsum("eai,bei->bea", grads, gradT)
        local = w[None, ..., None] * ((1 + Tg*Tg)[..., None] * gradterm[:, :, None, :] * N[None, ...] +
                                      .1 * Tg[..., None]**3 * N[None, ...] - source[..., None] * N[None, ...])
        out = torch.zeros((z.shape[0], z.shape[1]), dtype=z.dtype, device=device)
        for a in range(3):
            idx = elem_int[:, a]
            good = idx >= 0
            out.index_add_(1, idx[good], local[:, good, :, a].sum(dim=2))
        return out

import math
from typing import Optional, Tuple
import torch
from .spectral import SpectralPDE

def _infer_hw_from_in_features(in_features: int) -> Optional[Tuple[int, int]]:
    s = int(math.isqrt(int(in_features)))
    if s * s == int(in_features):
        return (s, s)
    return None

def apply_spatial_pde(param: torch.Tensor, tensor: torch.Tensor, op: SpectralPDE, step_t: int,
                      linear_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    g = tensor

    # Conv kernels: smooth last two dims
    if param.ndim == 4 and g.ndim == 4:
        kH, kW = int(g.shape[-2]), int(g.shape[-1])
        if min(kH, kW) >= 3:
            gg = g.reshape(-1, kH, kW)
            uu = op.apply_2d_batch(gg, step_t=step_t)
            return uu.reshape_as(g)
        return g

    # Linear weights: infer square or use provided hw
    if param.ndim == 2 and g.ndim == 2:
        in_features = int(param.shape[1])
        hw = linear_hw or _infer_hw_from_in_features(in_features)
        if hw is not None and hw[0] * hw[1] == in_features:
            out = int(g.shape[0])
            gg = g.reshape(out, hw[0], hw[1])
            uu = op.apply_2d_batch(gg, step_t=step_t)
            return uu.reshape_as(g)

    return g

def _exp_rho_2d(H: int, W: int, alpha: float, device, dtype) -> torch.Tensor:
    yy = torch.arange(H, device=device, dtype=dtype)[:, None]
    xx = torch.arange(W, device=device, dtype=dtype)[None, :]
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return torch.exp(-float(alpha) * dist)

def apply_spatial_pde_saitoh(param: torch.Tensor, tensor: torch.Tensor, op: SpectralPDE, step_t: int,
                            rho_alpha: float = 0.15, eps: float = 1e-6,
                            linear_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    g = tensor

    if param.ndim == 4 and g.ndim == 4:
        kH, kW = int(g.shape[-2]), int(g.shape[-1])
        if min(kH, kW) >= 3:
            gg = g.reshape(-1, kH, kW)
            rho = _exp_rho_2d(kH, kW, rho_alpha, gg.device, gg.dtype)[None, :, :]
            num = op.apply_2d_batch(gg * rho, step_t=step_t)
            den = op.apply_2d_batch(rho.expand_as(gg), step_t=step_t).clamp_min(float(eps))
            uu = num / den
            return uu.reshape_as(g)
        return g

    if param.ndim == 2 and g.ndim == 2:
        in_features = int(param.shape[1])
        hw = linear_hw or _infer_hw_from_in_features(in_features)
        if hw is not None and hw[0] * hw[1] == in_features:
            out = int(g.shape[0])
            gg = g.reshape(out, hw[0], hw[1])
            rho = _exp_rho_2d(hw[0], hw[1], rho_alpha, gg.device, gg.dtype)[None, :, :]
            num = op.apply_2d_batch(gg * rho, step_t=step_t)
            den = op.apply_2d_batch(rho.expand_as(gg), step_t=step_t).clamp_min(float(eps))
            uu = num / den
            return uu.reshape_as(g)

    return g

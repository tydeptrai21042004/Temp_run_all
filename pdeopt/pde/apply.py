import math
from typing import Optional, Tuple

import torch
from .spectral import SpectralPDE


def _infer_square_hw(in_features: int) -> Optional[Tuple[int, int]]:
    s = int(math.isqrt(int(in_features)))
    if s * s == int(in_features):
        return (s, s)
    return None


def _infer_rect_hw(in_features: int, min_side: int = 8) -> Optional[Tuple[int, int]]:
    """
    Find a near-square factorization H*W=in_features with H,W >= min_side,
    minimizing |H-W|. Useful for Linear layers like 512 -> 16x32.

    NOTE: This is heuristic. For deep nets, prefer explicit linear_hw for "real geometry".
    """
    n = int(in_features)
    best = None
    best_gap = 10**18
    r = int(math.isqrt(n))
    for h in range(1, r + 1):
        if n % h != 0:
            continue
        w = n // h
        if h < min_side or w < min_side:
            continue
        gap = abs(h - w)
        if gap < best_gap:
            best_gap = gap
            best = (h, w)
    return best


def _pick_linear_hw(
    in_features: int,
    linear_hw: Optional[Tuple[int, int]],
    allow_rect_linear: bool,
    min_rect_side: int,
) -> Optional[Tuple[int, int]]:
    """
    Deep-learning friendly policy:

    - Default: DO NOT reshape Linear weights (no fake geometry).
      => Only smooth Linear if user explicitly provides linear_hw
         OR explicitly enables allow_rect_linear.

    - If allow_rect_linear=True and linear_hw is None:
        Try square first; if not possible, try near-square rectangle.
    """
    if linear_hw is not None:
        H, W = int(linear_hw[0]), int(linear_hw[1])
        return (H, W) if H * W == int(in_features) else None

    if not bool(allow_rect_linear):
        return None

    hw = _infer_square_hw(in_features)
    if hw is None:
        hw = _infer_rect_hw(in_features, min_side=int(min_rect_side))
    return hw


def apply_spatial_pde(
    param: torch.Tensor,
    tensor: torch.Tensor,
    op: SpectralPDE,
    step_t: int,
    linear_hw: Optional[Tuple[int, int]] = None,
    allow_rect_linear: bool = False,
    min_rect_side: int = 8,
    min_kernel_hw: int = 3,
) -> torch.Tensor:
    """
    Apply PDE smoothing on tensors that have a meaningful 2D geometry:
      - Conv kernels: smooth last two dims (kH, kW)
      - Linear weights: smooth only if geometry is real (linear_hw) or allow_rect_linear=True
    """
    x = tensor

    # ---- Conv kernels: [out, in, kH, kW] ----
    if param.ndim == 4 and x.ndim == 4:
        kH, kW = int(x.shape[-2]), int(x.shape[-1])
        if min(kH, kW) >= int(min_kernel_hw):
            xx = x.contiguous().reshape(-1, kH, kW)
            yy = op.apply_2d_batch(xx, step_t=step_t)
            return yy.reshape_as(x)
        return x

    # ---- Linear weights: [out, in] ----
    if param.ndim == 2 and x.ndim == 2:
        in_features = int(param.shape[1])
        hw = _pick_linear_hw(
            in_features=in_features,
            linear_hw=linear_hw,
            allow_rect_linear=allow_rect_linear,
            min_rect_side=min_rect_side,
        )

        if hw is not None and hw[0] * hw[1] == in_features:
            out = int(x.shape[0])
            xx = x.contiguous().reshape(out, hw[0], hw[1])
            yy = op.apply_2d_batch(xx, step_t=step_t)
            return yy.reshape_as(x)

    return x


def _exp_rho_2d(H: int, W: int, alpha: float, device, dtype) -> torch.Tensor:
    yy = torch.arange(H, device=device, dtype=dtype)[:, None]
    xx = torch.arange(W, device=device, dtype=dtype)[None, :]
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return torch.exp(-float(alpha) * dist)


def apply_spatial_pde_saitoh(
    param: torch.Tensor,
    tensor: torch.Tensor,
    op: SpectralPDE,
    step_t: int,
    rho_alpha: float = 0.15,
    eps: float = 1e-6,
    linear_hw: Optional[Tuple[int, int]] = None,
    allow_rect_linear: bool = False,
    min_rect_side: int = 8,
    min_kernel_hw: int = 3,
) -> torch.Tensor:
    """
    Saitoh-style normalized smoothing:
        y = S(x * rho) / (S(rho) + eps)

    Same geometry policy as apply_spatial_pde:
      - Conv kernels always eligible (size check)
      - Linear only if linear_hw provided OR allow_rect_linear=True
    """
    x = tensor

    # ---- Conv kernels ----
    if param.ndim == 4 and x.ndim == 4:
        kH, kW = int(x.shape[-2]), int(x.shape[-1])
        if min(kH, kW) >= int(min_kernel_hw):
            xx = x.contiguous().reshape(-1, kH, kW)
            rho = _exp_rho_2d(kH, kW, rho_alpha, xx.device, xx.dtype)[None, :, :]
            num = op.apply_2d_batch(xx * rho, step_t=step_t)
            den = op.apply_2d_batch(rho.expand_as(xx), step_t=step_t).clamp_min(float(eps))
            yy = num / den
            return yy.reshape_as(x)
        return x

    # ---- Linear weights ----
    if param.ndim == 2 and x.ndim == 2:
        in_features = int(param.shape[1])
        hw = _pick_linear_hw(
            in_features=in_features,
            linear_hw=linear_hw,
            allow_rect_linear=allow_rect_linear,
            min_rect_side=min_rect_side,
        )

        if hw is not None and hw[0] * hw[1] == in_features:
            out = int(x.shape[0])
            xx = x.contiguous().reshape(out, hw[0], hw[1])
            rho = _exp_rho_2d(hw[0], hw[1], rho_alpha, xx.device, xx.dtype)[None, :, :]
            num = op.apply_2d_batch(xx * rho, step_t=step_t)
            den = op.apply_2d_batch(rho.expand_as(xx), step_t=step_t).clamp_min(float(eps))
            yy = num / den
            return yy.reshape_as(x)

    return x

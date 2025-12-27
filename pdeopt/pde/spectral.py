import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch

@dataclass
class PDEOperatorParams:
    lam: float = 1.0
    sigma_lap: float = 0.08
    k_2d: Optional[Dict[Tuple[int, int], float]] = None
    preserve_dc: bool = True
    eps_denom: float = 1e-6

    # anneal: scale_t = s0 / sqrt(t + eps)
    anneal_s0: float = 1.0
    anneal_eps: float = 10.0

    # Young-style stability enforcement
    enforce_young_stability: bool = True
    young_margin: float = 0.05  # denom(DC) >= young_margin * lam

class SpectralPDE:
    """
    Spectral PDE-like smoother on the last two dims, using FFT2/IFFT2.
    """
    def __init__(self, p: PDEOperatorParams):
        self.p = p
        if self.p.k_2d is None:
            self.p.k_2d = {(0, 1): 0.010, (1, 0): 0.010, (0, 2): 0.005, (2, 0): 0.005, (1, 1): 0.003}
        self._base_cache_2d = {}

    def scale(self, step_t: int) -> float:
        return float(self.p.anneal_s0) / math.sqrt(float(step_t) + float(self.p.anneal_eps))

    def _k_l1_bound(self) -> float:
        s = 0.0
        for (dy, dx), wgt in self.p.k_2d.items():
            w = abs(float(wgt))
            if dy == 0 and dx == 0:
                s += w
            else:
                s += 2.0 * w
        return float(s)

    def _symbol_2d(self, H: int, W: int, device, dtype, scale: float) -> torch.Tensor:
        key = (int(H), int(W), str(device), str(dtype))
        cached = self._base_cache_2d.get(key, None)
        if cached is None:
            wy = 2.0 * math.pi * torch.fft.fftfreq(H, device=device, dtype=dtype)
            wx = 2.0 * math.pi * torch.fft.fftfreq(W, device=device, dtype=dtype)
            WY = wy[:, None]
            WX = wx[None, :]

            lap = 4.0 * (torch.sin(WY * 0.5) ** 2 + torch.sin(WX * 0.5) ** 2)

            tcK = torch.zeros((H, W), device=device, dtype=dtype)
            for (dy, dx), wgt in self.p.k_2d.items():
                if dy == 0 and dx == 0:
                    tcK = tcK + float(wgt)
                else:
                    tcK = tcK + 2.0 * float(wgt) * torch.cos(WY * float(dy) + WX * float(dx))

            self._base_cache_2d[key] = (lap, tcK)
            cached = (lap, tcK)

        lap, tcK = cached

        # Young-style stability enforcement at DC
        tcK_eff = tcK
        if self.p.enforce_young_stability:
            K_l1 = self._k_l1_bound()
            denom0_est = float(self.p.lam) - float(scale) * float(K_l1)
            min_allowed = float(self.p.young_margin) * float(self.p.lam)
            if denom0_est < min_allowed:
                target = max(float(self.p.lam) - min_allowed, 0.0)
                shrink = target / max(float(scale) * float(K_l1), 1e-12)
                shrink = float(max(0.0, min(1.0, shrink)))
                tcK_eff = tcK * shrink

        denom = float(self.p.lam) + (float(scale) * float(self.p.sigma_lap)) * lap - (float(scale) * tcK_eff)
        denom = torch.clamp(denom, min=float(self.p.eps_denom))

        if self.p.preserve_dc:
            d0 = torch.clamp(denom[0, 0], min=float(self.p.eps_denom))
            Hsym = d0 / denom
        else:
            Hsym = 1.0 / denom
        return Hsym

    def apply_2d_batch(self, x: torch.Tensor, step_t: int) -> torch.Tensor:
        if x.ndim < 2:
            return x
        H, W = int(x.shape[-2]), int(x.shape[-1])
        scale = self.scale(step_t)
        Hsym = self._symbol_2d(H, W, x.device, x.dtype, scale)
        X = torch.fft.fft2(x, dim=(-2, -1))
        Y = X * Hsym
        return torch.fft.ifft2(Y, dim=(-2, -1)).real

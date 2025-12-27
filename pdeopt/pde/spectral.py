import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


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

    # ===== NEW: training-phase schedule multiplier s(t) =====
    # scale_t <- scale_t * sched_mult(t)
    sched_warmup_steps: int = 0        # 0 -> disable warmup
    sched_decay_start: int = 0         # 0 -> disable decay
    sched_decay_end: int = 0           # <= start -> disable decay
    sched_min_mult: float = 0.1        # multiplier at end of decay

    # Young-style stability enforcement
    enforce_young_stability: bool = True
    young_margin: float = 0.05  # min denom >= young_margin * lam

    # Extra stability: ensure global minimum denom >= young_margin * lam (helps deep nets)
    enforce_global_min: bool = True

    # ===== NEW: soft margin instead of hard clamp artifacts =====
    use_soft_margin: bool = False      # keep default False to preserve old behavior
    soft_margin_beta: float = 10.0     # higher -> closer to hard clamp
    # soft margin target is max(eps_denom, young_margin * lam) by default


class SpectralPDE:
    """
    Spectral PDE-like smoother on the last two dims, using FFT2/IFFT2.
    Constructs a real frequency response H(ω) = 1 / denom(ω), optionally preserve DC.

    Deep-learning improvements:
      - do FFT/filter in fp32 when x is fp16/bf16 to avoid numerical artifacts under AMP.
      - optional schedule multiplier in scale(step_t): warmup -> on -> decay
      - optional soft margin on denom to reduce hard clamp artifacts
    """

    def __init__(self, p: PDEOperatorParams):
        self.p = p
        if self.p.k_2d is None:
            self.p.k_2d = {
                (0, 1): 0.010, (1, 0): 0.010,
                (0, 2): 0.005, (2, 0): 0.005,
                (1, 1): 0.003
            }
        # cache base (lap, tcK) per (H,W,device,dtype)
        self._base_cache_2d = {}

    def _sched_mult(self, step_t: int) -> float:
        """s(t): ramp-up then optional decay."""
        t = int(step_t)

        # warmup: 0 -> 1
        w = int(self.p.sched_warmup_steps)
        if w > 0 and t < w:
            return float(t) / float(max(1, w))

        mult = 1.0

        # decay: 1 -> sched_min_mult
        ds = int(self.p.sched_decay_start)
        de = int(self.p.sched_decay_end)
        if ds > 0 and de > ds and t >= ds:
            frac = float(t - ds) / float(max(1, de - ds))
            frac = max(0.0, min(1.0, frac))
            mult = 1.0 - frac * (1.0 - float(self.p.sched_min_mult))

        return float(mult)

    def scale(self, step_t: int) -> float:
        base = float(self.p.anneal_s0) / math.sqrt(float(step_t) + float(self.p.anneal_eps))
        return base * self._sched_mult(step_t)

    def _k_l1_bound(self) -> float:
        """
        L1 bound for symmetric cosine kernel contribution.
        NOTE: assumes you store only one of ±(dy,dx) in k_2d (not both).
        """
        s = 0.0
        for (dy, dx), wgt in self.p.k_2d.items():
            w = abs(float(wgt))
            if dy == 0 and dx == 0:
                s += w
            else:
                s += 2.0 * w
        return float(s)

    def _soft_lower_barrier(self, denom: torch.Tensor, target_min: float) -> torch.Tensor:
        """
        Smoothly enforce denom >= target_min:
          denom <- denom + softplus(target_min - denom)
        This behaves like a smooth clamp but avoids hard kinks.
        """
        beta = float(self.p.soft_margin_beta)
        # softplus(x) ~ max(0,x) when beta large
        return denom + F.softplus((target_min - denom) * beta) / max(beta, 1e-12)

    def _symbol_2d(self, H: int, W: int, device, dtype, scale: float) -> torch.Tensor:
        key = (int(H), int(W), str(device), str(dtype))
        cached = self._base_cache_2d.get(key, None)

        if cached is None:
            wy = 2.0 * math.pi * torch.fft.fftfreq(H, device=device, dtype=dtype)
            wx = 2.0 * math.pi * torch.fft.fftfreq(W, device=device, dtype=dtype)
            WY = wy[:, None]
            WX = wx[None, :]

            # discrete Laplacian symbol (positive)
            lap = 4.0 * (torch.sin(WY * 0.5) ** 2 + torch.sin(WX * 0.5) ** 2)

            # cosine kernel symbol
            tcK = torch.zeros((H, W), device=device, dtype=dtype)
            for (dy, dx), wgt in self.p.k_2d.items():
                if dy == 0 and dx == 0:
                    tcK = tcK + float(wgt)
                else:
                    tcK = tcK + 2.0 * float(wgt) * torch.cos(WY * float(dy) + WX * float(dx))

            self._base_cache_2d[key] = (lap, tcK)
        else:
            lap, tcK = cached

        tcK_eff = tcK

        # Young-style enforcement (coarse bound)
        min_allowed = float(self.p.young_margin) * float(self.p.lam)

        if self.p.enforce_young_stability:
            K_l1 = self._k_l1_bound()
            denom0_est = float(self.p.lam) - float(scale) * float(K_l1)
            if denom0_est < min_allowed:
                # shrink tcK to satisfy lam - scale*K_l1 >= min_allowed
                target = max(float(self.p.lam) - min_allowed, 0.0)
                shrink = target / max(float(scale) * float(K_l1), 1e-12)
                shrink = float(max(0.0, min(1.0, shrink)))
                tcK_eff = tcK * shrink

        denom = (
            float(self.p.lam)
            + (float(scale) * float(self.p.sigma_lap)) * lap
            - (float(scale) * tcK_eff)
        )

        # ===== Stability floor =====
        eps_floor = float(self.p.eps_denom)
        target_min = max(eps_floor, min_allowed)

        if bool(self.p.use_soft_margin):
            denom = self._soft_lower_barrier(denom, target_min=target_min)
        else:
            denom = torch.clamp(denom, min=eps_floor)

        # Extra: ensure global minimum denom is not too small (prevents very small denominators)
        if self.p.enforce_young_stability and self.p.enforce_global_min:
            dmin = float(denom.amin().item())
            if dmin < min_allowed:
                denom = denom + (min_allowed - dmin)

        # Frequency response
        if self.p.preserve_dc:
            d0 = torch.clamp(denom[0, 0], min=eps_floor)
            Hsym = d0 / denom   # keep DC gain = 1
        else:
            Hsym = 1.0 / denom

        return Hsym

    def apply_2d_batch(self, x: torch.Tensor, step_t: int) -> torch.Tensor:
        """
        Apply spectral smoothing to x on the last two dims.

        AMP-safe behavior:
          - If x is fp16/bf16, run FFT/filter in fp32 and cast back.
        """
        if x.ndim < 2:
            return x

        H, W = int(x.shape[-2]), int(x.shape[-1])

        orig_dtype = x.dtype
        need_fp32 = orig_dtype in (torch.float16, torch.bfloat16)

        x_work = x.float() if need_fp32 else x

        scale = self.scale(step_t)
        Hsym = self._symbol_2d(H, W, x_work.device, x_work.dtype, scale)

        X = torch.fft.fft2(x_work, dim=(-2, -1))
        Y = X * Hsym
        y = torch.fft.ifft2(Y, dim=(-2, -1)).real

        if need_fp32:
            y = y.to(dtype=orig_dtype)

        return y

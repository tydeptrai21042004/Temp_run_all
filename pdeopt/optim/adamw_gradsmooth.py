from typing import Optional
import torch
from ..smooth.avg import apply_spatial_avg
from ..core.metrics import hf_energy_2d, lap_energy_2d

class AdamW_GradSmooth(torch.optim.Optimizer):
    """
    AdamW + GradSmooth (LSGD) baseline: smooths gradients (not preconditioner).

    g_used = (1-alpha) * g + alpha * Smooth(g)

    Optional:
      track_smoothness=True to log roughness metrics of g before/after smoothing.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        alpha: float = 0.5,
        clip_update: Optional[float] = None,
        track_smoothness: bool = False,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha=alpha, clip_update=clip_update, track_smoothness=track_smoothness)
        super().__init__(params, defaults)
        self._step = 0
        self._sm_sum = {"hf_b": 0.0, "hf_a": 0.0, "lap_b": 0.0, "lap_a": 0.0}
        self._sm_n = 0

    def get_epoch_stats(self):
        out = {}
        if self._sm_n > 0:
            out["g_hf_before"] = self._sm_sum["hf_b"] / self._sm_n
            out["g_hf_after"] = self._sm_sum["hf_a"] / self._sm_n
            out["g_lap_before"] = self._sm_sum["lap_b"] / self._sm_n
            out["g_lap_after"] = self._sm_sum["lap_a"] / self._sm_n
        return out

    def reset_epoch_stats(self):
        for k in self._sm_sum:
            self._sm_sum[k] = 0.0
        self._sm_n = 0

    def _maybe_track(self, p: torch.Tensor, g: torch.Tensor, gs: torch.Tensor):
        if p.ndim == 2 and g.ndim == 2 and int(p.shape[1]) == 28 * 28:
            gb = g.reshape(int(g.shape[0]), 28, 28)
            ga = gs.reshape(int(gs.shape[0]), 28, 28)
        elif p.ndim == 4 and g.ndim == 4:
            kH, kW = int(g.shape[-2]), int(g.shape[-1])
            gb = g.reshape(-1, kH, kW)
            ga = gs.reshape(-1, kH, kW)
        else:
            return
        self._sm_sum["hf_b"] += hf_energy_2d(gb)
        self._sm_sum["hf_a"] += hf_energy_2d(ga)
        self._sm_sum["lap_b"] += lap_energy_2d(gb)
        self._sm_sum["lap_a"] += lap_energy_2d(ga)
        self._sm_n += 1

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        t = self._step

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            alpha = float(group["alpha"])
            clip_update = group["clip_update"]
            track_smoothness = bool(group.get("track_smoothness", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    g = g.to_dense()

                g_s = apply_spatial_avg(p, g)
                if track_smoothness:
                    self._maybe_track(p, g, g_s)

                g_used = (1.0 - alpha) * g + alpha * g_s

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.mul_(beta1).add_(g_used, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_used, g_used, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2

                upd = m_hat / (torch.sqrt(v_hat) + eps)
                if clip_update is not None:
                    upd = torch.clamp(upd, min=-float(clip_update), max=float(clip_update))

                p.add_(upd, alpha=-lr)

        return None

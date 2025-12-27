from typing import Optional
import torch
from ..smooth.avg import apply_spatial_avg
from ..core.metrics import hf_energy_2d, lap_energy_2d

class AdamW_LRMP(torch.optim.Optimizer):
    """
    Baseline: AdamW that smooths the *preconditioner* r_t spatially (cheap avg kernel),
    mixed by LRMP coefficient rho_t.

    Ablations:
      rho_mode = "lrmp" | "const" | "off"

    Optional:
      track_smoothness=True to log roughness metrics of r_t before/after smoothing.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        rho0=0.6,
        lr_cap=1e3,
        clip_update: Optional[float] = 1.0,
        rho_mode: str = "lrmp",
        rho_const: Optional[float] = None,
        use_time_filter: bool = False,
        beta_r: Optional[float] = None,
        time_beta: float = 0.9,
        track_smoothness: bool = False,
    ):
        if beta_r is not None:
            use_time_filter = True
            time_beta = float(beta_r)

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            rho0=rho0, lr_cap=lr_cap, clip_update=clip_update,
            rho_mode=rho_mode, rho_const=rho_const,
            use_time_filter=use_time_filter, time_beta=time_beta,
            track_smoothness=track_smoothness,
        )
        super().__init__(params, defaults)

        self._step = 0
        self._time_state = {}
        self._rho_sum = 0.0
        self._rho_n = 0

        self._sm_sum = {"hf_b": 0.0, "hf_a": 0.0, "lap_b": 0.0, "lap_a": 0.0}
        self._sm_n = 0

    def get_epoch_stats(self):
        out = {}
        if self._rho_n > 0:
            out["rho_mean"] = self._rho_sum / max(1, self._rho_n)
        if self._sm_n > 0:
            out["r_hf_before"] = self._sm_sum["hf_b"] / self._sm_n
            out["r_hf_after"] = self._sm_sum["hf_a"] / self._sm_n
            out["r_lap_before"] = self._sm_sum["lap_b"] / self._sm_n
            out["r_lap_after"] = self._sm_sum["lap_a"] / self._sm_n
        return out

    def reset_epoch_stats(self):
        self._rho_sum = 0.0
        self._rho_n = 0
        for k in self._sm_sum:
            self._sm_sum[k] = 0.0
        self._sm_n = 0

    def _maybe_track(self, p: torch.Tensor, r: torch.Tensor, rs: torch.Tensor):
        if p.ndim == 2 and r.ndim == 2 and int(p.shape[1]) == 28 * 28:
            rb = r.reshape(int(r.shape[0]), 28, 28)
            ra = rs.reshape(int(rs.shape[0]), 28, 28)
        elif p.ndim == 4 and r.ndim == 4:
            kH, kW = int(r.shape[-2]), int(r.shape[-1])
            rb = r.reshape(-1, kH, kW)
            ra = rs.reshape(-1, kH, kW)
        else:
            return
        self._sm_sum["hf_b"] += hf_energy_2d(rb)
        self._sm_sum["hf_a"] += hf_energy_2d(ra)
        self._sm_sum["lap_b"] += lap_energy_2d(rb)
        self._sm_sum["lap_a"] += lap_energy_2d(ra)
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
            rho0 = float(group["rho0"])
            lr_cap = float(group["lr_cap"])
            clip_update = group["clip_update"]

            rho_mode = str(group.get("rho_mode", "lrmp")).lower().strip()
            rho_const = group.get("rho_const", None)
            rho_const = float(rho_const) if rho_const is not None else None

            use_time_filter = bool(group["use_time_filter"])
            time_beta = float(group["time_beta"])
            track_smoothness = bool(group.get("track_smoothness", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    g = g.to_dense()

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2

                r = torch.rsqrt(v_hat + eps)
                r = torch.clamp(r, max=lr_cap)

                if use_time_filter:
                    key = id(p)
                    te = self._time_state.get(key)
                    if te is None:
                        te = r.detach().clone()
                    else:
                        te.mul_(time_beta).add_(r, alpha=1.0 - time_beta)
                    self._time_state[key] = te
                    r = te

                r_smooth = apply_spatial_avg(p, r)

                if track_smoothness:
                    self._maybe_track(p, r, r_smooth)

                if rho_mode == "off":
                    rho = 0.0
                elif rho_mode == "const":
                    rho = float(rho_const if rho_const is not None else rho0)
                else:
                    g2 = torch.mean(g * g).clamp_min(1e-12)
                    nm = torch.mean((g - m) * (g - m)) / g2
                    rho = float(rho0) * float(torch.clamp(nm, 0.0, 1.0).item())

                self._rho_sum += rho
                self._rho_n += 1

                r_final = (1.0 - rho) * r + rho * r_smooth

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                upd = m_hat * r_final
                if clip_update is not None:
                    upd = torch.clamp(upd, min=-float(clip_update), max=float(clip_update))
                p.add_(upd, alpha=-lr)

        return None

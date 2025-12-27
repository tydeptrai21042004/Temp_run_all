from typing import Optional, Tuple
import torch
from ..pde.apply import apply_spatial_pde, apply_spatial_pde_saitoh
from ..pde.spectral import SpectralPDE
from ..core.metrics import hf_energy_2d, lap_energy_2d

class PDE_AdamW_LRMP(torch.optim.Optimizer):
    """
    PDE-AdamW-LRMP:
      - compute AdamW preconditioner r_t
      - optional time-filter on r_t
      - PDE smooth r_t (plain / saitoh)
      - mix r_t with r_smooth by LRMP rho_t

    Ablations:
      rho_mode = "lrmp" | "const" | "off"
      pde_mode = "plain" | "saitoh"

    Optional:
      track_smoothness=True to log roughness metrics of r_t before/after PDE smoothing.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        op: Optional[SpectralPDE] = None,
        rho0=0.6,
        lr_cap=1e3,
        clip_update: Optional[float] = 1.0,
        rho_mode: str = "lrmp",
        rho_const: Optional[float] = None,
        use_time_filter: bool = False,
        beta_r: Optional[float] = None,
        time_beta: float = 0.9,
        pde_mode: str = "plain",
        saitoh_rho_alpha: float = 0.15,
        saitoh_eps: float = 1e-6,
        linear_hw: Optional[Tuple[int, int]] = None,
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
            pde_mode=pde_mode, saitoh_rho_alpha=saitoh_rho_alpha, saitoh_eps=saitoh_eps,
            linear_hw=linear_hw,
            track_smoothness=track_smoothness,
        )
        super().__init__(params, defaults)

        self.op = op
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

            pde_mode = str(group.get("pde_mode", "plain")).lower().strip()
            saitoh_rho_alpha = float(group.get("saitoh_rho_alpha", 0.15))
            saitoh_eps = float(group.get("saitoh_eps", 1e-6))
            linear_hw = group.get("linear_hw", None)
            track_smoothness = bool(group.get("track_smoothness", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    g = g.to_dense()

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2

                r = torch.rsqrt(v_hat + eps)
                r = torch.clamp(r, max=lr_cap)

                if rho_mode == "off":
                    rho = 0.0
                elif rho_mode == "const":
                    rho = float(rho_const if rho_const is not None else rho0)
                else:
                    g2 = g.pow(2).mean()
                    if float(g2.item()) == 0.0:
                        rho = 0.0
                    else:
                        nr = (g - m).pow(2).mean() / (g2 + 1e-12)
                        nr = float(torch.clamp(nr, 0.0, 1.0).item())
                        rho = rho0 * nr

                self._rho_sum += rho
                self._rho_n += 1

                if use_time_filter:
                    key = id(p)
                    prev = self._time_state.get(key, None)
                    if prev is None or prev.shape != r.shape:
                        self._time_state[key] = r.detach().clone()
                    else:
                        self._time_state[key].mul_(time_beta).add_(r, alpha=1.0 - time_beta)
                    r = self._time_state[key]

                if self.op is not None and rho > 0.0:
                    if pde_mode == "saitoh":
                        rs = apply_spatial_pde_saitoh(
                            p, r, self.op, step_t=t,
                            rho_alpha=saitoh_rho_alpha, eps=saitoh_eps,
                            linear_hw=linear_hw,
                        )
                    else:
                        rs = apply_spatial_pde(p, r, self.op, step_t=t, linear_hw=linear_hw)

                    rs = torch.clamp(rs, min=0.0, max=lr_cap)

                    if track_smoothness:
                        self._maybe_track(p, r, rs)

                    r = (1.0 - rho) * r + rho * rs

                upd = m_hat * r
                if clip_update is not None:
                    nrm = upd.norm().clamp(min=1e-12)
                    c = float(clip_update)
                    if nrm.item() > c:
                        upd.mul_(c / nrm)

                p.add_(upd, alpha=-lr)

        return None

class PDE_AdamW_LRMP_SAITOH(PDE_AdamW_LRMP):
    def __init__(self, *args, saitoh_rho_alpha: float = 0.15, **kwargs):
        kwargs["pde_mode"] = "saitoh"
        kwargs["saitoh_rho_alpha"] = saitoh_rho_alpha
        super().__init__(*args, **kwargs)

from typing import Optional, Tuple
import math
import torch

from ..pde.apply import apply_spatial_pde, apply_spatial_pde_saitoh
from ..pde.spectral import SpectralPDE
from ..core.metrics import hf_energy_2d, lap_energy_2d


class PDE_AdamW_LRMP(torch.optim.Optimizer):
    """
    PDE-AdamW-LRMP (deep-learning improved, core math preserved):

      Core:
        - AdamW moments -> r_t = (v_hat + eps)^(-1/2)
        - PDE smooth r_t (plain/saitoh) on geometry tensors only
        - LRMP mixing (UPDATED):
            rho_t = rho_min + (rho_max - rho_min) * tanh(q_t)
          so rho can go near 0 when noise is low (prevents over-smoothing).

      Improvements (keep core math):
        1) Log-domain smoothing: smooth u=log(r+delta), then exp back
        2) Robust q: clip / optional quantile, compute stats in fp32
        3) rho EMA: smooth rho over time (per-parameter)
        4) PDE schedule: warmup -> on -> decay (multiplier on PDE effect)
        5) Update-to-weight coupling: use u=||Δθ||/(||θ||+eps) to self-tune PDE strength
        6) Time-filter EMA stored in state[p] (resume-safe)

      NEW (requested):
        7) Log-space mixing (geometric interpolation) instead of linear r-space mixing
        8) Preserve mean/DC scale of r after PDE smoothing (prevents implicit LR change)
        9) Apply PDE only to conv kernels by default (optional MNIST first linear reshape)

    Ablations:
      rho_mode = "lrmp" | "const" | "off"
      pde_mode = "plain" | "saitoh"
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        op: Optional[SpectralPDE] = None,

        # Backward compatible: rho0 remains; if rho_max=None, rho_max := rho0
        rho0=0.1,
        rho_min: float = 0.0,
        rho_max: Optional[float] = None,

        lr_cap=1e3,
        clip_update: Optional[float] = None,
        rho_mode: str = "lrmp",
        rho_const: Optional[float] = None,

        # --- LRMP noise estimation ---
        beta_g: Optional[float] = None,       # EMA on gradient for q_t (paper-style)
        beta_r: Optional[float] = None,       # backward compat (treated as beta_g)

        # --- optional time smoothing on r_t ---
        use_time_filter: bool = False,
        time_beta: float = 0.9,

        # --- PDE mode ---
        pde_mode: str = "plain",
        saitoh_rho_alpha: float = 0.15,
        saitoh_eps: float = 1e-6,

        # --- reshape for Linear ---
        linear_hw: Optional[Tuple[int, int]] = None,
        allow_rect_linear: bool = False,
        min_rect_side: int = 8,

        # --- apply frequency ---
        pde_every: int = 1,

        # --- skip tiny tensors / no-geometry ---
        min_param_numel: int = 512,

        # --- logging ---
        track_smoothness: bool = False,

        # ============================
        # log-domain smoothing
        # ============================
        use_log_smooth: bool = True,
        log_smooth_delta: float = 1e-12,

        # ============================
        # robust q + rho EMA
        # ============================
        q_clip: float = 10.0,
        q_quantile: Optional[float] = None,   # e.g. 0.5 for median; None => mean of clipped
        rho_ema_beta: float = 0.9,            # EMA on rho to avoid jitter (0 disables)

        # ============================
        # schedules (PDE strength)
        # ============================
        rho_warmup_steps: int = 0,            # warmup rho (mixing) from 0 -> rho over these steps
        pde_warmup_steps: int = 0,            # warmup PDE multiplier 0 -> 1
        pde_decay_start: int = 0,             # start step for decay (0 disables decay)
        pde_decay_end: int = 0,               # end step for decay (<=start disables decay)
        pde_min_mult: float = 0.1,            # multiplier at end of decay

        # ============================
        # update-to-weight coupling
        # ============================
        u_beta: float = 0.9,                  # EMA for update/weight ratio (0 disables)
        u_target: float = 1e-3,               # target ratio
        u_gain: float = 1.0,                  # exponent for scaling
        u_mult_min: float = 0.5,
        u_mult_max: float = 2.0,

        # ============================
        # NEW: mixing / geometry controls
        # ============================
        pde_log_mix: bool = True,             # geometric interpolation for r <-> rs
        pde_mix_eps: float = 1e-12,           # clamp_min before log in log-mix
        pde_preserve_mean: bool = True,       # keep mean(r) after PDE (prevents LR drift)
        pde_preserve_clip_lo: float = 0.5,
        pde_preserve_clip_hi: float = 2.0,

        pde_only_conv: bool = True,           # apply PDE only on conv kernels (ndim==4)
        pde_allow_mnist_first_linear: bool = False,  # allow ndim==2 with in_features==28*28
    ):
        # Backward compatibility
        if beta_g is None and beta_r is not None:
            beta_g = float(beta_r)

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,

            rho0=float(rho0),
            rho_min=float(rho_min),
            rho_max=(float(rho_max) if rho_max is not None else None),

            lr_cap=lr_cap, clip_update=clip_update,
            rho_mode=rho_mode, rho_const=rho_const,
            beta_g=beta_g,

            use_time_filter=use_time_filter, time_beta=time_beta,
            pde_mode=pde_mode, saitoh_rho_alpha=saitoh_rho_alpha, saitoh_eps=saitoh_eps,

            linear_hw=linear_hw,
            allow_rect_linear=allow_rect_linear,
            min_rect_side=min_rect_side,

            pde_every=int(max(1, pde_every)),
            min_param_numel=int(max(0, min_param_numel)),
            track_smoothness=track_smoothness,

            use_log_smooth=bool(use_log_smooth),
            log_smooth_delta=float(log_smooth_delta),

            q_clip=float(q_clip),
            q_quantile=(float(q_quantile) if q_quantile is not None else None),
            rho_ema_beta=float(rho_ema_beta),

            rho_warmup_steps=int(max(0, rho_warmup_steps)),
            pde_warmup_steps=int(max(0, pde_warmup_steps)),
            pde_decay_start=int(max(0, pde_decay_start)),
            pde_decay_end=int(max(0, pde_decay_end)),
            pde_min_mult=float(pde_min_mult),

            u_beta=float(u_beta),
            u_target=float(u_target),
            u_gain=float(u_gain),
            u_mult_min=float(u_mult_min),
            u_mult_max=float(u_mult_max),

            # NEW
            pde_log_mix=bool(pde_log_mix),
            pde_mix_eps=float(pde_mix_eps),
            pde_preserve_mean=bool(pde_preserve_mean),
            pde_preserve_clip_lo=float(pde_preserve_clip_lo),
            pde_preserve_clip_hi=float(pde_preserve_clip_hi),

            pde_only_conv=bool(pde_only_conv),
            pde_allow_mnist_first_linear=bool(pde_allow_mnist_first_linear),
        )
        super().__init__(params, defaults)

        self.op = op
        self._step = 0

        self._rho_sum = 0.0
        self._rho_n = 0

        self._sm_sum = {"hf_b": 0.0, "hf_a": 0.0, "lap_b": 0.0, "lap_a": 0.0}
        self._sm_n = 0

    def get_epoch_stats(self):
        out = {}
        if self._rho_n > 0:
            out["rho_mean"] = float(self._rho_sum / max(1, self._rho_n))
        if self._sm_n > 0:
            out["r_hf_before"] = float(self._sm_sum["hf_b"] / self._sm_n)
            out["r_hf_after"] = float(self._sm_sum["hf_a"] / self._sm_n)
            out["r_lap_before"] = float(self._sm_sum["lap_b"] / self._sm_n)
            out["r_lap_after"] = float(self._sm_sum["lap_a"] / self._sm_n)
        return out

    def reset_epoch_stats(self):
        self._rho_sum = 0.0
        self._rho_n = 0
        for k in self._sm_sum:
            self._sm_sum[k] = 0.0
        self._sm_n = 0

    def _maybe_track(self, p: torch.Tensor, r: torch.Tensor, rs: torch.Tensor):
        # Only track when we truly have 2D geometry
        if p.ndim == 2 and r.ndim == 2 and int(p.shape[1]) == 28 * 28:
            rb = r.reshape(int(r.shape[0]), 28, 28)
            ra = rs.reshape(int(rs.shape[0]), 28, 28)
        elif p.ndim == 4 and r.ndim == 4:
            kH, kW = int(r.shape[-2]), int(r.shape[-1])
            rb = r.reshape(-1, kH, kW)
            ra = rs.reshape(-1, kH, kW)
        else:
            return

        self._sm_sum["hf_b"] += float(hf_energy_2d(rb))
        self._sm_sum["hf_a"] += float(hf_energy_2d(ra))
        self._sm_sum["lap_b"] += float(lap_energy_2d(rb))
        self._sm_sum["lap_a"] += float(lap_energy_2d(ra))
        self._sm_n += 1

    @staticmethod
    def _pde_schedule_mult(
        t: int,
        warmup_steps: int,
        decay_start: int,
        decay_end: int,
        min_mult: float,
    ) -> float:
        # warmup: 0 -> 1
        if warmup_steps > 0 and t < warmup_steps:
            return float(t) / float(max(1, warmup_steps))

        mult = 1.0

        # decay: 1 -> min_mult
        if decay_start > 0 and decay_end > decay_start and t >= decay_start:
            frac = float(t - decay_start) / float(max(1, decay_end - decay_start))
            frac = max(0.0, min(1.0, frac))
            mult = 1.0 - frac * (1.0 - float(min_mult))

        return float(mult)

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        t = self._step

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            rho0 = float(group.get("rho0", 0.1))
            rho_min = float(group.get("rho_min", 0.0))
            rho_max = group.get("rho_max", None)
            rho_max = float(rho_max) if rho_max is not None else float(rho0)  # backward compat

            # sanitize rho bounds
            rho_min = float(max(0.0, min(0.999, rho_min)))
            rho_max = float(max(rho_min, min(0.999, rho_max)))

            lr_cap = float(group["lr_cap"])
            clip_update = group["clip_update"]

            rho_mode = str(group.get("rho_mode", "lrmp")).lower().strip()
            rho_const = group.get("rho_const", None)
            rho_const = float(rho_const) if rho_const is not None else None

            beta_g = group.get("beta_g", None)
            beta_g = float(beta_g) if beta_g is not None else None

            use_time_filter = bool(group["use_time_filter"])
            time_beta = float(group["time_beta"])

            pde_mode = str(group.get("pde_mode", "plain")).lower().strip()
            saitoh_rho_alpha = float(group.get("saitoh_rho_alpha", 0.15))
            saitoh_eps = float(group.get("saitoh_eps", 1e-6))

            linear_hw = group.get("linear_hw", None)
            allow_rect_linear = bool(group.get("allow_rect_linear", False))
            min_rect_side = int(group.get("min_rect_side", 8))

            pde_every = int(group.get("pde_every", 1))
            min_param_numel = int(group.get("min_param_numel", 0))
            track_smoothness = bool(group.get("track_smoothness", False))

            # log smoothing
            use_log_smooth = bool(group.get("use_log_smooth", True))
            log_delta = float(group.get("log_smooth_delta", 1e-12))

            # q/rho smoothing
            q_clip = float(group.get("q_clip", 10.0))
            q_quantile = group.get("q_quantile", None)
            q_quantile = float(q_quantile) if q_quantile is not None else None
            rho_ema_beta = float(group.get("rho_ema_beta", 0.9))

            # schedules
            rho_warmup_steps = int(group.get("rho_warmup_steps", 0))
            pde_warmup_steps = int(group.get("pde_warmup_steps", 0))
            pde_decay_start = int(group.get("pde_decay_start", 0))
            pde_decay_end = int(group.get("pde_decay_end", 0))
            pde_min_mult = float(group.get("pde_min_mult", 0.1))

            # update-to-weight coupling
            u_beta = float(group.get("u_beta", 0.9))
            u_target = float(group.get("u_target", 1e-3))
            u_gain = float(group.get("u_gain", 1.0))
            u_mult_min = float(group.get("u_mult_min", 0.5))
            u_mult_max = float(group.get("u_mult_max", 2.0))

            # NEW: mixing/geometry
            pde_log_mix = bool(group.get("pde_log_mix", True))
            pde_mix_eps = float(group.get("pde_mix_eps", 1e-12))
            pde_mix_eps = float(max(1e-20, pde_mix_eps))

            pde_preserve_mean = bool(group.get("pde_preserve_mean", True))
            pde_preserve_clip_lo = float(group.get("pde_preserve_clip_lo", 0.5))
            pde_preserve_clip_hi = float(group.get("pde_preserve_clip_hi", 2.0))
            if pde_preserve_clip_hi < pde_preserve_clip_lo:
                pde_preserve_clip_hi = pde_preserve_clip_lo

            pde_only_conv = bool(group.get("pde_only_conv", True))
            pde_allow_mnist_first_linear = bool(group.get("pde_allow_mnist_first_linear", False))

            # PDE schedule multiplier (group-level)
            pde_mult = self._pde_schedule_mult(
                t=t,
                warmup_steps=pde_warmup_steps,
                decay_start=pde_decay_start,
                decay_end=pde_decay_end,
                min_mult=pde_min_mult,
            )

            # rho warmup factor (group-level)
            rho_warm = 1.0
            if rho_warmup_steps > 0 and t < rho_warmup_steps:
                rho_warm = float(t) / float(max(1, rho_warmup_steps))

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

                # AdamW decoupled weight decay
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                # Moments
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2

                # Preconditioner
                r = torch.rsqrt(v_hat + eps)
                r = torch.clamp(r, max=lr_cap)

                # -------------------------
                # LRMP rho (per-parameter)
                # -------------------------
                if rho_mode == "off":
                    rho = 0.0
                elif rho_mode == "const":
                    rho = float(rho_const if rho_const is not None else rho0)
                    rho = float(max(0.0, min(0.999, rho)))
                else:
                    # q_t: noise indicator (compute in fp32 for AMP stability)
                    if beta_g is not None:
                        g_ema = state.get("g_ema", None)
                        if g_ema is None or g_ema.shape != g.shape:
                            g_ema = torch.zeros_like(g)
                        g_ema.mul_(beta_g).add_(g, alpha=1.0 - beta_g)
                        state["g_ema"] = g_ema

                        gf = g.float()
                        ge = g_ema.float()
                        q_elem = (gf - ge).pow(2) / (gf.pow(2) + 1e-12)
                    else:
                        mh = m_hat.float()
                        vh = v_hat.float()
                        q_elem = (vh - mh.pow(2)).clamp_min(0.0) / (vh + 1e-12)

                    if q_quantile is not None:
                        q_val = torch.quantile(q_elem.reshape(-1), q_quantile)
                        q_val = torch.clamp(q_val, 0.0, q_clip)
                    else:
                        q_val = q_elem.clamp(0.0, q_clip).mean()

                    # UPDATED mapping (Option B):
                    # rho = rho_min + (rho_max - rho_min) * tanh(q)
                    rho_t = rho_min + (rho_max - rho_min) * torch.tanh(q_val)
                    rho = float(torch.clamp(rho_t, 0.0, 0.999).item())

                # rho EMA (per-parameter) to reduce jitter
                if rho_ema_beta > 0.0 and rho_mode not in ("off",):
                    prev_rho = float(state.get("rho_ema", rho))
                    rho = float(rho_ema_beta * prev_rho + (1.0 - rho_ema_beta) * rho)
                    state["rho_ema"] = rho

                # Apply rho warmup (ramps mixing on)
                rho = float(rho * rho_warm)

                # -------------------------
                # Time-filter EMA on r (resume-safe)
                # -------------------------
                if use_time_filter:
                    r_ema = state.get("r_ema", None)
                    if r_ema is None or r_ema.shape != r.shape:
                        r_ema = r.detach().clone()
                    else:
                        r_ema.mul_(time_beta).add_(r, alpha=1.0 - time_beta)
                    state["r_ema"] = r_ema
                    r = r_ema

                # -------------------------
                # Update-to-weight coupling (uses previous u_ema)
                # -------------------------
                u_mult = 1.0
                if u_beta > 0.0 and u_target > 0.0:
                    u_ema_prev = float(state.get("u_ema", u_target))
                    ratio = u_ema_prev / float(u_target)
                    u_mult = float(ratio ** float(u_gain))
                    u_mult = float(max(u_mult_min, min(u_mult_max, u_mult)))

                # Effective PDE mixing coefficient (only affects PDE part)
                rho_pde = float(rho * pde_mult * u_mult)
                rho_pde = float(max(0.0, min(0.999, rho_pde)))

                # Track actual mixing used (more meaningful than base rho)
                self._rho_sum += rho_pde
                self._rho_n += 1

                # -------------------------
                # PDE smoothing (only on eligible tensors)
                # -------------------------
                is_conv = (p.ndim == 4)
                is_mnist_first_linear = (p.ndim == 2 and int(p.shape[1]) == 28 * 28)

                if pde_only_conv:
                    geom_ok = is_conv or (pde_allow_mnist_first_linear and is_mnist_first_linear)
                else:
                    geom_ok = (p.ndim in (2, 4))

                do_pde = (
                    self.op is not None
                    and rho_pde > 0.0
                    and geom_ok
                    and p.numel() >= min_param_numel
                )

                if do_pde:
                    need_refresh = (pde_every <= 1) or (t % pde_every == 0) or ("pde_rs" not in state)

                    if need_refresh:
                        # log-domain smoothing
                        if use_log_smooth:
                            u = r.clamp_min(log_delta).log()
                            if pde_mode == "saitoh":
                                u_s = apply_spatial_pde_saitoh(
                                    p, u, self.op, step_t=t,
                                    rho_alpha=saitoh_rho_alpha, eps=saitoh_eps,
                                    linear_hw=linear_hw,
                                    allow_rect_linear=allow_rect_linear,
                                    min_rect_side=min_rect_side,
                                )
                            else:
                                u_s = apply_spatial_pde(
                                    p, u, self.op, step_t=t,
                                    linear_hw=linear_hw,
                                    allow_rect_linear=allow_rect_linear,
                                    min_rect_side=min_rect_side,
                                )
                            rs = u_s.exp()
                        else:
                            if pde_mode == "saitoh":
                                rs = apply_spatial_pde_saitoh(
                                    p, r, self.op, step_t=t,
                                    rho_alpha=saitoh_rho_alpha, eps=saitoh_eps,
                                    linear_hw=linear_hw,
                                    allow_rect_linear=allow_rect_linear,
                                    min_rect_side=min_rect_side,
                                )
                            else:
                                rs = apply_spatial_pde(
                                    p, r, self.op, step_t=t,
                                    linear_hw=linear_hw,
                                    allow_rect_linear=allow_rect_linear,
                                    min_rect_side=min_rect_side,
                                )

                        rs = torch.clamp(rs, min=0.0, max=lr_cap)
                        state["pde_rs"] = rs.detach()
                    else:
                        rs = state["pde_rs"].to(device=r.device, dtype=r.dtype)

                    # NEW: preserve mean/DC of r (prevents implicit LR change)
                    if pde_preserve_mean:
                        r_mean = r.float().mean()
                        rs_mean = rs.float().mean()
                        scale = (r_mean / (rs_mean + 1e-12)).clamp(pde_preserve_clip_lo, pde_preserve_clip_hi)
                        rs = rs * scale

                    if track_smoothness and need_refresh:
                        self._maybe_track(p, r, rs)

                    # NEW: log-space mixing (geometric interpolation) by default
                    if pde_log_mix:
                        r = torch.exp(
                            (1.0 - rho_pde) * torch.log(r.clamp_min(pde_mix_eps)) +
                            rho_pde * torch.log(rs.clamp_min(pde_mix_eps))
                        )
                    else:
                        r = (1.0 - rho_pde) * r + rho_pde * rs

                # -------------------------
                # Update
                # -------------------------
                upd = m_hat * r

                # Optional per-tensor update norm clipping
                if clip_update is not None:
                    c = float(clip_update)
                    nrm = upd.norm().clamp(min=1e-12)
                    if nrm.item() > c:
                        upd.mul_(c / nrm)

                # Apply update
                p.add_(upd, alpha=-lr)

                # -------------------------
                # Update u_ema after we know upd (for next step)
                # -------------------------
                if u_beta > 0.0 and u_target > 0.0:
                    denom = float(p.data.norm().item()) + 1e-12
                    numer = float(upd.norm().item())
                    u_now = numer / denom
                    u_prev = float(state.get("u_ema", u_now))
                    state["u_ema"] = float(u_beta * u_prev + (1.0 - u_beta) * u_now)

        return None


class PDE_AdamW_LRMP_SAITOH(PDE_AdamW_LRMP):
    def __init__(self, *args, saitoh_rho_alpha: float = 0.15, **kwargs):
        kwargs["pde_mode"] = "saitoh"
        kwargs["saitoh_rho_alpha"] = saitoh_rho_alpha
        super().__init__(*args, **kwargs)

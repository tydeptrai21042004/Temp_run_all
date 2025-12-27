from pathlib import Path
import torch

from ..core import save_json, plot_curves
from ..data import get_mnist_loaders
from ..models import MNISTLogReg, MNISTMLP, MNISTSmallCNN
from ..optim import AdamW_LRMP, PDE_AdamW_LRMP, PDE_AdamW_LRMP_SAITOH, AdamW_GradSmooth
from ..train import train_one_epoch_classifier, evaluate_classifier, quick_tune_classifier

def _make_methods(pde_op, cfgs_common, cfgs_lrmp, cfgs_pde, cfgs_pde_saitoh, cfgs_lsgd, ablation_lrmp: str):
    def make_adam(model, cfg): return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    def make_adamw(model, cfg): return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    def make_lsgd(model, cfg):
        return AdamW_GradSmooth(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0), alpha=cfg.get("alpha", 0.5), track_smoothness=cfg.get("track_smoothness", False))

    def make_lrmp(model, cfg):
        return AdamW_LRMP(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0), rho0=cfg.get("rho0", 0.6),
                         clip_update=cfg.get("clip_update", 1.0), rho_mode=ablation_lrmp, rho_const=cfg.get("rho_const", None), track_smoothness=cfg.get("track_smoothness", False))

    def make_pde(model, cfg):
        return PDE_AdamW_LRMP(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0), op=pde_op,
                             rho0=cfg.get("rho0", 0.7), lr_cap=cfg.get("lr_cap", 1e3), clip_update=cfg.get("clip_update", 1.0),
                             rho_mode=ablation_lrmp, rho_const=cfg.get("rho_const", None), pde_mode="plain", track_smoothness=cfg.get("track_smoothness", False))

    def make_pde_s(model, cfg):
        return PDE_AdamW_LRMP_SAITOH(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0), op=pde_op,
                                     rho0=cfg.get("rho0", 0.7), lr_cap=cfg.get("lr_cap", 1e3), clip_update=cfg.get("clip_update", 1.0),
                                     rho_mode=ablation_lrmp, rho_const=cfg.get("rho_const", None), saitoh_rho_alpha=cfg.get("saitoh_rho_alpha", 0.15), track_smoothness=cfg.get("track_smoothness", False))

    return [
        ("ADAM", make_adam, cfgs_common),
        ("ADAMW", make_adamw, cfgs_common),
        ("AdamW+GradSmooth(LSGD)", make_lsgd, cfgs_lsgd),
        ("ADAMW-LRMP", make_lrmp, cfgs_lrmp),
        ("PDE-AdamW-LRMP", make_pde, cfgs_pde),
        ("PDE-AdamW-LRMP-Saitoh", make_pde_s, cfgs_pde_saitoh),
    ]

def _run_mnist(build_model_fn, pde_op, out_dir: Path, batch_size: int, epochs: int, device, dtype, ablation_lrmp: str):
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    cfgs_common = [
        {"lr": 0.3, "weight_decay": 0.0},
        {"lr": 0.1, "weight_decay": 0.0},
        {"lr": 0.03, "weight_decay": 0.0},
        {"lr": 0.01, "weight_decay": 0.0},
        {"lr": 0.003, "weight_decay": 0.0},
    ]
    cfgs_lrmp = [{"lr": c["lr"], "weight_decay": c["weight_decay"], "rho0": r, "track_smoothness": True} for c in cfgs_common[:4] for r in (0.3, 0.6)]
    cfgs_pde = [{"lr": c["lr"], "weight_decay": c["weight_decay"], "rho0": r, "track_smoothness": True} for c in cfgs_common[:4] for r in (0.4, 0.7)]
    cfgs_pde_saitoh = [{"lr": c["lr"], "weight_decay": c["weight_decay"], "rho0": r, "saitoh_rho_alpha": a, "track_smoothness": True}
                       for c in cfgs_common[:4] for r in (0.4, 0.7) for a in (0.08, 0.15)]
    cfgs_lsgd = [{"lr": c["lr"], "weight_decay": c["weight_decay"], "alpha": a, "track_smoothness": True}
                 for c in cfgs_common[:4] for a in (0.25, 0.5, 0.75)]

    methods = _make_methods(pde_op, cfgs_common, cfgs_lrmp, cfgs_pde, cfgs_pde_saitoh, cfgs_lsgd, ablation_lrmp=ablation_lrmp)

    histories = {}
    for name, make_opt, cfgs in methods:
        print(f"\n--- [MNIST] Optimizer = {name} | tuning ---")
        best_cfg = quick_tune_classifier(build_model_fn, make_opt, train_loader, test_loader, cfgs, name,
                                         device=device, dtype=dtype, tune_batches=120, eval_batches=50)
        print(f"--- [MNIST] {name} best cfg = {best_cfg} ---\n")

        model = build_model_fn().to(device, dtype=dtype)
        opt = make_opt(model, best_cfg)

        hist = []
        for ep in range(1, epochs + 1):
            if hasattr(opt, "reset_epoch_stats"):
                opt.reset_epoch_stats()
            tr = train_one_epoch_classifier(model, train_loader, opt, device=device, dtype=dtype)
            te = evaluate_classifier(model, test_loader, device=device, dtype=dtype)
            row = dict(epoch=ep, **tr, test_loss=te["loss"], test_acc=te["acc"])
            if hasattr(opt, "get_epoch_stats"):
                row.update(opt.get_epoch_stats())
            hist.append(row)
            print(f"[{name}] epoch={ep}/{epochs} train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
                  f"test_loss={row['test_loss']:.4f} test_acc={row['test_acc']:.4f} time={row['time_s']:.1f}s")
        histories[name] = hist

    save_json(histories, out_dir / "history.json")
    plot_curves(histories, out_dir, title=f"MNIST ({build_model_fn.__name__}) bs={batch_size}")

def run_mnist_logreg(pde_op, out_dir: Path, batch_size=128, epochs=5, device=None, dtype=None, ablation_lrmp="lrmp"):
    def build(): return MNISTLogReg()
    _run_mnist(build, pde_op, out_dir, batch_size, epochs, device, dtype, ablation_lrmp)

def run_mnist_mlp(pde_op, out_dir: Path, batch_size=128, epochs=5, device=None, dtype=None, ablation_lrmp="lrmp"):
    def build(): return MNISTMLP()
    _run_mnist(build, pde_op, out_dir, batch_size, epochs, device, dtype, ablation_lrmp)

def run_mnist_cnn(pde_op, out_dir: Path, batch_size=128, epochs=5, device=None, dtype=None, ablation_lrmp="lrmp"):
    def build(): return MNISTSmallCNN()
    _run_mnist(build, pde_op, out_dir, batch_size, epochs, device, dtype, ablation_lrmp)

from pathlib import Path
import torch

from ..core import save_json, plot_curves
from ..data import get_cifar10_loaders
from ..models import CIFARConvNet, CIFARResNet18
from ..optim import AdamW_LRMP, PDE_AdamW_LRMP, PDE_AdamW_LRMP_SAITOH, AdamW_GradSmooth
from ..train import train_one_epoch_classifier, evaluate_classifier, quick_tune_classifier

def _methods(pde_op, cfgs_common, cfgs_lrmp, cfgs_pde, cfgs_pde_saitoh, cfgs_lsgd, ablation_lrmp: str):
    def make_adam(model, cfg): return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    def make_adamw(model, cfg): return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    def make_lsgd(model, cfg): return AdamW_GradSmooth(model.parameters(), lr=cfg["lr"], weight_decay=0.0, alpha=cfg.get("alpha", 0.5))
    def make_lrmp(model, cfg): return AdamW_LRMP(model.parameters(), lr=cfg["lr"], weight_decay=0.0, rho0=cfg.get("rho0", 0.6), rho_mode=ablation_lrmp)
    def make_pde(model, cfg): return PDE_AdamW_LRMP(model.parameters(), lr=cfg["lr"], weight_decay=0.0, op=pde_op, rho0=cfg.get("rho0", 0.7), rho_mode=ablation_lrmp)
    def make_pde_s(model, cfg): return PDE_AdamW_LRMP_SAITOH(model.parameters(), lr=cfg["lr"], weight_decay=0.0, op=pde_op, rho0=cfg.get("rho0", 0.7), rho_mode=ablation_lrmp, saitoh_rho_alpha=cfg.get("saitoh_rho_alpha", 0.15))

    return [
        ("ADAM", make_adam, cfgs_common),
        ("ADAMW", make_adamw, cfgs_common),
        ("AdamW+GradSmooth(LSGD)", make_lsgd, cfgs_lsgd),
        ("ADAMW-LRMP", make_lrmp, cfgs_lrmp),
        ("PDE-AdamW-LRMP", make_pde, cfgs_pde),
        ("PDE-AdamW-LRMP-Saitoh", make_pde_s, cfgs_pde_saitoh),
    ]

def _run(build_model_fn, pde_op, out_dir: Path, batch_size: int, epochs: int, device, dtype, ablation_lrmp: str):
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    cfgs_common = [
        {"lr": 0.003, "weight_decay": 0.0},
        {"lr": 0.001, "weight_decay": 0.0},
        {"lr": 0.0003, "weight_decay": 0.0},
    ]
    cfgs_lrmp = [{"lr": c["lr"], "rho0": r} for c in cfgs_common for r in (0.3, 0.6)]
    cfgs_pde = [{"lr": c["lr"], "rho0": r} for c in cfgs_common for r in (0.4, 0.7)]
    cfgs_pde_saitoh = [{"lr": c["lr"], "rho0": r, "saitoh_rho_alpha": a} for c in cfgs_common for r in (0.4, 0.7) for a in (0.08, 0.15)]
    cfgs_lsgd = [{"lr": c["lr"], "alpha": a} for c in cfgs_common for a in (0.25, 0.5, 0.75)]

    methods = _methods(pde_op, cfgs_common, cfgs_lrmp, cfgs_pde, cfgs_pde_saitoh, cfgs_lsgd, ablation_lrmp)

    histories = {}
    for name, make_opt, cfgs in methods:
        print(f"\n--- [CIFAR-10] Optimizer = {name} | tuning ---")
        best_cfg = quick_tune_classifier(build_model_fn, make_opt, train_loader, test_loader, cfgs, name,
                                         device=device, dtype=dtype, tune_batches=90, eval_batches=50)
        print(f"--- [CIFAR-10] {name} best cfg = {best_cfg} ---\n")

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
    plot_curves(histories, out_dir, title=f"CIFAR-10 ({build_model_fn.__name__}) bs={batch_size}")

def run_cifar10_convnet(pde_op, out_dir: Path, batch_size=128, epochs=5, device=None, dtype=None, ablation_lrmp="lrmp"):
    def build(): return CIFARConvNet()
    _run(build, pde_op, out_dir, batch_size, epochs, device, dtype, ablation_lrmp)

def run_cifar10_resnet18(pde_op, out_dir: Path, batch_size=128, epochs=50, device=None, dtype=None, ablation_lrmp="lrmp"):
    def build(): return CIFARResNet18()
    _run(build, pde_op, out_dir, batch_size, epochs, device, dtype, ablation_lrmp)

#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from pdeopt.core import get_device, get_dtype, default_results_dir, set_seed
from pdeopt.pde import PDEOperatorParams, SpectralPDE
from pdeopt.exps import (
    run_mnist_logreg, run_mnist_mlp, run_mnist_cnn,
    run_cifar10_convnet, run_cifar10_resnet18,
    run_fashion_cnn, run_imdb, run_noisy_quadratic
)

def build_pde_op(enforce_young: bool, young_margin: float) -> SpectralPDE:
    pde_params = PDEOperatorParams(
        lam=1.0,
        sigma_lap=0.08,
        k_2d={(0, 1): 0.010, (1, 0): 0.010, (0, 2): 0.005, (2, 0): 0.005, (1, 1): 0.003},
        preserve_dc=True,
        eps_denom=1e-6,
        anneal_s0=1.0,
        anneal_eps=10.0,
        enforce_young_stability=bool(enforce_young),
        young_margin=float(young_margin),
    )
    return SpectralPDE(pde_params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=None)
    ap.add_argument("--exp", type=str, default="all",
                    choices=[
                        "all",
                        "mnist_logreg", "mnist_mlp", "mnist_cnn",
                        "cifar10_convnet", "cifar10_resnet18",
                        "fashion_cnn",
                        "imdb",
                        "noisy_quadratic",
                    ])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--ablation-lrmp", type=str, default="lrmp", choices=["lrmp", "const", "off"])
    ap.add_argument("--pde-enforce-young", type=int, default=1)
    ap.add_argument("--young-margin", type=float, default=0.05)
    args = ap.parse_args()

    device = get_device()
    dtype = get_dtype()
    results_dir = Path(args.results_dir) if args.results_dir else default_results_dir()

    print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} | device={device}")
    print(f"[INFO] results_dir={results_dir.resolve()}")
    print(f"[INFO] seeds={args.seeds} | exp={args.exp}")

    pde_op = build_pde_op(enforce_young=bool(args.pde_enforce_young), young_margin=args.young_margin)

    def run_one(seed: int):
        set_seed(seed)
        out = results_dir / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)

        if args.exp in ("all", "mnist_logreg"):
            run_mnist_logreg(pde_op, out / "mnist_logreg", batch_size=args.batch_size, epochs=args.epochs, device=device, dtype=dtype, ablation_lrmp=args.ablation_lrmp)
        if args.exp in ("all", "mnist_mlp"):
            run_mnist_mlp(pde_op, out / "mnist_mlp", batch_size=args.batch_size, epochs=args.epochs, device=device, dtype=dtype, ablation_lrmp=args.ablation_lrmp)
        if args.exp in ("all", "mnist_cnn"):
            run_mnist_cnn(pde_op, out / "mnist_cnn", batch_size=args.batch_size, epochs=args.epochs, device=device, dtype=dtype, ablation_lrmp=args.ablation_lrmp)

        if args.exp in ("all", "cifar10_convnet"):
            run_cifar10_convnet(pde_op, out / "cifar10_convnet", batch_size=args.batch_size, epochs=args.epochs, device=device, dtype=dtype, ablation_lrmp=args.ablation_lrmp)
        if args.exp in ("all", "cifar10_resnet18"):
            run_cifar10_resnet18(pde_op, out / "cifar10_resnet18", batch_size=args.batch_size, epochs=max(args.epochs, 50), device=device, dtype=dtype, ablation_lrmp=args.ablation_lrmp)

        if args.exp in ("all", "fashion_cnn"):
            run_fashion_cnn(pde_op, out / "fashion_cnn", batch_size=args.batch_size, epochs=args.epochs, device=device, dtype=dtype, ablation_lrmp=args.ablation_lrmp)

        if args.exp in ("all", "imdb"):
            run_imdb(pde_op, out / "imdb", batch_size=max(args.batch_size, 256), epochs=max(3, args.epochs // 2), device=device, dtype=dtype, ablation_lrmp=args.ablation_lrmp)

        if args.exp in ("all", "noisy_quadratic"):
            run_noisy_quadratic(out / "noisy_quadratic", device=device, dtype=dtype, steps=2000, noise_std=0.2, seed=seed)

    for s in args.seeds:
        run_one(int(s))

    print("\n[DONE] Experiments completed.")

if __name__ == "__main__":
    main()

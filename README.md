# PDE-AdamW-LRMP experiments (refactored)

This project refactors your single-file script into a small package with:
- PDE-AdamW-LRMP (plain / Saitoh)
- AdamW-LRMP (preconditioner smoothing baseline)
- **AdamW + GradSmooth (LSGD)** baseline (gradient smoothing)
- Multiple datasets/models (MNIST LogReg/MLP/CNN, FashionMNIST CNN, CIFAR-10 ConvNet/ResNet18, IMDB logreg)
- Optional multi-seed runs and longer budgets
- Optional ablations (LRMP off/const, Young-stability off)

## Install
```bash
pip install -r requirements.txt
```

## Run all (default)
```bash
python run_all.py
```

## Examples
Run MNIST LogReg:
```bash
python run_all.py --exp mnist_logreg --epochs 5 --batch-size 128
```

Multi-seed robust run:
```bash
python run_all.py --exp cifar10_resnet18 --epochs 50 --seeds 0 1 2 3 4
```

Disable LRMP (ablation):
```bash
python run_all.py --exp mnist_logreg --ablation-lrmp off
```

Turn off Young-stability in PDE operator (ablation):
```bash
python run_all.py --exp mnist_logreg --pde-enforce-young 0
```

Results are written under `results/` by default.

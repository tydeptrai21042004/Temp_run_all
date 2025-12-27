from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..core import save_json


def run_noisy_quadratic(out_dir: Path, device, dtype, d=4000, steps=2000, noise_std=0.2, seed=0):
    torch.manual_seed(seed)
    a = torch.logspace(0, 4, d, device=device, dtype=dtype)
    x0 = torch.randn(d, device=device, dtype=dtype)

    def run(name: str):
        x = x0.clone().requires_grad_(True)
        if name == "ADAM":
            opt = torch.optim.Adam([x], lr=1e-2)
        else:
            opt = torch.optim.AdamW([x], lr=1e-2)

        losses = []
        xs = []
        for t in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss = 0.5 * (a * x * x).sum()
            loss.backward()
            with torch.no_grad():
                x.grad.add_(torch.randn_like(x.grad) * noise_std)
            opt.step()
            if t % 50 == 0:
                xs.append(t)
                losses.append(float(loss.item()))
        return xs, losses

    x_adam, l_adam = run("ADAM")
    x_adamw, l_adamw = run("ADAMW")

    hist = {
        "ADAM": {"steps": x_adam, "loss": l_adam},
        "ADAMW": {"steps": x_adamw, "loss": l_adamw},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(hist, out_dir / "history.json")

    # 2D
    plt.figure()
    plt.plot(x_adam, l_adam, label="ADAM")
    plt.plot(x_adamw, l_adamw, label="ADAMW")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Noisy ill-conditioned quadratic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=160)
    plt.close()

    # 3D (step × optimizer × loss)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x_adam,  [0]*len(x_adam),  l_adam,  linewidth=2, label="ADAM")
    ax.plot(x_adamw, [1]*len(x_adamw), l_adamw, linewidth=2, label="ADAMW")

    ax.set_xlabel("Step")
    ax.set_ylabel("Optimizer")
    ax.set_zlabel("Loss")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ADAM", "ADAMW"])
    ax.set_title("Noisy ill-conditioned quadratic (3D)")
    plt.tight_layout()
    plt.savefig(out_dir / "loss3d.png", dpi=180)
    plt.close(fig)

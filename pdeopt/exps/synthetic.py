from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        for t in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss = 0.5 * (a * x * x).sum()
            loss.backward()
            with torch.no_grad():
                x.grad.add_(torch.randn_like(x.grad) * noise_std)
            opt.step()
            if t % 50 == 0:
                losses.append(float(loss.item()))
        return losses

    hist = {"ADAM": run("ADAM"), "ADAMW": run("ADAMW")}
    save_json(hist, out_dir / "history.json")

    plt.figure()
    for k, v in hist.items():
        xs = list(range(50, steps + 1, 50))
        plt.plot(xs[:len(v)], v, label=k)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Noisy ill-conditioned quadratic (sanity check)")
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "loss.png", dpi=160)
    plt.close()

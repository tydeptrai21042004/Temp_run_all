from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_curves(history_by_name: Dict[str, List[dict]], out_dir: Path, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # loss
    plt.figure()
    for name, hist in history_by_name.items():
        xs = list(range(1, len(hist) + 1))
        ys = [h.get("test_loss", float("nan")) for h in hist]
        plt.plot(xs, ys, label=name)
    plt.xlabel("epoch")
    plt.ylabel("test_loss")
    plt.title(title + " (loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=160)
    plt.close()

    # acc
    has_acc = any(("test_acc" in h) for hist in history_by_name.values() for h in hist)
    if has_acc:
        plt.figure()
        for name, hist in history_by_name.items():
            xs = list(range(1, len(hist) + 1))
            ys = [h.get("test_acc", float("nan")) for h in hist]
            plt.plot(xs, ys, label=name)
        plt.xlabel("epoch")
        plt.ylabel("test_acc")
        plt.title(title + " (accuracy)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "acc.png", dpi=160)
        plt.close()

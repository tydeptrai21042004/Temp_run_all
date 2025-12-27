
# pdeopt/core/plot3d.py
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_3d_loss_curves(
    histories: dict,
    out_dir: Path,
    title: str,
    key: str = "test_loss",
    filename: str = "loss3d_test.png",
):
    """
    3D plot:
      x = epoch
      y = optimizer index
      z = loss (train_loss or test_loss)

    Saves: out_dir/filename
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(histories.keys())
    if not names:
        return

    # Collect epochs and losses
    series = []
    max_ep = 0
    for i, name in enumerate(names):
        hist = histories[name]
        xs, zs = [], []
        for row in hist:
            if "epoch" in row and key in row:
                xs.append(int(row["epoch"]))
                zs.append(float(row[key]))
        if xs:
            max_ep = max(max_ep, max(xs))
        series.append((i, name, xs, zs))

    if max_ep == 0:
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, name, xs, zs in series:
        if not xs:
            continue
        ys = np.full(len(xs), i, dtype=np.float32)
        ax.plot(xs, ys, zs, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Optimizer")
    ax.set_zlabel(key)
    ax.set_title(title)

    # y ticks = optimizer names
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close(fig)

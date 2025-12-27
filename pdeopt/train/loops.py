import time
from typing import Optional, Dict
import torch
import torch.nn.functional as F


def _maybe_sync_cuda(device):
    # For accurate wall-clock on GPU (optional, but best for benchmark tables)
    if isinstance(device, str):
        is_cuda = device.startswith("cuda")
    else:
        is_cuda = getattr(device, "type", "") == "cuda"
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _pull_opt_epoch_stats(opt) -> Dict[str, float]:
    """
    Collect optional per-epoch stats from optimizer (rho_mean, smoothness metrics, pde_time_ms, etc.)
    and reset them for next epoch.
    """
    stats: Dict[str, float] = {}
    if hasattr(opt, "get_epoch_stats") and callable(getattr(opt, "get_epoch_stats")):
        s = opt.get_epoch_stats() or {}
        # prefix to avoid name collision
        for k, v in s.items():
            stats[f"opt_{k}"] = float(v) if isinstance(v, (int, float)) else v
    if hasattr(opt, "reset_epoch_stats") and callable(getattr(opt, "reset_epoch_stats")):
        opt.reset_epoch_stats()
    return stats


@torch.no_grad()
def evaluate_classifier(model, loader, device, dtype, max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    tot_loss = 0.0
    tot = 0
    correct = 0
    n_batches = 0
    for batch in loader:
        n_batches += 1
        if max_batches is not None and n_batches > max_batches:
            break
        x, y = batch
        x = x.to(device, dtype=dtype)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        tot_loss += loss.item() * y.size(0)
        tot += y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
    return {"loss": tot_loss / max(1, tot), "acc": correct / max(1, tot)}


@torch.no_grad()
def evaluate_imdb(model, loader, device, dtype, max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    tot_loss = 0.0
    tot = 0
    correct = 0
    n_batches = 0
    for (idx, offsets, y) in loader:
        n_batches += 1
        if max_batches is not None and n_batches > max_batches:
            break
        idx = idx.to(device)
        offsets = offsets.to(device)
        y = y.to(device)
        logits = model(idx, offsets)
        loss = F.cross_entropy(logits, y)
        tot_loss += loss.item() * y.size(0)
        tot += y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
    return {"loss": tot_loss / max(1, tot), "acc": correct / max(1, tot)}


def train_one_epoch_classifier(
    model,
    loader,
    opt,
    device,
    dtype,
    max_batches: Optional[int] = None,
    sync_cuda_time: bool = False,   # ✅ benchmark flag
) -> Dict[str, float]:
    model.train()
    tot_loss = 0.0
    tot = 0
    correct = 0
    n_batches = 0

    if sync_cuda_time:
        _maybe_sync_cuda(device)
    t0 = time.perf_counter()

    for batch in loader:
        n_batches += 1
        if max_batches is not None and n_batches > max_batches:
            break
        x, y = batch
        x = x.to(device, dtype=dtype)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        tot_loss += loss.item() * y.size(0)
        tot += y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()

    if sync_cuda_time:
        _maybe_sync_cuda(device)
    dt = time.perf_counter() - t0

    out = {
        "train_loss": tot_loss / max(1, tot),
        "train_acc": correct / max(1, tot),
        "time_s": dt,
        "batches": n_batches,
    }
    out.update(_pull_opt_epoch_stats(opt))   # ✅ logs opt_rho_mean / opt_r_hf_* / opt_pde_time_ms ...
    return out


def train_one_epoch_imdb(
    model,
    loader,
    opt,
    device,
    dtype,
    max_batches: Optional[int] = None,
    sync_cuda_time: bool = False,   # ✅ benchmark flag
) -> Dict[str, float]:
    model.train()
    tot_loss = 0.0
    tot = 0
    correct = 0
    n_batches = 0

    if sync_cuda_time:
        _maybe_sync_cuda(device)
    t0 = time.perf_counter()

    for (idx, offsets, y) in loader:
        n_batches += 1
        if max_batches is not None and n_batches > max_batches:
            break
        idx = idx.to(device)
        offsets = offsets.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(idx, offsets)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        tot_loss += loss.item() * y.size(0)
        tot += y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()

    if sync_cuda_time:
        _maybe_sync_cuda(device)
    dt = time.perf_counter() - t0

    out = {
        "train_loss": tot_loss / max(1, tot),
        "train_acc": correct / max(1, tot),
        "time_s": dt,
        "batches": n_batches,
    }
    out.update(_pull_opt_epoch_stats(opt))
    return out

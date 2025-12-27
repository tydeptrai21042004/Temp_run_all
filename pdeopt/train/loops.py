import time
from typing import Optional, Dict
import torch
import torch.nn.functional as F

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

def train_one_epoch_classifier(model, loader, opt, device, dtype, max_batches: Optional[int] = None):
    model.train()
    tot_loss = 0.0
    tot = 0
    correct = 0
    n_batches = 0
    t0 = time.time()

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

    dt = time.time() - t0
    return {"train_loss": tot_loss / max(1, tot), "train_acc": correct / max(1, tot), "time_s": dt}

def train_one_epoch_imdb(model, loader, opt, device, dtype, max_batches: Optional[int] = None):
    model.train()
    tot_loss = 0.0
    tot = 0
    correct = 0
    n_batches = 0
    t0 = time.time()

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

    dt = time.time() - t0
    return {"train_loss": tot_loss / max(1, tot), "train_acc": correct / max(1, tot), "time_s": dt}

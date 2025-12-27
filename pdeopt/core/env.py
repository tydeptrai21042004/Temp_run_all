import os
from pathlib import Path
import torch

def get_device(prefer: str = "auto") -> torch.device:
    """
    Auto-select device:
      - prefer="auto": cuda if available else mps (mac) else cpu
      - prefer="cuda": force cuda (error if unavailable)
      - prefer="cpu":  force cpu
      - prefer="mps":  force mps (error if unavailable)

    You can also override via env var:
      PDEOPT_DEVICE=cuda|cpu|mps|auto
    """
    prefer = (prefer or "auto").lower().strip()
    env_pref = os.getenv("PDEOPT_DEVICE", "").lower().strip()
    if env_pref:
        prefer = env_pref

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")

    if prefer == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """Print useful runtime info for debugging / reproducibility."""
    print(f"[INFO] device={device}")

    if device.type == "cuda":
        # Speed-up for CNNs with fixed input sizes
        torch.backends.cudnn.benchmark = True

        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        mem_total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"[INFO] CUDA device[{idx}] = {name}")
        print(f"[INFO] CUDA capability = {cap[0]}.{cap[1]}")
        print(f"[INFO] CUDA total memory = {mem_total_gb:.2f} GB")
        print(f"[INFO] cudnn.enabled={torch.backends.cudnn.enabled} benchmark={torch.backends.cudnn.benchmark}")

    elif device.type == "mps":
        print(f"[INFO] MPS available={torch.backends.mps.is_available()}")

    else:
        print("[INFO] Using CPU")


def get_dtype():
    return torch.float32


def default_results_dir() -> Path:
    p = Path(os.environ.get("RESULTS_DIR", "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p

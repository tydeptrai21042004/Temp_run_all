import os
from pathlib import Path
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dtype():
    return torch.float32

def default_results_dir() -> Path:
    p = Path(os.environ.get("RESULTS_DIR", "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p

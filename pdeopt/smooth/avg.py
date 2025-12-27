import torch
import torch.nn.functional as F

_AVG_K3 = torch.tensor([[0.0, 1.0, 0.0],
                        [1.0, 4.0, 1.0],
                        [0.0, 1.0, 0.0]], dtype=torch.float32) / 8.0

def apply_spatial_avg(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Cheap local averaging on natural 2D topology when possible.
    Used for:
      - preconditioner smoothing baseline (AdamW-LRMP)
      - gradient smoothing baseline (LSGD)
    """
    if x.ndim == 4:
        out, inp, kH, kW = x.shape
        xx = x.reshape(out * inp, 1, kH, kW)
        k = _AVG_K3.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        yy = F.conv2d(xx, k, padding=1)
        return yy.reshape_as(x)

    if x.ndim == 2 and x.shape[1] == 28 * 28:
        out = x.shape[0]
        xx = x.reshape(out, 1, 28, 28)
        k = _AVG_K3.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        yy = F.conv2d(xx, k, padding=1)
        return yy.reshape_as(x)

    return x

import torch

def hf_energy_2d(x: torch.Tensor, keep_dc: bool = False) -> float:
    """
    High-frequency energy of a 2D field (last two dims). Useful to quantify smoothing.
    """
    if x.ndim < 2:
        return float("nan")
    X = torch.fft.fft2(x, dim=(-2, -1))
    P = (X.real**2 + X.imag**2)
    if not keep_dc:
        P = P.clone()
        P[..., 0, 0] = 0.0
    return float(P.mean().detach().cpu().item())

def lap_energy_2d(x: torch.Tensor) -> float:
    """
    Discrete Laplacian energy ||Δx||_2^2 averaged (last two dims).
    """
    if x.ndim < 2:
        return float("nan")
    up = torch.roll(x, shifts=1, dims=-2)
    dn = torch.roll(x, shifts=-1, dims=-2)
    lf = torch.roll(x, shifts=1, dims=-1)
    rt = torch.roll(x, shifts=-1, dims=-1)
    lap = (up + dn + lf + rt - 4.0 * x)
    return float((lap * lap).mean().detach().cpu().item())

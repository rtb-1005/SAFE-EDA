from __future__ import annotations

import torch
from torch.nn import functional as F


def _grouped_kernel(signal: torch.Tensor, kernel: torch.Tensor) -> tuple[torch.Tensor, int]:
    if signal.ndim != 3 or signal.shape[1] != 1:
        raise ValueError("signal must have shape [B, 1, T]")
    if kernel.ndim != 3 or kernel.shape[1] != 1:
        raise ValueError("kernel must have shape [1 or B, 1, L]")
    batch = signal.shape[0]
    if kernel.shape[0] not in {1, batch}:
        raise ValueError("kernel batch dimension must be 1 or match signal")
    if kernel.shape[0] == 1:
        return kernel, 1
    return kernel, batch


def causal_convolution(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply y[t] = sum_l kernel[l] * signal[t-l] with zero prehistory."""
    kernel, groups = _grouped_kernel(signal, kernel)
    length = kernel.shape[-1]
    if groups == 1:
        padded = F.pad(signal, (length - 1, 0))
        return F.conv1d(padded, kernel.flip(-1))
    batch = signal.shape[0]
    padded = F.pad(signal, (length - 1, 0)).reshape(1, batch, -1)
    output = F.conv1d(padded, kernel, groups=batch)
    return output.reshape(batch, 1, -1)


def causal_convolution_adjoint(error: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply the exact finite-window adjoint of :func:`causal_convolution`."""
    kernel, groups = _grouped_kernel(error, kernel)
    length = kernel.shape[-1]
    if groups == 1:
        padded = F.pad(error, (0, length - 1))
        return F.conv1d(padded, kernel)
    batch = error.shape[0]
    padded = F.pad(error, (0, length - 1)).reshape(1, batch, -1)
    output = F.conv1d(padded, kernel, groups=batch)
    return output.reshape(batch, 1, -1)


def moving_average(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError("values must have shape [B, 1, T]")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    padding = kernel_size // 2
    padded = F.pad(values, (padding, padding), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1e-6)

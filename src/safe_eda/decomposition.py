from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from safe_eda.ops import causal_convolution, causal_convolution_adjoint, moving_average


def build_response_kernel(
    *,
    sample_rate_hz: float,
    support_seconds: float,
    tau_rise_seconds: float,
    tau_decay_seconds: float,
) -> torch.Tensor:
    if sample_rate_hz <= 0 or support_seconds <= 0:
        raise ValueError("sample rate and support must be positive")
    if not 0 < tau_rise_seconds < tau_decay_seconds:
        raise ValueError("tau_rise must be positive and less than tau_decay")
    length = max(3, int(round(sample_rate_hz * support_seconds)) + 1)
    time = torch.arange(length, dtype=torch.float32) / float(sample_rate_hz)
    response = torch.exp(-time / tau_decay_seconds) - torch.exp(-time / tau_rise_seconds)
    response = response.clamp_min(0.0)
    response = response / response.sum().clamp_min(torch.finfo(response.dtype).eps)
    return response.view(1, 1, -1)


class TonicProjector(nn.Module):
    """Fixed low-rank smooth projection; it has no task-trained waveform decoder."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        window_samples: int,
        knot_seconds: float,
        smoothness: float,
    ) -> None:
        super().__init__()
        if window_samples < 4 or sample_rate_hz <= 0 or knot_seconds <= 0:
            raise ValueError("invalid tonic projector dimensions")
        knots = max(3, int(round(window_samples / (sample_rate_hz * knot_seconds))) + 1)
        identity = torch.eye(knots, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        basis = F.interpolate(identity, size=window_samples, mode="linear", align_corners=True)
        basis = basis.squeeze(0).transpose(0, 1)
        second = torch.zeros(knots - 2, knots, dtype=torch.float32)
        for index in range(knots - 2):
            second[index, index : index + 3] = torch.tensor([1.0, -2.0, 1.0])
        system = basis.transpose(0, 1) @ basis + float(smoothness) * (second.transpose(0, 1) @ second)
        projection = basis @ torch.linalg.solve(system, basis.transpose(0, 1))
        self.register_buffer("projection", projection)
        self.knots = knots

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 1 or signal.shape[-1] != self.projection.shape[0]:
            raise ValueError("signal has incompatible tonic projector shape")
        return torch.matmul(signal, self.projection.transpose(0, 1))


@dataclass
class DecompositionOutput:
    tonic: torch.Tensor
    driver: torch.Tensor
    phasic: torch.Tensor
    reconstruction: torch.Tensor
    residual: torch.Tensor
    kernel: torch.Tensor
    tau_rise: torch.Tensor
    tau_decay: torch.Tensor
    confidence_t: torch.Tensor
    confidence_global: torch.Tensor


class ConstrainedDecomposer(nn.Module):
    """Deterministic constrained tonic/driver/response decomposition.

    The module deliberately exposes no free artifact or residual decoder.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        window_samples: int,
        tonic_knot_seconds: float = 2.0,
        tonic_smoothness: float = 10.0,
        kernel_support_seconds: float = 4.0,
        tau_rise_seconds: float = 0.75,
        tau_decay_seconds: float = 2.5,
        ista_iterations: int = 20,
        ista_step_size: float = 0.9,
        driver_l1: float = 0.02,
        confidence_kappa: float = 1.0,
    ) -> None:
        super().__init__()
        if ista_iterations <= 0 or ista_step_size <= 0 or driver_l1 < 0:
            raise ValueError("invalid ISTA configuration")
        self.sample_rate_hz = float(sample_rate_hz)
        self.window_samples = int(window_samples)
        self.ista_iterations = int(ista_iterations)
        self.ista_step_size = float(ista_step_size)
        self.driver_l1 = float(driver_l1)
        self.confidence_kappa = float(confidence_kappa)
        self.tonic = TonicProjector(
            sample_rate_hz=sample_rate_hz,
            window_samples=window_samples,
            knot_seconds=tonic_knot_seconds,
            smoothness=tonic_smoothness,
        )
        self.register_buffer(
            "kernel",
            build_response_kernel(
                sample_rate_hz=sample_rate_hz,
                support_seconds=kernel_support_seconds,
                tau_rise_seconds=tau_rise_seconds,
                tau_decay_seconds=tau_decay_seconds,
            ),
        )
        self.register_buffer("tau_rise_value", torch.tensor(float(tau_rise_seconds)))
        self.register_buffer("tau_decay_value", torch.tensor(float(tau_decay_seconds)))

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
    ) -> DecompositionOutput:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != self.window_samples:
            raise ValueError("x must have shape [B, 1, window_samples]")
        valid = torch.ones_like(x) if valid_mask is None else valid_mask.to(dtype=x.dtype)
        q = valid if quality is None else quality.to(dtype=x.dtype) * valid
        tonic = self.tonic(x)
        residual0 = x - tonic
        driver = torch.zeros_like(x)
        weight = q.square()
        for _ in range(self.ista_iterations):
            predicted = causal_convolution(driver, self.kernel.to(dtype=x.dtype))
            gradient = causal_convolution_adjoint(
                (predicted - residual0) * weight,
                self.kernel.to(dtype=x.dtype),
            )
            driver = F.relu(
                driver - self.ista_step_size * gradient - self.ista_step_size * self.driver_l1
            )
        phasic = causal_convolution(driver, self.kernel.to(dtype=x.dtype))
        reconstruction = tonic + phasic
        residual = x - reconstruction
        error = moving_average(
            residual.abs(),
            kernel_size=max(3, int(round(self.sample_rate_hz * 2.0)) // 2 * 2 + 1),
        )
        confidence_t = valid * q * torch.exp(-error / max(self.confidence_kappa, 1e-6))
        confidence_global = (
            (confidence_t * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1e-6)
        )
        batch = x.shape[0]
        tau_rise = self.tau_rise_value.to(dtype=x.dtype).expand(batch, 1)
        tau_decay = self.tau_decay_value.to(dtype=x.dtype).expand(batch, 1)
        return DecompositionOutput(
            tonic=tonic,
            driver=driver,
            phasic=phasic,
            reconstruction=reconstruction,
            residual=residual,
            kernel=self.kernel.to(dtype=x.dtype).expand(batch, -1, -1),
            tau_rise=tau_rise,
            tau_decay=tau_decay,
            confidence_t=confidence_t,
            confidence_global=confidence_global,
        )

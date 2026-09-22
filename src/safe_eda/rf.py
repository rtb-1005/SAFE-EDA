from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop path probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class SafeRFBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        physio_channels: int,
        sample_rate_hz: float,
        receptive_field_seconds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        depthwise_kernel_size: int = 7,
        gate_hidden_dim: int = 32,
        dropout: float = 0.1,
        drop_path_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if channels < 4 or channels % 4 or len(receptive_field_seconds) != 4:
            raise ValueError("channels must be divisible by four and four RFs are required")
        if depthwise_kernel_size % 2 == 0:
            raise ValueError("depthwise kernel size must be odd")
        self.channels = channels
        self.num_experts = 4
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 4, channels)
        dilations = tuple(
            max(
                1,
                int(round((field * sample_rate_hz - 1.0) / (depthwise_kernel_size - 1))),
            )
            for field in receptive_field_seconds
        )
        self.dilations = dilations
        self.experts = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=depthwise_kernel_size,
                    padding=(depthwise_kernel_size // 2) * dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in dilations
            ]
        )
        self.static_logits = nn.Parameter(torch.zeros(self.num_experts))
        self.gate = nn.Sequential(
            nn.Conv1d(physio_channels, gate_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(gate_hidden_dim, self.num_experts, kernel_size=1),
        )
        self.gate_temperature = 1.0
        self.beta_raw = nn.Parameter(torch.tensor(-6.0))
        self.mix_in = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.mix_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_probability)
        self.ffn_norm = nn.GroupNorm(8 if channels % 8 == 0 else 4, channels)
        self.ffn_in = nn.Conv1d(channels, 4 * channels, kernel_size=1)
        self.ffn_out = nn.Conv1d(4 * channels, channels, kernel_size=1)

    def forward(
        self,
        h: torch.Tensor,
        physio: torch.Tensor,
        confidence: torch.Tensor,
        *,
        force_static: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 3 or physio.ndim != 3 or confidence.ndim != 3:
            raise ValueError("SAFE-RF inputs must be rank-3 tensors")
        normalized = self.norm(h)
        expert_values = torch.stack([expert(normalized) for expert in self.experts], dim=1)
        static_weight = torch.softmax(self.static_logits, dim=0)
        static_gate = static_weight.view(1, self.num_experts, 1, 1).expand_as(expert_values)
        dynamic_gate = torch.softmax(self.gate(physio) / self.gate_temperature, dim=1)
        dynamic_gate = dynamic_gate.unsqueeze(2).expand_as(expert_values)
        confidence = confidence.clamp(0.0, 1.0).unsqueeze(1)
        if force_static:
            effective_gate = static_gate
        else:
            effective_gate = (1.0 - confidence) * static_gate + confidence * dynamic_gate
        static_mix = (expert_values * static_gate).sum(dim=1)
        dynamic_mix = (expert_values * effective_gate).sum(dim=1)
        beta = torch.sigmoid(self.beta_raw)
        mixed = static_mix if force_static else static_mix + beta * (dynamic_mix - static_mix)
        update = self.mix_out(self.dropout(F.glu(self.mix_in(mixed), dim=1)))
        h = h + self.drop_path(update)
        ffn = self.ffn_out(self.dropout(F.gelu(self.ffn_in(self.ffn_norm(h)))))
        h = h + self.drop_path(ffn)
        return h, {
            "static_weight": static_weight,
            "dynamic_gate": dynamic_gate,
            "effective_gate": effective_gate,
            "beta": beta.detach(),
            "dilations": torch.tensor(self.dilations, device=h.device),
        }

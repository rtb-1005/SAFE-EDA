from __future__ import annotations

import torch
from torch import nn

from safe_eda.decomposition import ConstrainedDecomposer
from safe_eda.ops import masked_mean
from safe_eda.pooling import build_physio_grid_features, build_physio_summary, pooled_features
from safe_eda.rf import SafeRFBlock


class SafeEDAClassifier(nn.Module):
    """Classify normalized EDA windows shaped ``[B, 1, T]``.

    Returns class logits, embeddings, hidden features, and decomposition
    diagnostics. The validity channel is all ones, and signal decomposition
    is detached from task gradients.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        window_samples: int,
        num_classes: int,
        model_dim: int = 64,
        num_blocks: int = 6,
        depthwise_kernel_size: int = 7,
        receptive_field_seconds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        gate_hidden_dim: int = 32,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        drop_path_max: float = 0.1,
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
        if num_classes < 2 or model_dim < 8 or model_dim % 8:
            raise ValueError("invalid classifier dimensions")
        self.model_dim = model_dim
        self.decomposer = ConstrainedDecomposer(
            sample_rate_hz=sample_rate_hz,
            window_samples=window_samples,
            tonic_knot_seconds=tonic_knot_seconds,
            tonic_smoothness=tonic_smoothness,
            kernel_support_seconds=kernel_support_seconds,
            tau_rise_seconds=tau_rise_seconds,
            tau_decay_seconds=tau_decay_seconds,
            ista_iterations=ista_iterations,
            ista_step_size=ista_step_size,
            driver_l1=driver_l1,
            confidence_kappa=confidence_kappa,
        )
        self.stem = nn.Sequential(
            nn.Conv1d(4, model_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, model_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                SafeRFBlock(
                    channels=model_dim,
                    physio_channels=9,
                    sample_rate_hz=sample_rate_hz,
                    receptive_field_seconds=receptive_field_seconds,
                    depthwise_kernel_size=depthwise_kernel_size,
                    gate_hidden_dim=gate_hidden_dim,
                    dropout=dropout,
                    drop_path_probability=drop_path_max * (index + 1) / max(1, num_blocks),
                )
                for index in range(num_blocks)
            ]
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.event_mlp = nn.Sequential(
            nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Linear(2 * model_dim, model_dim),
        )
        self.phys_mlp = nn.Sequential(
            nn.Linear(10, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.alpha_event_raw = nn.Parameter(torch.tensor(-4.0))
        self.alpha_phys_raw = nn.Parameter(torch.tensor(-4.0))
        self.final_norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(model_dim, num_classes),
        )

    @staticmethod
    def _derivative(values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(values[..., 1:] - values[..., :-1], (1, 0))

    def forward(self, x: torch.Tensor, *, force_static: bool = False) -> dict[str, object]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("SAFE-EDA expects [B, 1, T] input")
        valid = torch.ones_like(x)
        with torch.no_grad():
            decomposition = self.decomposer(x, valid_mask=valid, quality=valid)
            physio = build_physio_grid_features(decomposition, valid)
        dx = self._derivative(x)
        ddx = self._derivative(dx)
        hidden = self.stem(torch.cat([x, dx, ddx, valid], dim=1))
        gate_maps: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            hidden, info = block(
                hidden,
                physio,
                decomposition.confidence_t,
                force_static=force_static,
            )
            gate_maps.append(info)
        mean = masked_mean(hidden, valid.expand(-1, hidden.shape[1], -1))
        centered = hidden - mean.unsqueeze(-1)
        std = torch.sqrt(masked_mean(centered.square(), valid.expand(-1, hidden.shape[1], -1)) + 1e-6)
        z_global = self.global_mlp(torch.cat([mean, std], dim=1))
        z_event = self.event_mlp(pooled_features(hidden, decomposition, valid))
        z_phys = self.phys_mlp(build_physio_summary(decomposition, valid))
        c_global = decomposition.confidence_global
        event_confidence = c_global * (1.0 - torch.exp(-decomposition.driver.mean(dim=-1)))
        if force_static:
            event_scale = torch.zeros_like(c_global)
            phys_scale = torch.zeros_like(c_global)
        else:
            event_scale = torch.sigmoid(self.alpha_event_raw) * event_confidence
            phys_scale = torch.sigmoid(self.alpha_phys_raw) * c_global
        embedding = self.final_norm(z_global + event_scale * z_event + phys_scale * z_phys)
        return {
            "logits": self.head(embedding),
            "embedding": embedding,
            "hidden": hidden,
            "decomposition": decomposition,
            "quality": valid,
            "gate_maps": gate_maps,
            "event_scale": event_scale,
            "phys_scale": phys_scale,
        }

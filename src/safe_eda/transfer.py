from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from safe_eda.model import SafeEDAClassifier


TRANSFER_PREFIXES = ("stem.", "blocks.")


class ArtifactSafeEDA(nn.Module):
    """SAFE-EDA backbone with a source-only artifact-mask supervision head.

    The head predicts a binary mask on EDABE only. It never reconstructs an
    artifact waveform and is not present in either downstream classifier.
    """

    def __init__(self, backbone: SafeEDAClassifier) -> None:
        super().__init__()
        self.backbone = backbone
        self.artifact_head = nn.Conv1d(backbone.model_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        hidden = output["hidden"]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("SAFE-EDA backbone did not return a tensor hidden state")
        return self.artifact_head(hidden).squeeze(1)

    def transfer_state(self) -> dict[str, torch.Tensor]:
        """Return only trainable, resolution-independent SAFE temporal weights."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.backbone.state_dict().items()
            if name.startswith(TRANSFER_PREFIXES)
        }


def load_transfer_state(
    target: SafeEDAClassifier,
    state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    """Load EDABE-trained temporal weights while keeping target heads fresh."""
    if not state or any(not name.startswith(TRANSFER_PREFIXES) for name in state):
        raise ValueError("transfer state must contain only SAFE temporal trunk weights")
    incompatible = target.load_state_dict(state, strict=False)
    unexpected = tuple(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"unexpected transfer keys: {unexpected}")
    loaded = tuple(sorted(state))
    if not loaded:
        raise ValueError("no temporal transfer weights were loaded")
    return loaded


def assert_transfer_prefixes(names: Iterable[str]) -> None:
    invalid = [name for name in names if not name.startswith(TRANSFER_PREFIXES)]
    if invalid:
        raise ValueError(f"non-temporal transfer keys: {invalid}")

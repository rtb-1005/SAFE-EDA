from __future__ import annotations

import torch
from torch.nn import functional as F

from safe_eda.decomposition import DecompositionOutput
from safe_eda.ops import masked_mean


def normalize_weight(weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = weight.clamp_min(0.0) * mask
    return weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _derivative(values: torch.Tensor) -> torch.Tensor:
    return F.pad(values[..., 1:] - values[..., :-1], (1, 0))


def response_pool_weights(
    decomposition: DecompositionOutput, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence = decomposition.confidence_t
    driver = decomposition.driver.clamp_min(0.0)
    phasic = decomposition.phasic.clamp_min(0.0)
    derivative = _derivative(decomposition.phasic)
    rise = derivative.relu()
    recovery = (-derivative).relu()
    return tuple(
        normalize_weight(confidence * value, valid_mask)
        for value in (driver, phasic, rise, recovery)
    )


def pooled_features(
    hidden: torch.Tensor,
    decomposition: DecompositionOutput,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    weights = response_pool_weights(decomposition, valid_mask)
    return torch.cat([(hidden * weight).sum(dim=-1) for weight in weights], dim=-1)


def build_physio_grid_features(
    decomposition: DecompositionOutput, valid_mask: torch.Tensor
) -> torch.Tensor:
    driver = decomposition.driver
    phasic = decomposition.phasic
    driver_scale = torch.quantile(driver.detach().abs().flatten(1), 0.95, dim=1).view(-1, 1, 1).clamp_min(1e-4)
    phasic_scale = torch.quantile(phasic.detach().abs().flatten(1), 0.95, dim=1).view(-1, 1, 1).clamp_min(1e-4)
    d_phasic = _derivative(phasic)
    d_tonic = _derivative(decomposition.tonic)
    tau_rise = decomposition.tau_rise.unsqueeze(-1).expand_as(driver)
    tau_decay = decomposition.tau_decay.unsqueeze(-1).expand_as(driver)
    features = torch.cat(
        [
            driver / driver_scale,
            phasic / phasic_scale,
            d_phasic,
            d_tonic,
            decomposition.residual.abs(),
            valid_mask,
            decomposition.confidence_t,
            tau_rise / 4.0,
            tau_decay / 4.0,
        ],
        dim=1,
    )
    return features.detach()


def build_physio_summary(
    decomposition: DecompositionOutput, valid_mask: torch.Tensor
) -> torch.Tensor:
    driver = decomposition.driver
    phasic = decomposition.phasic
    tonic = decomposition.tonic
    tonic_slope = _derivative(tonic).abs()
    summary = torch.cat(
        [
            decomposition.tau_rise / 4.0,
            decomposition.tau_decay / 4.0,
            masked_mean(tonic, valid_mask),
            masked_mean(tonic_slope, valid_mask),
            masked_mean(driver, valid_mask),
            (driver * valid_mask).amax(dim=-1).squeeze(1).unsqueeze(1),
            masked_mean(phasic.square(), valid_mask),
            masked_mean((driver > 1e-3).to(driver.dtype), valid_mask),
            masked_mean(decomposition.residual.abs(), valid_mask),
            masked_mean(decomposition.confidence_t, valid_mask),
        ],
        dim=1,
    )
    return summary.detach()

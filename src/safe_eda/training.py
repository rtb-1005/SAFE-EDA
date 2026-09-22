from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from safe_eda.model import SafeEDAClassifier
from safe_eda.transfer import ArtifactSafeEDA, load_transfer_state


ProgressCallback = Callable[[dict[str, float | int]], None]


class ClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.labels[index]


class ArtifactDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, target: np.ndarray) -> None:
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.target = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.target[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_safe_model(config: dict[str, Any]) -> SafeEDAClassifier:
    model = config["model"]
    input_config = config["target_input"]
    return SafeEDAClassifier(
        sample_rate_hz=float(input_config["sample_rate_hz"]),
        window_samples=int(input_config["window_samples"]),
        num_classes=3,
        model_dim=int(model["model_dim"]),
        num_blocks=int(model["num_blocks"]),
        depthwise_kernel_size=int(model["depthwise_kernel_size"]),
        receptive_field_seconds=tuple(float(value) for value in model["receptive_field_seconds"]),
        gate_hidden_dim=int(model["gate_hidden_dim"]),
        dropout=float(model["dropout"]),
        head_dropout=float(model["head_dropout"]),
        drop_path_max=float(model["drop_path_max"]),
        tonic_knot_seconds=float(model["tonic_knot_seconds"]),
        tonic_smoothness=float(model["tonic_smoothness"]),
        kernel_support_seconds=float(model["kernel_support_seconds"]),
        tau_rise_seconds=float(model["tau_rise_seconds"]),
        tau_decay_seconds=float(model["tau_decay_seconds"]),
        ista_iterations=int(model["ista_iterations"]),
        ista_step_size=float(model["ista_step_size"]),
        driver_l1=float(model["driver_l1"]),
        confidence_kappa=float(model["confidence_kappa"]),
    )


def build_source_model(config: dict[str, Any]) -> ArtifactSafeEDA:
    model = config["model"]
    input_config = config["source_input"]
    backbone = SafeEDAClassifier(
        sample_rate_hz=float(input_config["sample_rate_hz"]),
        window_samples=int(input_config["window_samples"]),
        num_classes=3,
        model_dim=int(model["model_dim"]),
        num_blocks=int(model["num_blocks"]),
        depthwise_kernel_size=int(model["depthwise_kernel_size"]),
        receptive_field_seconds=tuple(float(value) for value in model["receptive_field_seconds"]),
        gate_hidden_dim=int(model["gate_hidden_dim"]),
        dropout=float(model["dropout"]),
        head_dropout=float(model["head_dropout"]),
        drop_path_max=float(model["drop_path_max"]),
        tonic_knot_seconds=float(model["tonic_knot_seconds"]),
        tonic_smoothness=float(model["tonic_smoothness"]),
        kernel_support_seconds=float(model["kernel_support_seconds"]),
        tau_rise_seconds=float(model["tau_rise_seconds"]),
        tau_decay_seconds=float(model["tau_decay_seconds"]),
        ista_iterations=int(model["ista_iterations"]),
        ista_step_size=float(model["ista_step_size"]),
        driver_l1=float(model["driver_l1"]),
        confidence_kappa=float(model["confidence_kappa"]),
    )
    return ArtifactSafeEDA(backbone)


def _loader(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], *, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("all three native labels must occur during target training")
    return torch.from_numpy((counts.sum() / (3.0 * counts)).astype(np.float32))


def _diagnostics(output: dict[str, object]) -> dict[str, float]:
    decomposition = output["decomposition"]
    gate_maps = output["gate_maps"]
    return {
        "confidence_global_mean": float(decomposition.confidence_global.mean().detach().cpu()),
        "driver_nonzero_ratio": float((decomposition.driver > 1e-3).float().mean().detach().cpu()),
        "event_scale_mean": float(output["event_scale"].mean().detach().cpu()),
        "phys_scale_mean": float(output["phys_scale"].mean().detach().cpu()),
        "beta_mean": float(torch.stack([entry["beta"] for entry in gate_maps]).mean().detach().cpu()),
    }


def train_artifact(
    model: ArtifactSafeEDA,
    x: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    progress: ProgressCallback | None,
) -> list[float]:
    seed_everything(seed)
    training = config["training"]
    dataset = ArtifactDataset(x, target)
    loader = _loader(dataset, batch_size=int(training["batch_size"]), seed=seed)
    positives = float((target >= 0.5).sum())
    negatives = float(target.size - positives)
    pos_weight = torch.tensor(max(1.0, negatives / max(1.0, positives)), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    losses: list[float] = []
    for epoch in range(1, int(training["artifact_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for inputs, masks in loader:
            logits = model(inputs.to(device, non_blocking=True))
            loss = criterion(logits, masks.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite artifact loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip_norm"]))
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite artifact gradient at epoch {epoch}")
            optimizer.step()
            batch = int(inputs.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch
            count += batch
        epoch_loss = loss_sum / max(1, count)
        losses.append(epoch_loss)
        if progress is not None:
            progress({"epoch": epoch, "epochs": int(training["artifact_epochs"]), "loss": epoch_loss})
    return losses


def train_classifier(
    model: SafeEDAClassifier,
    x: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    progress: ProgressCallback | None,
) -> tuple[list[float], dict[str, float]]:
    seed_everything(seed)
    training = config["training"]
    loader = _loader(
        ClassificationDataset(x, labels), batch_size=int(training["batch_size"]), seed=seed
    )
    criterion = nn.CrossEntropyLoss(weight=_class_weights(labels).to(device))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    losses: list[float] = []
    diagnostics: dict[str, float] = {}
    for epoch in range(1, int(training["target_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for inputs, targets in loader:
            output = model(inputs.to(device, non_blocking=True))
            loss = criterion(output["logits"], targets.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite target loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip_norm"]))
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite target gradient at epoch {epoch}")
            optimizer.step()
            batch = int(inputs.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch
            count += batch
            diagnostics = _diagnostics(output)
        epoch_loss = loss_sum / max(1, count)
        losses.append(epoch_loss)
        if progress is not None:
            progress({"epoch": epoch, "epochs": int(training["target_epochs"]), "loss": epoch_loss, **diagnostics})
    return losses, diagnostics


def save_source_checkpoint(path: Path, model: ArtifactSafeEDA, *, seed: int, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "artifact_training_losses": losses,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "transfer_state": model.transfer_state(),
        "architecture": "safe_eda_v1_source_artifact_head",
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_source_transfer_state(path: Path, *, seed: int) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"source checkpoint seed mismatch: {path}")
    state = payload.get("transfer_state")
    if not isinstance(state, dict):
        raise ValueError(f"source checkpoint lacks transfer state: {path}")
    return {str(name): value for name, value in state.items() if isinstance(value, torch.Tensor)}


def load_source_checkpoint(
    path: Path,
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> ArtifactSafeEDA:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"source checkpoint seed mismatch: {path}")
    model = build_source_model(config)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"source checkpoint lacks full model state: {path}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def initialize_target(
    config: dict[str, Any], *, seed: int, transfer_state: dict[str, torch.Tensor] | None
) -> tuple[SafeEDAClassifier, tuple[str, ...]]:
    seed_everything(seed)
    model = build_safe_model(config)
    loaded: tuple[str, ...] = ()
    if transfer_state is not None:
        loaded = load_transfer_state(model, transfer_state)
    return model, loaded


def save_target_checkpoint(
    path: Path,
    model: SafeEDAClassifier,
    *,
    arm: str,
    seed: int,
    losses: list[float],
    diagnostics: dict[str, float],
    loaded_transfer_keys: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm,
        "seed": int(seed),
        "fixed_final_checkpoint": True,
        "training_losses": losses,
        "diagnostics": diagnostics,
        "loaded_transfer_keys": list(loaded_transfer_keys),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_target_checkpoint(
    path: Path, config: dict[str, Any], *, arm: str, seed: int, device: torch.device
) -> SafeEDAClassifier:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"target checkpoint identity mismatch: {path}")
    model = build_safe_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_classifier(model: SafeEDAClassifier, x: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        ClassificationDataset(x, np.zeros(x.shape[0], dtype=np.int64)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    predictions: list[np.ndarray] = []
    for inputs, _ in loader:
        output = model(inputs.to(device, non_blocking=True))
        logits = output["logits"]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite target logits")
        predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False))
    return np.concatenate(predictions, axis=0)


@torch.no_grad()
def predict_artifact(model: ArtifactSafeEDA, x: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        ArtifactDataset(x, np.zeros((x.shape[0], x.shape[-1]), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    probabilities: list[np.ndarray] = []
    model.to(device)
    model.eval()
    for inputs, _ in loader:
        probabilities.append(torch.sigmoid(model(inputs.to(device, non_blocking=True))).cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(np.float32, copy=False)

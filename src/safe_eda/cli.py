"""Command-line model checks, training, prediction, and numerical reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _windows(path: Path, samples: int, target: str | None = None):
    """Read normalized EDA windows and optional labels from a local NPZ file."""
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float32)
        y = np.asarray(archive[target]) if target else None
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1:] != (1, samples):
        raise ValueError(f"x must have shape [N, 1, {samples}], with N > 0")
    if not np.isfinite(x).all():
        raise ValueError("x contains non-finite samples")
    if target == "y":
        if y.shape != (len(x),) or not np.isin(y, [0, 1, 2]).all():
            raise ValueError("y must contain one integer class in {0, 1, 2} per window")
        y = y.astype(np.int64)
    elif target == "target":
        if y.shape != (len(x), samples) or not np.isin(y, [0, 1]).all():
            raise ValueError(f"target must contain binary artifact labels with shape [N, {samples}]")
        y = y.astype(np.float32)
    return x, y


def _check(config):
    import torch
    from .training import build_safe_model, build_source_model, seed_everything
    from .transfer import load_transfer_state

    seed_everything(config["training"]["seeds"][0])
    target = build_safe_model(config).eval()
    source = build_source_model(config).eval()
    loaded = load_transfer_state(target, source.transfer_state())
    with torch.no_grad():
        logits = target(torch.zeros(2, 1, config["target_input"]["window_samples"]))["logits"]
        artifact_logits = source(torch.zeros(2, 1, config["source_input"]["window_samples"]))
    parameters = sum(p.numel() for p in target.parameters())
    transferred = sum(p.numel() for name, p in target.named_parameters()
                      if name.startswith(("stem.", "blocks.")))
    assert parameters == 359739 and transferred == 296822
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    assert artifact_logits.shape == (2, 256) and torch.isfinite(artifact_logits).all()
    assert loaded and all(name.startswith(("stem.", "blocks.")) for name in loaded)
    print(json.dumps({"status": "ok", "parameters": parameters,
                      "transferred_parameters": transferred,
                      "target_output_shape": list(logits.shape),
                      "source_output_shape": list(artifact_logits.shape)}, indent=2))


def _train(args, config):
    import torch
    from . import training

    device = torch.device(args.device)
    def progress(event):
        print(json.dumps(event), flush=True)

    if args.command == "train-source":
        x, y = _windows(args.data, config["source_input"]["window_samples"], "target")
        training.seed_everything(args.seed)
        model = training.build_source_model(config)
        losses = training.train_artifact(model, x, y, seed=args.seed, config=config,
                                         device=device, progress=progress)
        training.save_source_checkpoint(args.output, model, seed=args.seed, losses=losses)
    else:
        x, y = _windows(args.data, config["target_input"]["window_samples"], "y")
        state = (training.load_source_transfer_state(args.source, seed=args.seed)
                 if args.source else None)
        model, loaded = training.initialize_target(config, seed=args.seed, transfer_state=state)
        losses, diagnostics = training.train_classifier(
            model, x, y, seed=args.seed, config=config, device=device, progress=progress)
        training.save_target_checkpoint(
            args.output, model, arm="safe_edabe_transfer" if args.source else "safe_scratch",
            seed=args.seed, losses=losses, diagnostics=diagnostics, loaded_transfer_keys=loaded)
    print(f"Saved {args.output}")


def _predict(args, config):
    import torch
    from . import training

    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    source = "artifact_training_losses" in metadata
    samples = config["source_input" if source else "target_input"]["window_samples"]
    x, _ = _windows(args.data, samples)
    if source:
        model = training.load_source_checkpoint(
            args.checkpoint, config, seed=metadata["seed"], device=device)
        probabilities = training.predict_artifact(
            model, x, device=device, batch_size=config["training"]["batch_size"])
        arrays = {"artifact_probability": probabilities}
    else:
        model = training.load_target_checkpoint(
            args.checkpoint, config, arm=metadata["arm"], seed=metadata["seed"], device=device)
        labels = training.predict_classifier(
            model, x, device=device, batch_size=config["training"]["batch_size"])
        arrays = {"predicted_class": labels}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Saved {args.output}")


def main():
    parser = argparse.ArgumentParser(prog="safe-eda", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check model shapes and transferred parameters")
    reproduce = commands.add_parser("reproduce", help="Recompute experimental tables and numerical figures")
    reproduce.add_argument("--inputs", type=Path, default=Path("data/metrics.zip"))
    reproduce.add_argument("--literature", type=Path, default=Path("data/literature/study_coding.csv"))
    reproduce.add_argument("--output", type=Path, default=Path("outputs"))
    source = commands.add_parser("train-source", help="Train an EDABE pointwise artifact model")
    target = commands.add_parser("train-target", help="Train a target affect classifier")
    predict = commands.add_parser("predict", help="Predict with a saved source or target checkpoint")
    for command in [check, source, target, predict]:
        command.add_argument("--config", type=Path, default=Path("configs/model.json"))
    for command in [source, target, predict]:
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    for command in [source, target]:
        command.add_argument("--seed", type=int, default=1005)
    target.add_argument("--source", type=Path, help="Source checkpoint; omit to train from scratch")
    predict.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reproduce":
        from .analysis import build_tables
        from .plotting import plot_figures
        tables = build_tables(args.inputs, args.output / "tables", args.literature)
        figures = plot_figures(args.output / "tables", args.output / "figures")
        print(f"Wrote {len(tables)} tables and {len(figures)} figures to {args.output}")
        return
    config = json.loads(args.config.read_text())
    if args.command == "check":
        _check(config)
    elif args.command == "predict":
        _predict(args, config)
    else:
        _train(args, config)

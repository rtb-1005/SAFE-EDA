import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from safe_eda.cli import _windows
from safe_eda.training import build_safe_model, build_source_model, seed_everything
from safe_eda.transfer import load_transfer_state

ROOT = Path(__file__).resolve().parents[1]


class ModelSmokeTest(unittest.TestCase):
    def test_model_shapes_and_parameter_counts(self):
        config = json.loads((ROOT / "configs" / "model.json").read_text())
        seed_everything(config["training"]["seeds"][0])
        target = build_safe_model(config).eval()
        source = build_source_model(config).eval()
        head = target.head[0].weight.detach().clone()
        loaded = load_transfer_state(target, source.transfer_state())
        with torch.no_grad():
            target_out = target(torch.randn(2, 1, 240))
            source_out = source(torch.randn(2, 1, 256))
        self.assertEqual(sum(p.numel() for p in target.parameters()), 359739)
        self.assertEqual(
            sum(p.numel() for name, p in target.named_parameters()
                if name.startswith(("stem.", "blocks."))),
            296822,
        )
        self.assertEqual(tuple(target_out["logits"].shape), (2, 3))
        self.assertEqual(tuple(source_out.shape), (2, 256))
        self.assertTrue(loaded)
        self.assertTrue(all(name.startswith(("stem.", "blocks.")) for name in loaded))
        self.assertTrue(torch.equal(head, target.head[0].weight))
        self.assertTrue(torch.isfinite(target_out["logits"]).all())
        self.assertTrue(torch.isfinite(source_out).all())

    def test_training_and_prediction_commands(self):
        # Tiny synthetic inputs exercise the CLI, not experimental performance.
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            config = json.loads((ROOT / "configs" / "model.json").read_text())
            config["training"].update(artifact_epochs=1, target_epochs=1, batch_size=3)
            settings = work / "model.json"
            settings.write_text(json.dumps(config))
            rng = np.random.default_rng(1005)
            source_data, target_data = work / "source.npz", work / "target.npz"
            np.savez(source_data, x=rng.normal(size=(3, 1, 256)).astype(np.float32),
                     target=rng.integers(0, 2, size=(3, 256)))
            np.savez(target_data, x=rng.normal(size=(3, 1, 240)).astype(np.float32),
                     y=np.arange(3))

            def run(command, *arguments):
                result = subprocess.run(
                    [sys.executable, "-m", "safe_eda", command, "--config", str(settings),
                     *map(str, arguments)], cwd=ROOT, capture_output=True, text=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            source = work / "source.pt"
            run("train-source", "--data", source_data, "--output", source)
            source_checkpoint = torch.load(source, weights_only=True)
            self.assertEqual(len(source_checkpoint["artifact_training_losses"]), 1)
            self.assertTrue(np.isfinite(source_checkpoint["artifact_training_losses"]).all())
            source_prediction = work / "source_prediction.npz"
            run("predict", "--data", source_data, "--checkpoint", source,
                "--output", source_prediction)
            with np.load(source_prediction, allow_pickle=False) as result:
                probabilities = result["artifact_probability"]
                self.assertEqual(probabilities.shape, (3, 256))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

            for arm in ("scratch", "transfer"):
                with self.subTest(arm=arm):
                    checkpoint, prediction = work / f"{arm}.pt", work / f"{arm}.npz"
                    args = ["--data", target_data, "--output", checkpoint]
                    if arm == "transfer":
                        args += ["--source", source]
                    run("train-target", *args)
                    payload = torch.load(checkpoint, weights_only=True)
                    self.assertEqual(len(payload["training_losses"]), 1)
                    self.assertTrue(np.isfinite(payload["training_losses"]).all())
                    self.assertEqual(bool(payload["loaded_transfer_keys"]), arm == "transfer")
                    run("predict", "--data", target_data, "--checkpoint", checkpoint,
                        "--output", prediction)
                    with np.load(prediction, allow_pickle=False) as result:
                        self.assertEqual(result["predicted_class"].shape, (3,))
                        self.assertTrue(np.isin(result["predicted_class"], [0, 1, 2]).all())

    def test_invalid_window_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.npz"
            for x, y in [(np.zeros((3, 240)), np.arange(3)),
                         (np.full((3, 1, 240), np.nan), np.arange(3)),
                         (np.zeros((3, 1, 240)), np.array([0, 1, 3]))]:
                with self.subTest(shape=x.shape, labels=y.tolist()):
                    np.savez(path, x=x, y=y)
                    with self.assertRaises(ValueError):
                        _windows(path, 240, "y")

    def test_all_reference_tables(self):
        from safe_eda.analysis import build_tables

        with tempfile.TemporaryDirectory() as directory:
            tables = build_tables(
                ROOT / "data" / "metrics.zip", Path(directory),
                ROOT / "data" / "literature" / "study_coding.csv",
            )
            reference = sorted((ROOT / "data" / "results").glob("*.csv"))
            self.assertEqual(len(tables), 20)
            self.assertEqual({p.name for p in tables}, {p.name for p in reference})
            for table in tables:
                with self.subTest(table=table.name):
                    self.assertEqual(table.read_bytes(), (ROOT / "data/results" / table.name).read_bytes())


if __name__ == "__main__":
    unittest.main()

import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from catboost import CatBoostError
import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from datavidia.model import fit, macro_f1, partition, tune
from datavidia.submission import aligned


class SmokeTests(unittest.TestCase):
    def test_device_fallback_rules(self):
        with patch("datavidia.model.CatBoostClassifier") as classifier:
            classifier.return_value.fit.side_effect = [CatBoostError("CUDA driver error"), None]
            with self.assertLogs(level="WARNING"):
                self.assertEqual(fit({}, None, None, "auto", "0")[1], "CPU")
            self.assertEqual([call.kwargs["task_type"] for call in classifier.call_args_list], ["GPU", "CPU"])
            classifier.return_value.fit.side_effect = CatBoostError("Invalid column")
            with self.assertRaisesRegex(CatBoostError, "Invalid column"):
                fit({}, None, None, "auto", "0")
            classifier.return_value.fit.side_effect = CatBoostError("CUDA driver error")
            with self.assertRaisesRegex(CatBoostError, "CUDA driver error"):
                fit({}, None, None, "gpu", "0")

    def test_split_and_tuning(self):
        y = np.tile(np.arange(3), 40)
        train, val, hold = partition(y, .2, 42)
        self.assertEqual(len(hold), 18)
        self.assertEqual(len(set(np.r_[train, val, hold])), len(y))
        p = np.full((len(val), 3), .1)
        p[:, 0] += .4
        p[np.arange(len(val)), y[val]] += .3
        weights = tune(y[val], p)
        self.assertGreaterEqual(
            macro_f1(y[val], (p * weights).argmax(axis=1), 3),
            macro_f1(y[val], p.argmax(axis=1), 3),
        )

    def test_alignment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prob.csv"
            path.write_text("B,id,A\n0.8,002,0.2\n0.1,001,0.9\n", encoding="utf-8")
            np.testing.assert_allclose(
                aligned(path, "id", ["001", "002"], ["A", "B"]),
                [[.9, .1], [.2, .8]],
            )
            with self.assertRaises(ValueError):
                aligned(path, "id", ["001", "003"], ["A", "B"])

    def test_train_and_submit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({
                "id": [f"{i:04}" for i in range(120)],
                "status": ["ABC"[i % 3] for i in range(120)],
                "hub": ["0" + str(i % 2) for i in range(120)],
                "value": [i % 3 for i in range(120)],
                "booked_at": ["2026-01-02 00:00:00"] * 120,
            }).write_csv(root / "train.csv")
            pl.DataFrame({
                "id": ["0200", "0201"], "hub": ["01", "02"], "value": [0, 1],
                "booked_at": ["2026-01-02 00:00:00"] * 2,
            }).write_csv(root / "test.csv")
            (root / "sample.csv").write_text("id,status\n0201,\n0200,\n", encoding="utf-8")
            pl.DataFrame({"hub": ["00", "01"], "region": ["E", "W"]}).write_parquet(root / "hubs.parquet")
            pl.DataFrame({
                "hub": ["01", "01"], "event_at": ["2026-01-01 00:00:00", "2026-01-03 00:00:00"],
                "distance": [1., 999.],
            }).write_parquet(root / "events.parquet")
            config = root / "test.toml"
            config.write_text(
                """
[data]
root = "."
id = "id"
target = "status"
train = "train.csv"
test = "test.csv"
sample = "sample.csv"
features = ["hub", "value", "booked_at"]
categorical = ["hub"]
dates = { booked_at = "%Y-%m-%d %H:%M:%S" }
[training]
seed = 42
validation_fraction_of_development = 0.2
device = "auto"
gpu_devices = "0"
early_stopping = 3
class_weights = "SqrtBalanced"
[profiles.baseline]
iterations = 8
depth = 3
learning_rate = 0.2
[[relations]]
name = "hub"
kind = "lookup"
path = "hubs.parquet"
left_key = "hub"
right_key = "hub"
columns = ["region"]
[[relations]]
name = "history"
kind = "aggregate"
path = "events.parquet"
left_key = "hub"
right_key = "hub"
columns = ["distance"]
event_time = "event_at"
cutoff = "booked_at"
dates = { event_at = "%Y-%m-%d %H:%M:%S" }
""",
                encoding="utf-8",
            )
            output = root / "run"
            result = subprocess.run(
                [sys.executable, str(PROJECT / "run.py"), "--config", str(config),
                 "train", "--device", "cpu", "--out", str(output)],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            features = pl.read_parquet(output / "test_features.parquet")
            self.assertEqual(features["hub__region"].to_list(), ["W", None])
            self.assertEqual(features["history__distance__mean"].to_list(), [1., None])
            self.assertEqual(features["history__count"].to_list(), [1, None])
            with (output / "submission.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual([row["id"] for row in csv.DictReader(handle)], ["0201", "0200"])
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["device"], "CPU")
            self.assertEqual(metrics["rows"]["holdout"], 18)
            values = aligned(output / "probabilities.csv", "id", ["0201", "0200"], ["A", "B", "C"])
            self.assertEqual(values.shape, (2, 3))
            teammate = root / "teammate.csv"
            teammate.write_text("C,id,B,A\n1,0200,0,0\n0,0201,1,0\n", encoding="utf-8")
            blend = root / "blend.csv"
            result = subprocess.run(
                [sys.executable, str(PROJECT / "run.py"), "--config", str(config), "ensemble",
                 "--run", str(output), "--teammate", str(teammate), "--weight", "1",
                 "--out", str(blend), "--rules-permit"],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with blend.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.reader(handle)), [["id", "status"], ["0201", "B"], ["0200", "C"]])


if __name__ == "__main__":
    unittest.main()


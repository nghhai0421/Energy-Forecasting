"""Check ARIMA integration on a small synthetic series."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import yaml

import main
from src.models.ARIMA import run as arima
from src.models.ridge_regression import run as ridge
from src.outputs.plot import create_tuning_plot


class ArimaWorkflowTests(unittest.TestCase):
    def setUp(self):
        """Create temporary data and experiment/HPO YAML files."""
        self.directory = tempfile.TemporaryDirectory(prefix="energy-arima-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Choices for a categorical distribution.*")

        # Create Data
        observations = np.arange(40)
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC"),
            "target": 30 + observations * 0.1 + np.sin(observations / 3),
        })
        data_path = self.root / "data.csv"
        frame.to_csv(data_path, index=False)
        self.experiment = {
            "model": "ARIMA",
            "data": {"path": str(data_path), "timestamp_column": "timestamp", "target_column": "target"},
            "forecast": {"horizon": 2, "step_length": 8},
            "split": {"test_fraction": 0.2, "train_window_fraction": 0.75},
            "output_root": str(self.root / "outputs"),
            "run_name": "arima-test",
        }
        self.exp_path = self.root / "experiment.yaml"
        self.exp_path.write_text(yaml.safe_dump(self.experiment), encoding="utf-8")
        self.mdl_path = self.root / "model.yaml"
        self.mdl_path.write_text(yaml.safe_dump({
            "hpo": {"n_trials": 1, "seed": 7, "orders": [[0, 1, 0]]},
        }), encoding="utf-8")

    def test_default_evaluation(self):
        """Defaults evaluate without HPO and save checkpoints and plots."""
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            arima, "ForecastingOptunaSearchCV", side_effect=AssertionError("Unexpected HPO"),
        ):
            run_path = arima.run_experiment(self.exp_path)

        # Check Outputs
        model = joblib.load(run_path / "checkpoints/model.joblib")
        self.assertEqual(model.order, (1, 1, 1))
        self.assertEqual(int(model.cutoff[0]), 31)
        selection = json.loads((run_path / "artifacts/selected_parameters.json").read_text())
        self.assertEqual(selection["source"], "defaults")
        self.assertFalse((run_path / "artifacts/hpo_trials.csv").exists())
        manifest = json.loads((run_path / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertIsNone(manifest["config_files"]["model"])
        self.assertEqual(len(list((run_path / "plots").glob("*.png"))), 4)
        for split in ("cv_validation", "final_test"):
            rows = pd.read_csv(run_path / "predictions" / f"{split}.csv")
            columns = {"forecast_origin", "target_timestamp", "horizon_step", "actual", "prediction"}
            self.assertTrue(columns.issubset(rows))
            self.assertTrue(np.isfinite(rows["prediction"]).all())
            metrics = json.loads((run_path / "metrics" / f"{split}.json").read_text())
            self.assertEqual(metrics["overall"]["n_predictions"], len(rows))
            mae = float((rows["actual"] - rows["prediction"]).abs().mean())
            self.assertAlmostEqual(metrics["overall"]["mae"], mae)
        test_rows = pd.read_csv(run_path / "predictions/final_test.csv")
        first_target = pd.to_datetime(test_rows["target_timestamp"], utc=True).min()
        self.assertEqual(first_target, pd.Timestamp("2024-01-01 08:00", tz="UTC"))

    def test_hpo_and_saved_parameters(self):
        """A one-trial search tunes development data; its JSON order can be reused."""
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(arima, "create_tuning_plot"),
            patch.object(arima, "create_forecast_plots"),
        ):
            paths = arima.run_experiment(self.exp_path, self.mdl_path, stage="all")
            selection = json.loads((paths["hpo"] / "artifacts/selected_parameters.json").read_text())
            self.assertEqual(selection["parameters"]["order"], [0, 1, 0])
            self.assertEqual(selection["source"], "hpo")
            trials = pd.read_csv(paths["hpo"] / "artifacts/hpo_trials.csv")
            self.assertEqual(len(trials), 1)
            self.assertTrue({"trial", "score"}.issubset(trials))
            with patch.object(arima, "ForecastingOptunaSearchCV", side_effect=AssertionError("Unexpected HPO")):
                reused_path = arima.run_experiment(self.exp_path, hpo_run=paths["hpo"])

        # Check Reuse
        self.assertNotEqual(reused_path, paths["evaluation"])
        model = joblib.load(reused_path / "checkpoints/model.joblib")
        self.assertEqual(model.order, (0, 1, 0))
        manifest = json.loads((reused_path / "run_manifest.json").read_text())
        self.assertEqual(manifest["hpo_source"], str(paths["hpo"]))
        create_tuning_plot(paths["hpo"])
        self.assertGreater((paths["hpo"] / "plots/optuna_validation_history.png").stat().st_size, 1000)

    def test_overrides_and_config_requirement(self):
        """YAML overrides work without HPO; tuning requires its own settings."""
        overrides = {"order": [0, 1, 0], "enforce_stationarity": True}
        self.mdl_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), patch.object(arima, "create_forecast_plots"):
            path = arima.run_experiment(self.exp_path, self.mdl_path)
        model = joblib.load(path / "checkpoints/model.joblib")
        self.assertEqual(model.order, (0, 1, 0))
        self.assertTrue(model.enforce_stationarity)
        for stage in ("hpo", "all"):
            with self.assertRaisesRegex(ValueError, "hpo section"):
                arima.run_experiment(self.exp_path, stage=stage)

    def test_cli_dispatch(self):
        """The CLI selects either runner and retains optional model configuration."""
        for name, module in (("ARIMA", arima), ("ridge_regression", ridge)):
            with self.subTest(model=name):
                self.experiment["model"] = name
                self.exp_path.write_text(yaml.safe_dump(self.experiment), encoding="utf-8")
                with (
                    patch.object(sys, "argv", ["main.py", "--exp-cfg", str(self.exp_path)]),
                    patch.object(module, "run_experiment") as start,
                ):
                    main.main()
                self.assertIsNone(start.call_args.args[1])
                self.assertEqual(start.call_args.kwargs["stage"], "evaluate")


if __name__ == "__main__":
    unittest.main()

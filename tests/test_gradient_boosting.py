"""Leakage, replay parity, and source isolation checks for final GBT experiments."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.gradient_boosting.run import fit_model, run_experiment
from src.models.gradient_boosting.inference import forecast
from src.models.gradient_boosting.process import FEATURE_VERSION, build_features, load_holidays, load_sources, required_history


class FinalGBTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dates = pd.date_range("2024-10-15", periods=20 * 96, freq="15min", tz="UTC")
        self.data = dict(frequency="15min", timestamp_column="timestamp_utc", target_column="load",
                         lag_hours=[24, 168], observation_delay_minutes=60,
                         lag_fields=dict(load="load", temperature="temperature", price="price"), sources={})
        for name, column in (("consumption", "load"), ("weather", "temperature"), ("prices", "price")):
            path = self.root / f"{name}.csv"
            # Consumption crosses autumn DST; weather remains fixed at +02:00.
            zone = "+02:00" if name == "weather" else "Europe/Berlin"
            fmt = "%Y-%m-%d %H:%M:%S" if name == "weather" else "%b %d, %Y %I:%M %p"
            pd.DataFrame({"time": self.dates.tz_convert(zone).strftime(fmt),
                          column: 100 + np.sin(np.arange(len(self.dates)) / 20)}).to_csv(path, index=False)
            self.data["sources"][name] = dict(development=str(path), demo=str(self.root / "ABSENT.csv"),
                                             timestamp_column="time", timezone=zone, columns=[column])
        holiday = self.root / "holidays.csv"
        pd.DataFrame({"date": ["2024-01-01"]}).to_csv(holiday, index=False)
        self.data["sources"]["holidays"] = dict(path=str(holiday), date_column="date", timezone="Europe/Berlin")
        self.history, self.actual, self.state, _ = load_sources(self.data)
        self.origin = self.dates[16 * 96]
        warmup = required_history(self.data)
        self.X = build_features(self.history, self.dates[warmup:16 * 96], self.data, ["2024-01-01"])
        model, count, _ = fit_model(self.X, self.actual.reindex(self.X.index), dict(max_iter=3, max_leaf_nodes=4, random_state=42))
        self.bundle = dict(forecaster=model, feature_version=FEATURE_VERSION, feature_columns=list(self.X),
                           dataset=self.data, holiday_dates=["2024-01-01"], horizon=96, training_rows=count,
                           training_cutoff=self.dates[16 * 96 - 1].isoformat())

    def test_source_alignment_across_dst(self):
        pd.testing.assert_index_equal(self.history.index, self.dates.rename("timestamp_utc"), check_names=True)
        np.testing.assert_allclose(self.history.load, self.history.temperature)

    def test_future_calendar_is_available_without_demo_observations(self):
        dates = load_holidays(self.data, years=[2024, 2025])
        self.assertIn("2025-01-01", dates)
        self.assertIn("2025-01-06", dates)
        history = self.history.copy()
        history.index = pd.date_range("2024-12-12", periods=len(history), freq="15min", tz="UTC")
        features = build_features(history, pd.date_range("2025-01-01", periods=96, freq="15min", tz="UTC"), self.data, dates)
        self.assertEqual(features.is_holiday.iloc[0], 1)

    def test_future_mutation_and_replay_do_not_change_weights(self):
        before = joblib.hash(self.bundle["forecaster"].estimator_)
        expected = forecast(self.bundle, self.history, self.origin)
        changed = self.history.copy()
        changed.loc[changed.index >= self.origin] = -999999
        with patch.object(self.bundle["forecaster"], "fit", side_effect=AssertionError("Unexpected fit")), \
             patch.object(self.bundle["forecaster"], "update", side_effect=AssertionError("Unexpected update")):
            actual = forecast(self.bundle, changed, self.origin)
        pd.testing.assert_series_equal(expected["prediction"], actual["prediction"])
        self.assertEqual(before, joblib.hash(self.bundle["forecaster"].estimator_))
        expected_inputs = build_features(self.history, expected["prediction"].index, self.data, ["2024-01-01"])
        pd.testing.assert_frame_equal(expected["features"], expected_inputs)
        self.assertEqual(expected_inputs.iloc[-1]["load_lag_24h_delay_60m"], self.history.loc[self.origin - pd.Timedelta(minutes=75), "load"])

    def test_batch_validation_matches_origin_replay(self):
        dates = self.dates[16 * 96:18 * 96]
        X = build_features(self.history, dates, self.data, ["2024-01-01"]).reset_index(drop=True)
        count = self.bundle["training_rows"]
        X.index = pd.RangeIndex(count, count + len(X))
        batch = self.bundle["forecaster"].predict(fh=np.arange(1, len(X) + 1), X=X).to_numpy()
        replay = np.concatenate([forecast(self.bundle, self.history, dates[i])["prediction"].to_numpy() for i in (0, 96)])
        np.testing.assert_allclose(batch, replay)

    def test_missing_labels_remain_unobserved_and_are_excluded_from_fit(self):
        source = self.data["sources"]["consumption"]["development"]
        raw = pd.read_csv(source)
        raw.loc[800, "load"] = np.nan
        raw.to_csv(source, index=False)
        prepared, actual, _, _ = load_sources(self.data)
        self.assertTrue(pd.isna(actual.iloc[800]))
        self.assertAlmostEqual(prepared.load.iloc[800], actual.iloc[:800].mean())
        _, count, _ = fit_model(self.X, actual.reindex(self.X.index), dict(max_iter=2))
        self.assertEqual(count, len(self.X) - 1)

    def test_hpo_completes_when_demo_files_do_not_exist(self):
        experiment = dict(model="gradient_boosting", run_name="isolation_test", data=copy.deepcopy(self.data),
                          forecast=dict(lookback=676, horizon=96, step_length=96), output_root=str(self.root / "outputs"),
                          split=dict(train_start=self.dates[0].isoformat(),
                                     test_start=(self.dates[-1] + pd.Timedelta(minutes=15)).isoformat(),
                                     test_end=(self.dates[-1] + pd.Timedelta(days=3)).isoformat(),
                                     validation_starts=[self.origin.isoformat()]))
        settings = dict(max_iter=2, hpo=dict(n_trials=1, seed=42,
                        search_space=dict(max_leaf_nodes=dict(type="int", low=3, high=4))))
        experiment["parameters"] = {key: value for key, value in settings.items() if key != "hpo"}
        experiment["hpo"] = settings["hpo"]
        config = self.root / "experiment.yaml"
        config.write_text(yaml.safe_dump(experiment), encoding="utf-8")
        path = run_experiment(config, stage="hpo")
        self.assertTrue((path / "artifacts/selected_parameters.json").is_file())
        self.assertFalse(list(path.rglob("*.png")))

    def test_rejects_incomplete_context_and_training_origins(self):
        with self.assertRaises(ValueError):
            forecast(self.bundle, self.history.drop(self.origin - pd.Timedelta(minutes=15)), self.origin)
        with self.assertRaises(ValueError):
            forecast(self.bundle, self.history, self.origin - pd.Timedelta(days=1))

    def test_hourly_replay_stays_inside_test_boundary_and_one_run(self):
        data = copy.deepcopy(self.data)
        for name in ("consumption", "weather", "prices"):
            source = data["sources"][name]
            raw = pd.read_csv(source["development"])
            raw.iloc[:16 * 96].to_csv(source["development"], index=False)
            source["demo"] = str(self.root / f"{name}_demo.csv")
            raw.iloc[16 * 96:].to_csv(source["demo"], index=False)
        experiment = dict(model="gradient_boosting", run_name="hourly_test", data=data,
            forecast=dict(lookback=676, horizon=96, step_length=4), output_root=str(self.root / "outputs"),
            parameters=dict(max_iter=2, max_leaf_nodes=4),
            split=dict(train_start=self.dates[0].isoformat(), test_start=self.origin.isoformat(),
                       test_end=(self.origin + pd.Timedelta(days=2)).isoformat(),
                       validation_starts=[self.dates[12 * 96].isoformat()]))
        config = self.root / "experiment.yaml"
        config.write_text(yaml.safe_dump(experiment), encoding="utf-8")
        with patch("builtins.print"):
            path = run_experiment(config)
        predictions = pd.read_csv(path / "predictions/final_test.csv")
        self.assertEqual(len(predictions), 25 * 96)
        self.assertEqual(predictions.forecast_origin.nunique(), 25)
        self.assertEqual(predictions.target_timestamp.nunique(), 192)
        self.assertEqual(list(predictions), ["split", "fold", "forecast_origin", "cutoff_timestamp",
                         "target_timestamp", "horizon_step", "actual", "prediction", "residual"])
        self.assertLess(pd.to_datetime(predictions.target_timestamp, utc=True).max(), self.origin + pd.Timedelta(days=2))
        self.assertTrue((path / "checkpoints/model.joblib").is_file())
        self.assertTrue((path / "artifacts/development_features.csv").is_file())
        self.assertTrue((path / "plots/forecast_example_1.png").is_file())
        self.assertEqual(list((self.root / "outputs").iterdir()), [path.parent])
        manifest = json.loads((path / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")


if __name__ == "__main__":
    unittest.main()

"""Reject invalid comparisons and avoid counting overlapping energy twice."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.compare_runs import compare_runs


class MatchedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.left, self.right = [Path(self.temp.name) / model for model in ("gbt", "chronos")]
        dates = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
        for path, prediction in ((self.left, 8.0), (self.right, 13.0)):
            for folder in ("predictions", "metrics", "plots"):
                (path / folder).mkdir(parents=True)
            manifest = {"status": "completed", "run_id": path.name,
                        "data": {"target_column": "Residual load [MWh]", "frequency": "15min"},
                        "forecast": {"horizon": 4, "step_length": 2, "lookback": 8}}
            (path / "run_manifest.json").write_text(json.dumps(manifest))
            rows = pd.concat([pd.DataFrame({"split": "final_test", "forecast_origin": dates[i],
                "cutoff_timestamp": dates[i] - pd.Timedelta(minutes=15), "target_timestamp": dates[i:i+4],
                "horizon_step": np.arange(1, 5), "actual": 10.0, "prediction": prediction}) for i in (0, 2, 4)])
            rows.to_csv(path / "predictions/final_test.csv", index=False)

    def test_direction_and_unique_delivery_totals(self):
        result = compare_runs(self.left, self.right)
        self.assertEqual(result["all_forecast_pairs"]["gbt"]["underprediction_total"], 24)
        self.assertEqual(result["non_overlapping_delivery"]["gbt"]["underprediction_total"], 16)
        self.assertEqual(result["non_overlapping_delivery"]["chronos2"]["overprediction_total"], 24)
        rows = pd.read_csv(self.left / "predictions/chronos2_matched_delivery.csv")
        self.assertEqual(len(rows), 8)
        self.assertFalse(rows.target_timestamp.duplicated().any())

    def test_different_targets_rejected(self):
        path = self.right / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["data"]["target_column"] = "grid load [MWh]"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Different target"):
            compare_runs(self.left, self.right)

    def test_missing_origin_is_not_silently_intersected(self):
        path = self.right / "predictions/final_test.csv"
        rows = pd.read_csv(path)
        rows.iloc[4:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "no silent intersection"):
            compare_runs(self.left, self.right)

    def test_revised_labels_rejected(self):
        path = self.right / "predictions/final_test.csv"
        rows = pd.read_csv(path)
        rows.loc[0, "actual"] += 1
        rows.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Actual labels differ"):
            compare_runs(self.left, self.right)

    def test_invalid_horizon_timestamp_rejected(self):
        path = self.right / "predictions/final_test.csv"
        rows = pd.read_csv(path)
        rows.loc[0, "horizon_step"] = 2
        rows.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "horizon steps"):
            compare_runs(self.left, self.right)


if __name__ == "__main__":
    unittest.main()

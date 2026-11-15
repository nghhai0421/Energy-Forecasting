"""Validate identical interval scoring, missing benchmarks and revised labels."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.compare_operational import compare, load_smard


class OperationalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Cross the repeated autumn hour using distinct UTC delivery intervals.
        self.dates = pd.date_range("2025-10-25T22:00:00Z", periods=96, freq="15min")
        self.actual = self.root / "actual.csv"
        self.official = self.root / "official.csv"
        self.predictions = self.root / "predictions.csv"
        for path, value in ((self.actual, "1,000.00"), (self.official, "1,010.00")):
            pd.DataFrame({"Start date": self.dates.tz_convert("Europe/Berlin").strftime("%b %d, %Y %I:%M %p"),
                          "grid load [MWh] Original resolutions": value}).to_csv(path, sep=";", index=False)
        pd.DataFrame(dict(split="final_test", target_timestamp=self.dates,
                          forecast_origin=self.dates[0], horizon_step=np.arange(1, 97),
                          actual=1000., prediction=1002.)).to_csv(self.predictions, index=False)

    def test_dst_and_units_are_preserved(self):
        frame = load_smard(self.actual, "actual_mwh")
        self.assertEqual(list(frame.target_timestamp), list(self.dates))
        self.assertTrue(frame.actual_mwh.eq(1000).all())

    def test_missing_forecast_excluded_identically_for_all_models(self):
        raw = pd.read_csv(self.official, sep=";")
        raw.loc[0, "grid load [MWh] Original resolutions"] = "-"
        raw.to_csv(self.official, sep=";", index=False)
        output = self.root / "comparison"
        with patch("builtins.print"):
            compare(self.predictions, self.actual, self.official, output)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["scored_rows"], 95)
        self.assertEqual(summary["excluded_missing_rows"], 1)
        for metrics in summary["metrics"].values():
            self.assertEqual(metrics["intervals"], 95)
        self.assertEqual(summary["metrics"]["GBT"]["mae_mwh"], 2.)
        self.assertEqual(summary["metrics"]["Published TransnetBW forecast"]["rmse_mwh"], 10.)
        self.assertEqual(summary["gbt_mae_change_vs_published_percent"], -80.)
        self.assertEqual(set(summary["metrics"]), {"GBT", "Published TransnetBW forecast"})

    def test_unused_legacy_baseline_does_not_exclude_intervals(self):
        predictions = pd.read_csv(self.predictions)
        predictions["weekly_baseline"] = np.nan
        predictions.to_csv(self.predictions, index=False)
        output = self.root / "comparison"
        with patch("builtins.print"):
            compare(self.predictions, self.actual, self.official, output)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["scored_rows"], 96)
        self.assertNotIn("weekly_baseline", pd.read_csv(output / "aligned_predictions.csv").columns)

    def test_revised_actuals_rejected(self):
        raw = pd.read_csv(self.actual, sep=";")
        raw.loc[0, "grid load [MWh] Original resolutions"] = "999.00"
        raw.to_csv(self.actual, sep=";", index=False)
        with self.assertRaisesRegex(ValueError, "reconcile revisions"):
            compare(self.predictions, self.actual, self.official, self.root / "comparison")


if __name__ == "__main__":
    unittest.main()

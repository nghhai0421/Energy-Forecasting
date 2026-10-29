"""Check hourly input preservation and forecast-origin availability on synthetic regional inputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.gradient_boosting.process import load_data


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.csv"
        self.raw = pd.DataFrame({
            "timestamp_utc": pd.date_range("2024-10-25", periods=120, freq="h", tz="UTC"),
            "load": np.ones(120), "temperature": np.arange(120),
            "is_holiday": np.zeros(120),
        })
        self.config = dict(path=str(self.path), timestamp_column="timestamp_utc",
                           target_column="load", historical_weather=["temperature"],
                           weather_lag_hours=24)

    def load(self):
        self.raw.to_csv(self.path, index=False)
        return load_data(self.config, 24)

    def test_hourly_values_preserved_and_dst_continuity(self):
        y, X, series, _ = self.load()
        self.assertTrue(y.eq(1).all())
        self.assertEqual(len(y), 96)
        self.assertTrue(series.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all())
        self.assertEqual(X.iloc[0]["temperature_lag_24h"], 0)

    def test_future_observations_cannot_change_horizon_features(self):
        _, before, _, _ = self.load()
        # Origin is the end of raw hour 59; forecast hours 60..83 use weather 36..59.
        self.raw.loc[60:, ["temperature", "load"]] = 100000
        _, after, _, _ = self.load()
        pd.testing.assert_frame_equal(before.iloc[36:60], after.iloc[36:60])

    def test_gaps_and_short_weather_lags_rejected(self):
        self.config["weather_lag_hours"] = 23
        with self.assertRaisesRegex(ValueError, "Weather lag"):
            self.load()
        self.config["weather_lag_hours"] = 24
        self.raw = self.raw.drop(index=10)
        with self.assertRaisesRegex(ValueError, "hourly"):
            self.load()

    def test_quarter_hourly_and_duplicate_inputs_rejected(self):
        self.raw["timestamp_utc"] = pd.date_range("2024-01-01", periods=120, freq="15min", tz="UTC")
        with self.assertRaisesRegex(ValueError, "hourly"):
            self.load()
        self.raw.loc[1, "timestamp_utc"] = self.raw.loc[0, "timestamp_utc"]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load()

    def test_calendar_only_retains_identical_rows(self):
        y, _, series, _ = self.load()
        self.config["historical_weather"] = []
        other_y, X, other_series, _ = self.load()
        pd.testing.assert_series_equal(y, other_y)
        pd.testing.assert_frame_equal(series, other_series)
        self.assertEqual(list(X), ["hour", "weekday", "month", "is_holiday"])


if __name__ == "__main__":
    unittest.main()

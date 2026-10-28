"""Check aggregation and forecast-origin availability on synthetic regional inputs."""

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
            "timestamp_utc": pd.date_range("2024-10-25", periods=4 * 120, freq="15min", tz="UTC"),
            "load": np.ones(4 * 120), "temperature": np.arange(4 * 120),
            "is_holiday": np.zeros(4 * 120),
        })
        self.config = dict(path=str(self.path), timestamp_column="timestamp_utc",
                           target_column="load", historical_weather=["temperature"],
                           weather_lag_hours=24)

    def load(self):
        self.raw.to_csv(self.path, index=False)
        return load_data(self.config, 24)

    def test_energy_sum_and_dst_continuity(self):
        y, X, series, _ = self.load()
        self.assertTrue(y.eq(4).all())
        self.assertEqual(len(y), 96)
        self.assertTrue(series.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all())
        self.assertEqual(X.iloc[0]["temperature_lag_24h"], 1.5)

    def test_future_observations_cannot_change_horizon_features(self):
        _, before, _, _ = self.load()
        # Origin is the end of raw hour 59; forecast hours 60..83 use weather 36..59.
        self.raw.loc[240:, ["temperature", "load"]] = 100000
        _, after, _, _ = self.load()
        pd.testing.assert_frame_equal(before.iloc[36:60], after.iloc[36:60])

    def test_gaps_and_short_weather_lags_rejected(self):
        self.config["weather_lag_hours"] = 23
        with self.assertRaisesRegex(ValueError, "Weather lag"):
            self.load()
        self.config["weather_lag_hours"] = 24
        self.raw = self.raw.drop(index=10)
        with self.assertRaisesRegex(ValueError, "quarter-hourly"):
            self.load()


if __name__ == "__main__":
    unittest.main()

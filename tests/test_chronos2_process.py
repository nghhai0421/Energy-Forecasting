"""Check quarter-hour source alignment and causal value imputation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.add_seasonal_features import add_seasonal_features
from src.models.chronos2.process import load_data, prepare_future_covariates


class ChronosPreparationTests(unittest.TestCase):
    """Exercise preparation with small source CSVs."""

    def setUp(self):
        """Create timestamp-aligned consumption, weather, and holiday fixtures.

        Returns
        -------
        None
            Fixtures are stored on the test instance.
        """
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.dates = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
        self.consumption = pd.DataFrame({
            "Start date": self.dates.tz_convert("Europe/Berlin").strftime("%b %d, %Y %I:%M %p"),
            "Residual load [MWh]": [-10., 20., 30., 40., 50., np.nan, 70., 80., 90., 100., 110., 120.],
            "grid load [MWh]": [10., 20., np.nan, 40., 50., 60., 70., 80., 90., 100., 110., 120.],
            "Grid load incl. hydro pumped storage [MWh]": np.arange(12, dtype=float) + 100.,
            "Hydro pumped storage [MWh]": np.arange(12, dtype=float),
        })
        self.weather = pd.DataFrame({
            "time": self.dates.tz_convert("Etc/GMT-2").strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": [-5., -4., -3., np.nan, -1., 0., 1., 2., 3., 4., 5., 6.],
            "wind_speed_10m (km/h)": [1., 2., 3., 4., -1., 6., 7., 8., 9., 10., 11., 12.],
            "dew_point": np.full(12, -8.),
        })
        self.config = {
            "sources": {
                "consumption": {"development": str(root / "consumption.csv"), "timestamp_column": "Start date", "timezone": "Europe/Berlin", "columns": ["grid load [MWh]", "Grid load incl. hydro pumped storage [MWh]", "Hydro pumped storage [MWh]"]},
                "weather": {"development": str(root / "weather.csv"), "timestamp_column": "time", "timezone": "+02:00", "columns": ["temperature", "wind_speed_10m (km/h)", "dew_point"]},
                "holidays": {"path": str(root / "holidays.csv"), "date_column": "Datum", "timezone": "Europe/Berlin"},
            },
            "timestamp_column": "timestamp_utc", "target_column": "Residual load [MWh]",
            "frequency": "15min",
            "covariates": {"historical": ["grid load [MWh]", "Grid load incl. hydro pumped storage [MWh]", "Hydro pumped storage [MWh]", "temperature", "wind_speed_10m (km/h)", "dew_point"], "future": ["is_holiday"]},
        }
        calendar = add_seasonal_features(pd.DataFrame({"timestamp_utc": self.dates}), "timestamp_utc")
        self.config["covariates"]["future"].extend(calendar.columns.drop("timestamp_utc"))
        pd.DataFrame({"Datum": ["2024-01-01"]}).to_csv(root / "holidays.csv", index=False)

    def load(self):
        """Write fixtures and prepare their model inputs.

        Returns
        -------
        tuple
            Target, covariates, observed series, and preparation metadata.
        """
        self.consumption.to_csv(self.config["sources"]["consumption"]["development"], index=False)
        self.weather.to_csv(self.config["sources"]["weather"]["development"], index=False)
        return load_data(self.config)

    def test_causal_imputation_and_preserved_negatives(self):
        """Check prior means, valid negatives, and original target gaps.

        Returns
        -------
        None
            Assertions verify prepared values and provenance.
        """
        y, X, series, metadata = self.load()
        self.assertEqual(y.iloc[0], -10.)
        self.assertEqual(y.iloc[5], 26.)
        self.assertEqual(X["grid load [MWh]"].iloc[2], 15.)
        self.assertEqual(list(X), self.config["covariates"]["historical"] + self.config["covariates"]["future"])
        self.assertEqual(X["temperature"].iloc[3], -4.)
        self.assertEqual(X["wind_speed_10m (km/h)"].iloc[4], 2.5)
        self.assertTrue(X["dew_point"].eq(-8.).all())
        self.assertTrue(X["is_holiday"].eq(1).all())
        self.assertTrue(pd.isna(series["target"].iloc[5]))
        self.assertEqual(metadata["unobserved_targets"], 1)
        self.weather.loc[8:, "temperature"] = 999.
        self.consumption.loc[8:, "grid load [MWh]"] = 999.
        _, changed, _, _ = self.load()
        pd.testing.assert_frame_equal(X.iloc[:8], changed.iloc[:8])

    def test_weather_alignment_retains_target_rows(self):
        """Check that trailing weather gaps retain target rows and are reported.

        Returns
        -------
        None
            Assertions verify alignment and imputation counts.
        """
        self.weather = self.weather.iloc[:-1]
        y, X, _, metadata = self.load()
        self.assertEqual(len(y), 12)
        self.assertEqual(metadata["weather_alignment_missing_rows"], 1)
        self.assertFalse(X.isna().any().any())

    def test_leading_missing_and_time_gaps_raise(self):
        """Check unrecoverable leading gaps and broken time grids.

        Returns
        -------
        None
            Assertions verify clear input failures.
        """
        self.weather.loc[0, "temperature"] = np.nan
        with self.assertRaisesRegex(ValueError, "prior history"):
            self.load()
        self.weather.loc[0, "temperature"] = -5.
        self.consumption = self.consumption.drop(index=3)
        with self.assertRaisesRegex(ValueError, "complete 15-minute"):
            self.load()

    def test_holiday_coverage_and_future_weather_raise(self):
        """Check missing holiday years and observed weather availability rules.

        Returns
        -------
        None
            Assertions verify source coverage and feature classification.
        """
        self.config["covariates"]["historical"].remove("temperature")
        self.config["covariates"]["future"].append("temperature")
        with self.assertRaisesRegex(ValueError, "Observed weather"):
            self.load()
        self.config["covariates"]["future"].remove("temperature")
        self.config["covariates"]["historical"].append("temperature")
        pd.DataFrame({"Datum": ["2023-01-01"]}).to_csv(self.config["sources"]["holidays"]["path"], index=False)
        with self.assertRaisesRegex(ValueError, "holiday CSV"):
            self.load()

    def test_future_consumption_covariates_raise(self):
        """Reject consumption observations in the known-future feature group.

        Returns
        -------
        None
            Assertions verify forecast-origin availability.
        """
        column = "grid load [MWh]"
        self.config["covariates"]["historical"].remove(column)
        self.config["covariates"]["future"].append(column)
        with self.assertRaisesRegex(ValueError, "consumption covariates must be historical"):
            self.load()

    def test_calendar_features_follow_berlin_dst(self):
        """Verify future calendar features across the Berlin spring transition.

        Returns
        -------
        None
            Assertions verify time alignment and known-future feature selection.
        """
        self.dates = pd.date_range("2024-03-31", periods=12, freq="15min", tz="UTC")
        self.consumption["Start date"] = self.dates.tz_convert("Europe/Berlin").strftime("%b %d, %Y %I:%M %p")
        self.weather["time"] = self.dates.tz_convert("Etc/GMT-2").strftime("%Y-%m-%d %H:%M:%S")
        _, X, series, metadata = self.load()
        self.assertEqual(len(self.config["covariates"]["future"]), 18)
        self.assertEqual(X.shape, (12, 24))
        self.assertTrue(X["calendar_year"].eq(2024).all())
        self.assertAlmostEqual(X["calendar_hour_of_day_sin"].iloc[0], np.sin(2 * np.pi / 23))
        self.assertAlmostEqual(X["calendar_hour_of_day_sin"].iloc[4], np.sin(2 * np.pi * 3 / 23))
        self.assertEqual(metadata["calendar_timezone"], "Europe/Berlin")
        single = add_seasonal_features(series[["timestamp"]].iloc[[4]], "timestamp")
        calendar_columns = single.columns.drop("timestamp")
        pd.testing.assert_frame_equal(X.loc[[4], calendar_columns], single[calendar_columns])
        self.config["covariates"]["future"].remove("calendar_year")
        with self.assertRaisesRegex(ValueError, "Calendar features"):
            self.load()

    def test_future_inputs_need_only_timestamps_and_holidays(self):
        """Prepare horizon features before consumption/weather fixture files exist.

        Returns
        -------
        None
            Assertions verify independent future inputs and historical consistency.
        """
        future = prepare_future_covariates(self.dates, self.config)
        self.assertEqual(future.shape, (12, 18))
        pd.testing.assert_index_equal(future.index, self.dates)
        _, X, _, _ = self.load()
        expected = X[self.config["covariates"]["future"]].copy()
        expected.index = self.dates
        pd.testing.assert_frame_equal(future, expected)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            prepare_future_covariates(self.dates.tz_localize(None), self.config)

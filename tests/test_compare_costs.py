"""Settlement sign, energy units, asymmetric prices, and complete joins."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.compare_costs import cashflows, compare_costs


class PriceComparisonTests(unittest.TestCase):
    def test_four_payment_directions_and_no_power_conversion(self):
        result = cashflows([10, 10, 10, 10], [8, 12, 8, 12], 50,
                           [100, 100, -100, -100], [100, 100, -100, -100])
        np.testing.assert_array_equal(result["settlement_eur"], [200, -200, -200, 200])
        np.testing.assert_array_equal(result["total_cost_eur"], [600, 400, 200, 800])
        np.testing.assert_array_equal(result["settlement_charges_eur"], [200, 0, 0, 200])
        np.testing.assert_array_equal(result["settlement_credits_eur"], [0, 200, 200, 0])

    def test_directional_prices_and_negative_day_ahead(self):
        result = cashflows([10, 10], [8, 12], -50, 100, -30)
        np.testing.assert_array_equal(result["settlement_eur"], [200, 60])
        np.testing.assert_array_equal(result["total_cost_eur"], [-200, -540])

    def test_equal_prices_make_forecast_errors_cost_neutral(self):
        for price in (-30, 80):
            result = cashflows([10, 10, 10], [8, 10, 12], price, price, price)
            np.testing.assert_array_equal(result["total_cost_eur"], np.full(3, 10 * price))
            np.testing.assert_array_equal(result["cost_vs_actual_volume_at_day_ahead_eur"], [0, 0, 0])

    def test_nonfinite_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            cashflows(10, 8, 50, np.nan, 100)

    def test_missing_prices_and_duplicate_deliveries_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "predictions").mkdir()
            rows = pd.DataFrame({"target_timestamp": ["2025-01-01T00:00Z", "2025-01-01T01:00Z"],
                                 "horizon_step": 96, "actual": 10,
                                 "gbt_prediction": 8, "chronos2_prediction": 9, "transnetbw_prediction": 12})
            path = run / "predictions/evaluation_overview_step96.csv"
            rows.to_csv(path, index=False)
            dates = pd.to_datetime(rows.target_timestamp, utc=True)
            prices = pd.DataFrame({"target_timestamp": dates.iloc[:1], "day_ahead_eur_per_mwh": 50})
            settlement = pd.DataFrame({"target_timestamp": dates, "rebap_short_eur_per_mwh": 100, "rebap_long_eur_per_mwh": 100})
            with patch("src.evaluation.compare_costs.load_day_ahead", return_value=prices), \
                 patch("src.evaluation.compare_costs.load_settlement", return_value=settlement):
                with self.assertRaisesRegex(ValueError, "no rows are silently excluded"):
                    compare_costs(run, "unused", "unused")
            pd.concat([rows, rows.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one forecast"):
                compare_costs(run, "unused", "unused")


if __name__ == "__main__":
    unittest.main()

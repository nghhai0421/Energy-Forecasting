"""Verify reusable inference and evaluation with a lightweight forecaster."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sktime.split import SlidingWindowSplitter

from src.models.chronos2.inference import forecast, load_forecaster
from src.models.chronos2.run import evaluate_folds, format_results


class FakeForecaster:
    """Record model contexts and return deterministic forecasts."""

    def __init__(self, **kwargs):
        """Record constructor settings and expose fake model revision metadata.

        Parameters
        ----------
        **kwargs : dict
            Model constructor settings.

        Returns
        -------
        None
            Settings and fit records are stored on the instance.
        """
        self.settings = kwargs
        self.histories = []
        self.model_pipeline = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision")))

    def set_config(self, **config):
        """Record explicit sktime configuration overrides.

        Parameters
        ----------
        **config : dict
            Configuration values to retain.

        Returns
        -------
        FakeForecaster
            Configured test instance.
        """
        self.config = config
        return self

    def fit(self, y, X=None, fh=None):
        """Replace the current target and covariate context.

        Parameters
        ----------
        y : pandas.Series
            Target history.
        X : pandas.DataFrame or None, default None
            Aligned historical covariates.
        fh : array-like or None, default None
            Requested horizon steps.

        Returns
        -------
        FakeForecaster
            Updated test instance.
        """
        self.y = y.copy()
        self.X = X.copy() if X is not None else None
        self.histories.append(y.copy())
        return self

    def predict(self, fh, X):
        """Return the latest target plus each horizon step.

        Parameters
        ----------
        fh : array-like
            Relative horizon steps.
        X : pandas.DataFrame
            Known-future covariates.

        Returns
        -------
        pandas.Series
            Predictions indexed after the current positional cutoff.
        """
        self.future = X.copy()
        return pd.Series(self.y.iloc[-1] + np.asarray(fh), index=self.y.index[-1] + np.asarray(fh))


class InferenceTests(unittest.TestCase):
    """Check repeated forecasts and fold scoring without model dependencies."""

    def test_prediction_interval_uses_one_native_call(self):
        """Verify quantile inputs, aligned bounds, and invalid-interval rejection.

        Returns
        -------
        None
            Assertions verify native median/interval extraction without weights.
        """
        model = MagicMock()
        model.seed = 42
        model.config = {"context_length": 3, "limit_prediction_length": True}
        y = pd.Series([1., 2., 3.], name="load")
        X = pd.DataFrame({"weather": [5., 6., 7.], "is_holiday": [0, 0, 0]})
        dates = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
        future = pd.DataFrame({"is_holiday": [1, 1]}, index=dates)
        values = np.array([[[2., 4., 6.], [1., 5., 9.]]])
        model.model_pipeline.predict_quantiles.return_value = ([SimpleNamespace(numpy=lambda: values)], [])
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(set_seed=MagicMock())}):
            result = forecast(model, y, X, future, prediction_interval=True)
            np.testing.assert_allclose(result["prediction"], [4., 5.])
            np.testing.assert_allclose(result["prediction_lower"], [2., 1.])
            np.testing.assert_allclose(result["prediction_upper"], [6., 9.])
            for name in ("prediction", "prediction_lower", "prediction_upper"):
                pd.testing.assert_index_equal(result[name].index, dates)
            model.model_pipeline.predict_quantiles.assert_called_once()
            model.predict.assert_not_called()
            call = model.model_pipeline.predict_quantiles.call_args
            self.assertEqual(call.kwargs["quantile_levels"], [0.1, 0.5, 0.9])
            self.assertEqual(call.kwargs["context_length"], 3)
            inputs = call.args[0][0]
            np.testing.assert_allclose(inputs["target"], [[1., 2., 3.]])
            self.assertEqual(list(inputs["future_covariates"]), ["is_holiday"])
            self.assertIn("weather", inputs["past_covariates"])
            values[0, 0, 0] = 10.
            with self.assertRaisesRegex(ValueError, "ordered"):
                forecast(model, y, X, future, prediction_interval=True)

    def test_forecast_replaces_context_and_preserves_future_labels(self):
        """Verify model reuse, context isolation, and timestamp-labelled outputs.

        Returns
        -------
        None
            Assertions verify both successive forecast calls.
        """
        model = FakeForecaster()
        y = pd.Series([1., 2., 3.], index=[20, 21, 22], name="load")
        X = pd.DataFrame({"weather": [1., 2., 3.], "is_holiday": [0, 0, 0]}, index=y.index)
        dates = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
        future = pd.DataFrame({"is_holiday": [1, 1]}, index=dates)
        result = forecast(model, y, X, future)
        pd.testing.assert_index_equal(result["prediction"].index, dates)
        np.testing.assert_allclose(result["prediction"], [4., 5.])
        self.assertEqual(list(model.future), ["is_holiday"])
        self.assertEqual(list(model.future.index), [3, 4])
        changed = y * 10
        result = forecast(model, changed, X, future)
        np.testing.assert_allclose(result["prediction"], [31., 32.])
        np.testing.assert_allclose(model.histories[-1], changed)
        self.assertEqual(len(model.histories), 2)
        self.assertFalse(model.config["remember_data"])
        with self.assertRaisesRegex(ValueError, "Future columns"):
            forecast(model, y, X, future.assign(weather=999.))
        with self.assertRaisesRegex(ValueError, "finite"):
            forecast(model, y, X, future.assign(is_holiday=np.nan))

    def test_load_settings_and_revision(self):
        """Verify default inference settings without importing real Torch weights.

        Returns
        -------
        None
            Assertions verify constructor settings, revision, and initial context.
        """
        y = pd.Series([1., 2., 3.])
        X = pd.DataFrame({"is_holiday": [0, 0, 0]})
        with patch.dict("sys.modules", {"torch": SimpleNamespace(float32="float32")}):
            with patch("sktime.forecasting.chronos2.Chronos2Forecaster", FakeForecaster):
                loaded = load_forecaster(y, X, context_length=3)
        self.assertEqual(loaded["revision"], "test-revision")
        self.assertEqual(loaded["parameters"]["context_length"], 3)
        self.assertEqual(loaded["parameters"]["model_path"], "amazon/chronos-2")
        self.assertEqual(len(loaded["forecaster"].histories), 1)
        self.assertFalse(loaded["forecaster"].config["remember_data"])
        self.assertFalse(loaded["parameters"]["remember_data"])

    def test_evaluation_uses_shared_forecast_and_excludes_missing_actuals(self):
        """Verify positional alignment, causal folds, and original-target scoring.

        Returns
        -------
        None
            Assertions verify evaluated predictions and standardized result rows.
        """
        y = pd.Series(np.arange(12, dtype=float), name="load")
        X = pd.DataFrame({"weather": np.arange(12), "is_holiday": np.zeros(12)})
        series = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC"), "target": y})
        series.loc[5, "target"] = np.nan
        cv = SlidingWindowSplitter(fh=[1, 2, 3, 4], window_length=4, step_length=4)
        model = FakeForecaster()
        with patch("src.models.chronos2.run.prepare_future_covariates") as prepare:
            prepare.side_effect = lambda timestamps, dataset: pd.DataFrame({"is_holiday": np.zeros(len(timestamps))}, index=timestamps.index)
            evaluations = evaluate_folds(model, cv, y, X, series, {})
        self.assertEqual(evaluations["test_MeanAbsoluteError"].tolist(), [0., 0.])
        self.assertEqual(model.histories[0].tolist(), [0., 1., 2., 3.])
        self.assertEqual(model.histories[1].tolist(), [4., 5., 6., 7.])
        results = format_results({"cv_validation": evaluations}, series, {"context_length": 4})
        self.assertEqual(len(results["predictions"]), 7)
        self.assertFalse(results["predictions"]["actual"].isna().any())

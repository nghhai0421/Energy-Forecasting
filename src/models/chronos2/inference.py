"""Load and forecast with Chronos-2 for evaluation and interactive demos."""

from time import perf_counter
import warnings

import numpy as np
import pandas as pd


def load_forecaster(y, X=None, context_length=672, model_path="amazon/chronos-2",
                    device_map="cpu", seed=42):
    """Load pretrained Chronos-2 weights and initialize the observed context.

    Parameters
    ----------
    y : pandas.Series
        Initial observed, finite target history in chronological order.
    X : pandas.DataFrame or None, default None
        Historical covariates, including past calendar/holiday values, aligned to y.
    context_length : int, default 672
        Maximum number of target observations used as context.
    model_path : str, default 'amazon/chronos-2'
        Pretrained checkpoint identifier or local model path.
    device_map : str, default 'cpu'
        Device used for inference.
    seed : int, default 42
        Inference random seed.

    Returns
    -------
    dict
        Forecaster, effective parameters, model revision, and load time in seconds.

    Raises
    ------
    ValueError
        If history is empty, nonfinite, misaligned, or context_length is invalid.
    """
    # Check History
    if context_length < 1 or y.empty or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Finite target history and a positive context_length are required.")
    if X is not None and (not X.index.equals(y.index) or not np.isfinite(X.to_numpy(dtype=float)).all()):
        raise ValueError("Historical covariates must be finite and aligned to the target.")

    # Load Model
    import torch
    from sktime.forecasting.chronos2 import Chronos2Forecaster

    parameters = {
        "model_path": model_path, "context_length": context_length,
        "device_map": device_map, "torch_dtype": "float32",
        "limit_prediction_length": True, "seed": seed,
        "remember_data": False,
    }
    started = perf_counter()
    # sktime 1.2 warns during construction/reset before instance config is restored.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"The default of config ``remember_data`` will change.*",
            category=FutureWarning, module=r"sktime\.forecasting\.base\._base",
        )
        forecaster = Chronos2Forecaster(
            model_path=model_path, seed=seed,
            config={
                "context_length": context_length, "device_map": device_map,
                "torch_dtype": torch.float32, "limit_prediction_length": True,
            },
        )
        forecaster.set_config(remember_data=False)
        forecaster.fit(y.reset_index(drop=True), X=X.reset_index(drop=True) if X is not None else None)
    revision = getattr(forecaster.model_pipeline.model.config, "_commit_hash", None)
    return {
        "forecaster": forecaster, "parameters": parameters, "revision": revision,
        "load_seconds": perf_counter() - started,
    }


def forecast(forecaster, y, X, future_X, prediction_interval=False):
    """Forecast one horizon after replacing the forecaster's observed context.

    Parameters
    ----------
    forecaster : sktime forecaster
        Forecaster returned by load_forecaster; calls use its state sequentially.
    y : pandas.Series
        Chronological, finite target history available at the forecast origin.
    X : pandas.DataFrame
        Finite historical covariates with the same index as y.
    future_X : pandas.DataFrame
        Known-future covariates from prepare_future_covariates. Its row count
        defines the horizon; its index labels the returned predictions.
    prediction_interval : bool, default False
        Return the native Chronos-2 median and 10th/90th percentile bounds.

    Returns
    -------
    dict
        Prediction Series indexed like future_X, fit_seconds, and predict_seconds.
        Interval mode also includes prediction_lower and prediction_upper Series.

    Raises
    ------
    ValueError
        If inputs are empty, nonfinite, misaligned, or future columns are invalid.

    Notes
    -----
    Only calendar and holiday values enter future_X. Observed consumption and
    weather belong in X. Context is reset per call; sktime caches pretrained weights.
    """
    # Check Inputs
    if y.empty or future_X.empty:
        raise ValueError("Target history and future covariates must be nonempty.")
    if not X.index.equals(y.index):
        raise ValueError("Historical covariates must align with the target index.")
    if not X.columns.is_unique or not future_X.columns.is_unique:
        raise ValueError("Covariate column names must be unique.")
    allowed = [name for name in X if name == "is_holiday" or name.startswith("calendar_")]
    if list(future_X.columns) != allowed:
        raise ValueError("Future columns must match the historical calendar and holiday columns in order.")
    if not future_X.index.is_unique:
        raise ValueError("Future row labels must be unique.")
    for values in (y, X, future_X):
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Forecast inputs must be finite and complete.")

    # Refresh Context
    history = y.reset_index(drop=True)
    history_X = X.reset_index(drop=True)
    horizon = np.arange(1, len(future_X) + 1)
    model_future = future_X.copy()
    model_future.index = pd.RangeIndex(len(history), len(history) + len(future_X))
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"The default of config ``remember_data`` will change.*",
            category=FutureWarning, module=r"sktime\.forecasting\.base\._base",
        )
        forecaster.set_config(remember_data=False)
        forecaster.fit(history, X=history_X, fh=horizon)
    fit_seconds = perf_counter() - started

    # Forecast
    started = perf_counter()
    if prediction_interval:
        from transformers import set_seed

        set_seed(forecaster.seed)
        inputs = {
            "target": history.to_numpy(dtype=np.float32).reshape(1, -1),
            "past_covariates": {column: history_X[column].to_numpy() for column in history_X},
            "future_covariates": {column: model_future[column].to_numpy() for column in model_future},
        }
        quantiles, _ = forecaster.model_pipeline.predict_quantiles(
            [inputs], prediction_length=len(horizon), quantile_levels=[0.1, 0.5, 0.9],
            context_length=forecaster.config["context_length"],
            limit_prediction_length=forecaster.config["limit_prediction_length"],
        )
        values = quantiles[0].numpy()[0]
        if values.shape != (len(horizon), 3) or not np.isfinite(values).all():
            raise ValueError("Model must return finite lower, median, and upper values for each step.")
        if (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any():
            raise ValueError("Prediction quantiles must be ordered lower <= median <= upper.")
        prediction = pd.Series(values[:, 1])
    else:
        prediction = forecaster.predict(fh=horizon, X=model_future)
    predict_seconds = perf_counter() - started
    if len(prediction) != len(future_X) or not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise ValueError("Model must return one finite prediction per future row.")
    prediction = pd.Series(prediction.to_numpy().reshape(-1), index=future_X.index, name=y.name)
    result = {"prediction": prediction, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}
    if prediction_interval:
        result["prediction_lower"] = pd.Series(values[:, 0], index=future_X.index, name="prediction_lower")
        result["prediction_upper"] = pd.Series(values[:, 2], index=future_X.index, name="prediction_upper")
    return result

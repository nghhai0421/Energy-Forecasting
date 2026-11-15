"""Load frozen GBT weights and forecast with newly observed context, without fitting."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.models.gradient_boosting.process import (
    FEATURE_VERSION, INTERVAL, boundary, build_features, required_history, validate_index,
)


def load_forecaster(checkpoint):
    """Load one portable model bundle; no CSV files or external model downloads required."""
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.joblib"
    bundle = joblib.load(path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Unsupported checkpoint feature schema.")
    if not bundle["forecaster"].is_fitted:
        raise ValueError("Checkpoint does not contain fitted weights.")
    return bundle


def forecast(bundle, history, forecast_start, horizon=None):
    """Predict the next horizon from prepared UTC observations and a frozen bundle.

    Observations at/after forecast_start are discarded, even if the caller supplies
    a full replay table. No fit/update method is called. Returned predictions retain
    delivery timestamps. Earlier demo observations can enter later origins' lags.
    """
    start = boundary(forecast_start)
    horizon = bundle["horizon"] if horizon is None else horizon
    dataset = bundle["dataset"]
    if horizon != bundle["horizon"] or horizon > min(dataset["lag_hours"]) * 4:
        raise ValueError("Use the trained horizon, no longer than the shortest lag.")
    if start <= boundary(bundle["training_cutoff"]):
        raise ValueError("Forecast origin must follow the training cutoff.")
    # Use only the bounded observed context; no future load/weather/price is read.
    history = history.loc[history.index < start].tail(required_history(dataset))
    validate_index(history.index)
    if len(history) < required_history(dataset) or history.index[-1] + INTERVAL != start:
        raise ValueError("Supply contiguous observed context ending immediately before origin.")
    dates = pd.date_range(start, periods=horizon, freq="15min")
    features = build_features(history, dates, dataset, bundle["holiday_dates"])
    if list(features) != bundle["feature_columns"]:
        raise ValueError("Inference feature columns differ from training.")
    # YfromX is a pooled tabular regressor: positional indices label each request.
    model_X = features.reset_index(drop=True)
    model_X.index = pd.RangeIndex(bundle["training_rows"], bundle["training_rows"] + horizon)
    started = perf_counter()
    values = bundle["forecaster"].predict(fh=np.arange(1, horizon + 1), X=model_X).to_numpy().reshape(-1)
    elapsed = perf_counter() - started
    if len(values) != horizon or not np.isfinite(values).all():
        raise ValueError("Model must return one finite prediction per delivery interval.")
    return {"prediction": pd.Series(values, index=dates, name=dataset["target_column"]),
            "predict_seconds": elapsed, "features": features}

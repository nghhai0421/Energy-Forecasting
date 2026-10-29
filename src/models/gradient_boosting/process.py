"""Prepare the regional CSV without modifying raw files or using future observations."""

import numpy as np
import pandas as pd


def load_data(dataset, horizon):
    """Return hourly target, origin-safe features, timestamps, and preparation metadata."""
    # Load the already aggregated hourly CSV.
    raw = pd.read_csv(dataset["path"])
    dates = pd.to_datetime(raw[dataset["timestamp_column"]], utc=True)
    # Validate UTC order and spacing, including local DST transitions.
    if dataset.get("frequency", "h") != "h":
        raise ValueError("This loader expects frequency: h and hourly input data.")
    if len(raw) <= dataset.get("weather_lag_hours", 24):
        raise ValueError("Input must contain more rows than the weather lag.")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Timestamps must be valid, unique and sorted in UTC.")
    if not dates.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Expected a complete hourly UTC grid; no implicit filling or aggregation.")
    # Select load and historical weather; prices are not forecast-time inputs.
    target = dataset["target_column"]
    weather = dataset.get("historical_weather", [])
    columns = [target, "is_holiday", *weather]
    numeric = raw[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Selected columns must be finite and complete.")
    numeric.index = pd.DatetimeIndex(dates)
    hourly = numeric
    # Every horizon step must use weather observed by the forecast origin.
    lag = dataset.get("weather_lag_hours", 24)
    if lag < horizon:
        raise ValueError("Weather lag must cover the entire horizon to prevent leakage.")
    # Calendar features follow local time; model rows stay on the UTC grid.
    local = hourly.index.tz_convert("Europe/Berlin")
    X = pd.DataFrame(index=hourly.index)
    X["hour"] = local.hour
    X["weekday"] = local.dayofweek
    X["month"] = local.month
    X["is_holiday"] = hourly["is_holiday"]
    # Shift observed weather instead of exposing realized future weather.
    for column in weather:
        X[f"{column}_lag_{lag}h"] = hourly[column].shift(lag)
    # Trim the same warm-up rows with and without weather for comparable splits.
    X = X.iloc[lag:].reset_index(drop=True)
    hourly = hourly.iloc[lag:]
    y = hourly[target].reset_index(drop=True)
    series = pd.DataFrame({"timestamp": hourly.index, "target": y.to_numpy()})
    metadata = {
        "variant": "regional_hourly_v2", "raw_rows": len(raw), "hourly_rows": len(y),
        "aggregation": "none; input is already hourly", "timezone": "UTC; calendar in Europe/Berlin",
        "missing_policy": "reject; no interpolation", "warmup_hours": lag,
        "weather_availability": "observed weather lagged by at least the forecast horizon",
        "features": list(X.columns),
        "forecast_task": f"next {horizon} elapsed hours; not an auction-timed delivery-day forecast",
    }
    return y, X, series, metadata

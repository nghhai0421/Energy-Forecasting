"""Combine configured consumption, weather, and holiday inputs for Chronos-2."""

import numpy as np
import pandas as pd

from preprocessing.add_seasonal_features import add_seasonal_features


def load_data(dataset, period="development"):
    """Prepare aligned targets and covariates from the configured CSV sources.

    Parameters
    ----------
    dataset : dict
        Sources, timestamp_column, target_column, frequency, and covariate groups.
        Source paths must be resolved before loading.
    period : {'development', 'demo'}, default 'development'
        Consumption and weather file group to read.

    Returns
    -------
    tuple of pandas.Series, pandas.DataFrame, pandas.DataFrame, dict
        Imputed target, covariates, original timestamp/target observations, and
        preparation metadata. All outputs share positional row indices.

    Raises
    ------
    ValueError
        If timestamps, source coverage, feature selection, holiday coverage, or
        leading missing values prevent preparation.

    Notes
    -----
    Missing/nonfinite values and negative wind speeds use the mean of earlier
    valid observations. Valid negative temperatures, dew points, and residual
    loads are preserved. Original targets remain available for honest scoring.
    Calendar features use the consumption source timezone and are known-future
    covariates generated independently from each timestamp.
    """
    # Set Inputs
    if period not in {"development", "demo"}:
        raise ValueError("period must be development or demo.")
    sources = dataset["sources"]
    target = dataset["target_column"]
    timestamp = dataset["timestamp_column"]
    frequency = pd.Timedelta(dataset["frequency"])
    if frequency != pd.Timedelta(minutes=15):
        raise ValueError("This loader expects quarter-hourly frequency: 15min.")
    covariates = dataset.get("covariates", {})
    historical = covariates.get("historical", [])
    future = covariates.get("future", [])
    features = historical + future
    consumption_columns = sources["consumption"].get("columns", [])
    weather_columns = sources["weather"]["columns"]
    input_columns = [target, *consumption_columns, *weather_columns, "is_holiday"]
    if len(input_columns) != len(set(input_columns)):
        raise ValueError("Target and source feature column names must be distinct.")
    if len(features) != len(set(features)):
        raise ValueError("Covariates must select each feature once.")
    if set(consumption_columns).intersection(future):
        raise ValueError("Observed consumption covariates must be historical.")
    if set(weather_columns).intersection(future):
        raise ValueError("Observed weather must be historical; future weather requires forecasts.")

    # Load Sources
    frames = {}
    for name in ("consumption", "weather"):
        source = sources[name]
        raw = pd.read_csv(source[period])
        column = source["timestamp_column"]
        date_format = "%b %d, %Y %I:%M %p" if name == "consumption" else "%Y-%m-%d %H:%M:%S"
        dates = pd.to_datetime(raw[column], format=date_format, errors="raise")
        if dates.isna().any():
            raise ValueError(f"Missing timestamps in {name}.")
        # SMARD repeats autumn local times in chronological file order.
        dates = dates.dt.tz_localize(source["timezone"], ambiguous="infer").dt.tz_convert("UTC")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{name} timestamps must be unique and sorted in UTC.")
        if not dates.diff().dropna().eq(frequency).all() or dates.empty:
            raise ValueError(f"{name} must contain a complete 15-minute UTC grid.")
        columns = [target, *consumption_columns] if name == "consumption" else weather_columns
        frame = raw[columns].apply(pd.to_numeric, errors="coerce")
        frame.index = pd.DatetimeIndex(dates, name=timestamp)
        frames[name] = frame

    # Align Weather
    combined = frames["consumption"].join(frames["weather"], how="left", validate="one_to_one")
    if not combined.index.isin(frames["weather"].index).any():
        raise ValueError("Consumption and weather timestamps do not overlap.")
    missing_weather_rows = int((~combined.index.isin(frames["weather"].index)).sum())
    combined = combined.replace([np.inf, -np.inf], np.nan)
    observed = combined[target].copy()

    # Impute Values
    imputation = {}
    for column in [target, *consumption_columns, *weather_columns]:
        values = combined[column]
        invalid_negative = values.lt(0) if column.startswith("wind_speed_") else pd.Series(False, index=values.index)
        valid = values.mask(invalid_negative)
        missing = valid.isna()
        combined[column] = valid.fillna(valid.expanding(min_periods=1).mean().shift(1))
        if combined[column].isna().any():
            raise ValueError(f"Cannot mean-impute leading missing values in {column}; prior history is required.")
        imputation[column] = {
            "missing_or_nonnumeric": int(values.isna().sum()),
            "invalid_negative": int(invalid_negative.sum()),
            "imputed": int(missing.sum()),
        }

    # Add Known-Future Features
    combined = combined.reset_index()
    known = prepare_future_covariates(combined[timestamp], dataset)
    combined[known.columns] = known
    expected = {*consumption_columns, *weather_columns, *known.columns}
    if set(features) != expected:
        raise ValueError("Covariates must select each source, holiday, and calendar feature once.")

    # Format Inputs
    series = pd.DataFrame({"timestamp": combined[timestamp], "target": observed.to_numpy()})
    y = combined[target].reset_index(drop=True)
    X = combined[features].reset_index(drop=True)
    metadata = {
        "period": period, "frequency": dataset["frequency"], "rows": len(y),
        "weather_alignment_missing_rows": missing_weather_rows,
        "imputation": {"method": "expanding mean of prior valid observations", "columns": imputation},
        "unobserved_targets": int(observed.isna().sum()),
        "historical_covariates": historical, "future_covariates": future,
        "calendar_timezone": sources["consumption"]["timezone"],
    }
    return y, X, series, metadata


def prepare_future_covariates(timestamps, dataset):
    """Generate calendar and holiday inputs without future observed values.

    Parameters
    ----------
    timestamps : pandas.Series or pandas.DatetimeIndex
        Timezone-aware, sorted, contiguous horizon timestamps. Series retain
        their index; a DatetimeIndex becomes the returned row index.
    dataset : dict
        Timestamp column, frequency, source timezones, holiday CSV path, and
        ordered future covariate names from the experiment configuration.

    Returns
    -------
    pandas.DataFrame
        Ordered calendar and holiday covariates with the input row labels.

    Raises
    ------
    ValueError
        If timestamps, frequency, holiday coverage, or future feature names are invalid.
    """
    # Check Timestamps
    if isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.Series(timestamps, index=timestamps)
    if not isinstance(timestamps, pd.Series) or pd.api.types.is_numeric_dtype(timestamps):
        raise ValueError("Supply a timestamp Series or DatetimeIndex.")
    dates = pd.to_datetime(timestamps, format="mixed", errors="raise")
    if dates.empty or dates.isna().any() or not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError("Future timestamps must be nonempty, valid, and timezone-aware.")
    if dates.dt.tz is None:
        raise ValueError("Future timestamps must be timezone-aware.")
    dates = dates.dt.tz_convert("UTC")
    frequency = pd.Timedelta(dataset["frequency"])
    if not dates.index.is_unique or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Future timestamps and row labels must be unique and sorted.")
    if frequency != pd.Timedelta(minutes=15) or not dates.diff().dropna().eq(frequency).all():
        raise ValueError("Future timestamps must form a complete 15-minute UTC grid.")

    # Generate Calendar Features
    timestamp = dataset["timestamp_column"]
    frame = add_seasonal_features(
        pd.DataFrame({timestamp: dates}), timestamp,
        timezone=dataset["sources"]["consumption"]["timezone"],
    )
    calendar_columns = frame.columns.drop(timestamp).tolist()
    future = dataset["covariates"]["future"]
    if not set(calendar_columns).issubset(future):
        raise ValueError("Calendar features must be listed as future covariates.")
    if len(future) != len(set(future)) or set(future) != {*calendar_columns, "is_holiday"}:
        raise ValueError("Future covariates must contain exactly the calendar and holiday features.")

    # Match Holidays
    source = dataset["sources"]["holidays"]
    holidays = pd.read_csv(source["path"])
    holiday_dates = pd.to_datetime(holidays[source["date_column"]], errors="raise")
    if holiday_dates.isna().any():
        raise ValueError("Holiday dates must not be missing.")
    local_dates = dates.dt.tz_convert(source["timezone"])
    missing_years = set(local_dates.dt.year) - set(holiday_dates.dt.year)
    if missing_years:
        raise ValueError(f"Regenerate the holiday CSV to cover years {sorted(missing_years)}.")
    frame["is_holiday"] = local_dates.dt.date.isin(set(holiday_dates.dt.date)).astype(int)
    return frame[future].copy()

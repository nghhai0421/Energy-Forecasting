"""Prepare final split CSVs and origin-safe GBT inputs independently of other models.

Use the same source timezones, causal missing-value policy and shared calendar
routine as Chronos-2. Model-specific lag features use observed values only.
"""

import numpy as np
import pandas as pd
import holidays

from preprocessing.add_seasonal_features import add_seasonal_features

INTERVAL = pd.Timedelta(minutes=15)
FEATURE_VERSION = "split_gbt_lags_calendar_v1"


def boundary(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Boundary timestamps must be timezone-aware.")
    return stamp.tz_convert("UTC")


def validate_index(index):
    """Validate a continuous UTC quarter-hour sequence, including DST crossings."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Use a timezone-aware DatetimeIndex.")
    if index.empty or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Timestamps must be nonempty, valid, unique and sorted.")
    if not index.equals(index.floor("15min")) or not (index[1:] - index[:-1] == INTERVAL).all():
        raise ValueError("Expected a complete 15-minute timestamp grid.")


def required_history(dataset):
    lags = dataset["lag_hours"]
    delay = dataset["observation_delay_minutes"]
    if not lags or len(set(lags)) != len(lags) or any(not isinstance(v, int) or v <= 0 for v in lags):
        raise ValueError("Lag hours must be distinct positive integers.")
    if not isinstance(delay, int) or delay < 0 or delay % 15:
        raise ValueError("Observation delay must be a nonnegative multiple of 15 minutes.")
    return max(lags) * 4 + delay // 15


def load_holidays(dataset, years=None):
    """Read team calendar and generate absent years of known BW holidays."""
    source = dataset["sources"]["holidays"]
    dates = pd.to_datetime(pd.read_csv(source["path"])[source["date_column"]], errors="raise")
    if dates.isna().any():
        raise ValueError("Holiday dates cannot be missing.")
    result = set(dates.dt.strftime("%Y-%m-%d"))
    missing_years = set(years or []) - set(dates.dt.year)
    if missing_years:
        result.update(day.isoformat() for day in holidays.Germany(years=sorted(missing_years), subdiv="BW"))
    return sorted(result)


def load_sources(dataset, period="development", initial_state=None):
    """Read ONLY the requested source period; return prepared and original targets.

    The returned imputation state contains sums/counts of valid observations, not
    imputed values. Pass development state to demo replay to maintain causality.
    All filling uses strictly preceding observations; labels remain unmodified.
    """
    if period not in {"development", "demo"} or dataset["frequency"] != "15min":
        raise ValueError("Expected development/demo and frequency 15min.")
    frames = {}
    for name in ("consumption", "weather", "prices"):
        source = dataset["sources"][name]
        raw = pd.read_csv(source[period])
        date_format = "%Y-%m-%d %H:%M:%S" if name == "weather" else "%b %d, %Y %I:%M %p"
        dates = pd.to_datetime(raw[source["timestamp_column"]], format=date_format, errors="raise")
        dates = dates.dt.tz_localize(source["timezone"], ambiguous="infer").dt.tz_convert("UTC")
        index = pd.DatetimeIndex(dates, name=dataset["timestamp_column"])
        validate_index(index)
        frame = raw[source["columns"]].apply(pd.to_numeric, errors="coerce")
        frame.index = index
        frames[name] = frame
    combined = frames["consumption"].copy()
    alignment = {}
    for name in ("weather", "prices"):
        if not combined.index.isin(frames[name].index).any():
            raise ValueError(f"No overlapping {name} timestamps.")
        alignment[name] = int((~combined.index.isin(frames[name].index)).sum())
        combined = combined.join(frames[name], how="left", validate="one_to_one")
    combined = combined.replace([np.inf, -np.inf], np.nan)
    target = dataset["target_column"]
    observed = combined[target].copy()
    state, counts = {}, {}
    for column in combined:
        values = combined[column]
        negative = values.lt(0) if column.startswith("wind_speed_") else pd.Series(False, index=values.index)
        valid = values.mask(negative)
        previous = (initial_state or {}).get(column, {"sum": 0.0, "count": 0})
        sums = valid.fillna(0).cumsum().shift(1, fill_value=0) + previous["sum"]
        totals = valid.notna().cumsum().shift(1, fill_value=0) + previous["count"]
        combined[column] = valid.fillna(sums / totals.replace(0, np.nan))
        if not np.isfinite(combined[column]).all():
            raise ValueError(f"Cannot fill leading missing observations in {column} without prior history.")
        state[column] = {"sum": float(valid.sum() + previous["sum"]),
                         "count": int(valid.notna().sum() + previous["count"])}
        counts[column] = {"missing": int(values.isna().sum()), "invalid_negative": int(negative.sum()),
                          "imputed": int(valid.isna().sum())}
    metadata = {"period": period, "rows": len(combined), "alignment_missing_rows": alignment,
                "imputation": counts, "unobserved_targets": int(observed.isna().sum()),
                "start": combined.index[0].isoformat(), "end": combined.index[-1].isoformat()}
    return combined, observed, state, metadata


def build_features(history, dates, dataset, holiday_dates):
    """Build identical timestamp-aligned tabular inputs for training and inference.

    Historical values are looked up by timestamp, not position. The caller must
    restrict inference history to observations preceding the forecast origin.
    All calendar columns come from the team's shared add_seasonal_features().
    """
    validate_index(history.index)
    dates = pd.DatetimeIndex(dates).tz_convert("UTC")
    validate_index(dates)
    required_history(dataset)
    features = pd.DataFrame(index=dates)
    for alias, column in dataset["lag_fields"].items():
        for hours in dataset["lag_hours"]:
            offset = pd.Timedelta(hours=hours, minutes=dataset["observation_delay_minutes"])
            features[f"{alias}_lag_{hours}h_delay_{dataset['observation_delay_minutes']}m"] = (
                history[column].reindex(dates - offset).to_numpy())
    calendar = add_seasonal_features(
        pd.DataFrame({dataset["timestamp_column"]: dates}, index=dates), dataset["timestamp_column"],
        timezone=dataset["sources"]["consumption"]["timezone"],
    ).drop(columns=dataset["timestamp_column"])
    features = features.join(calendar)
    local = dates.tz_convert(dataset["sources"]["holidays"]["timezone"])
    holidays = pd.DatetimeIndex(pd.to_datetime(holiday_dates))
    if not set(local.year).issubset(set(holidays.year)):
        raise ValueError("Holiday file does not cover the requested year.")
    features["is_holiday"] = np.isin(local.strftime("%Y-%m-%d"), holiday_dates).astype(int)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Missing lag inputs; provide sufficient prepared history.")
    return features

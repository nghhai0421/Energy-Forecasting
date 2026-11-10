"""Generate timestamp-based calendar features for training and inference."""

import numpy as np
import pandas as pd


def add_seasonal_features(
    df: pd.DataFrame,
    timestamp_column: str,
    timezone: str = "Europe/Berlin",
    input_timezone: str | None = None,
) -> pd.DataFrame:
    """Add 17 calendar features immediately after the timestamp column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows; original values, column order, and index are preserved.
    timestamp_column : str
        Column containing parseable timestamps.
    timezone : str, default 'Europe/Berlin'
        Timezone used to calculate calendar features.
    input_timezone : str or None, default None
        Source timezone for naive timestamps; required when no timezone is present.

    Returns
    -------
    pandas.DataFrame
        Copy with eight calendar sine/cosine pairs and numeric year.

    Raises
    ------
    ValueError
        If timestamps are missing, invalid, numeric, lack a required timezone,
        or column names conflict.

    Notes
    -----
    Encodings follow TabPFN-TS calendar conventions. Mixed timezone offsets must
    be normalized to UTC; ambiguous or nonexistent local DST times raise.
    """
    # Check Input
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_column}")
    if pd.api.types.is_numeric_dtype(df[timestamp_column]):
        raise ValueError("Convert numeric timestamps to explicit datetimes first.")

    # Parse Timestamps
    timestamps = pd.to_datetime(df[timestamp_column], format="mixed", errors="raise")
    if timestamps.isna().any():
        raise ValueError("Timestamps must not be missing.")
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError("Normalize mixed timezone offsets to UTC before calling this function.")
    if timestamps.dt.tz is None:
        if input_timezone is None:
            raise ValueError("Supply input_timezone for timezone-naive timestamps.")
        # Reject ambiguous or nonexistent local DST times.
        timestamps = timestamps.dt.tz_localize(input_timezone)
    timestamps = timestamps.dt.tz_convert(timezone)

    # Extract Calendar Values
    components = {
        "second_of_minute": (timestamps.dt.second, 60),
        "minute_of_hour": (timestamps.dt.minute, 60),
        "hour_of_day": (timestamps.dt.hour, 24),
        "day_of_week": (timestamps.dt.dayofweek, 7),
        "day_of_month": (timestamps.dt.day - 1, 30.5),
        "day_of_year": (timestamps.dt.dayofyear - 1, 365),
        "week_of_year": (timestamps.dt.isocalendar().week - 1, 52),
        "month_of_year": (timestamps.dt.month - 1, 12),
    }

    # Encode Features
    features = {}
    for name, (values, period) in components.items():
        # Match TabPFN-TS's period-minus-one convention exactly.
        angle = 2 * np.pi * values.to_numpy(dtype=np.int32) / (period - 1)
        features[f"calendar_{name}_sin"] = np.sin(angle)
        features[f"calendar_{name}_cos"] = np.cos(angle)
    features["calendar_year"] = timestamps.dt.year.to_numpy(dtype=np.int32)

    collisions = df.columns.intersection(list(features))
    if len(collisions):
        raise ValueError(f"Calendar feature columns already exist: {list(collisions)}")

    # Attach Features
    position = df.columns.get_loc(timestamp_column) + 1
    feature_df = pd.DataFrame(features, index=df.index)
    return pd.concat(
        [df.iloc[:, :position], feature_df, df.iloc[:, position:]], axis=1
    ).copy()

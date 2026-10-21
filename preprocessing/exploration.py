"""Explore raw energy consumption, day-ahead price, and weather datasets.

Checks each dataset for:
1. Incorrect data types and non-numeric placeholder tokens (e.g., '-', '#NUM!')
2. Missing values (raw NaNs and post-coercion NaNs) and temporal gaps
3. Duplicate rows and duplicate timestamps (including DST fall-back transitions)
4. Statistical outliers (IQR / Z-score) and domain-specific impossible values
"""
from pathlib import Path

import numpy as np
import pandas as pd

def get_project_root() -> Path:
    """Find the repository root whether run as a script or in an interactive cell."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "Dataset").is_dir():
            return candidate
    return start


PROJECT_ROOT = get_project_root()
DATASET_DIR = PROJECT_ROOT / "Dataset"

CONSUMPTION_FILE = DATASET_DIR / "Actual_consumption_201701010000_202601010000_Quarterhour.xlsx"
PRICES_FILE = DATASET_DIR / "Day-ahead_prices_201701010000_202601010000_Quarterhour.xlsx"
WEATHER_FILE = DATASET_DIR / "open-meteo-48.83N9.15E293m.xlsx"


def load_raw_datasets() -> dict[str, pd.DataFrame]:
    """Load the three raw Excel datasets without altering their values."""
    print("Loading raw Excel files from:", DATASET_DIR)
    consumption = pd.read_excel(CONSUMPTION_FILE, skiprows=9)
    prices = pd.read_excel(PRICES_FILE, skiprows=9)
    weather = pd.read_excel(WEATHER_FILE, skiprows=3)
    return {
        "Actual Consumption": consumption,
        "Day-Ahead Prices": prices,
        "Open-Meteo Weather": weather,
    }


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse timestamps from either string dates or Excel serial day numbers."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        # Excel serial date (1899-12-30 origin)
        return pd.to_datetime(series, unit="D", origin="1899-12-30").dt.round("min")
    return pd.to_datetime(series, format="%b %d, %Y %I:%M %p", errors="coerce")


def check_data_types(df: pd.DataFrame, timestamp_cols: list[str]) -> pd.DataFrame:
    """Inspect column dtypes, identify non-numeric tokens, and coerce value columns."""
    print("\n--- 1. Data Types & Non-Numeric Tokens ---")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    value_cols = [c for c in df.columns if c not in timestamp_cols]
    coerced_df = df.copy()

    type_rows = []
    for col in df.columns:
        raw_dtype = str(df[col].dtype)
        if col in timestamp_cols:
            parsed_ts = parse_timestamp_series(df[col])
            invalid_count = int(parsed_ts.isna().sum() - df[col].isna().sum())
            coerced_df[col] = parsed_ts
            type_rows.append({
                "column": col,
                "raw_dtype": raw_dtype,
                "expected_type": "datetime",
                "non_numeric_or_invalid": invalid_count,
                "sample_invalid_tokens": "",
            })
        else:
            s = df[col]
            numeric_s = pd.to_numeric(s, errors="coerce")
            non_numeric_mask = s.notna() & numeric_s.isna()
            non_numeric_count = int(non_numeric_mask.sum())
            sample_tokens = (
                s[non_numeric_mask].astype(str).value_counts().head(3).to_dict()
                if non_numeric_count > 0
                else {}
            )
            coerced_df[col] = numeric_s
            type_rows.append({
                "column": col,
                "raw_dtype": raw_dtype,
                "expected_type": "float64",
                "non_numeric_or_invalid": non_numeric_count,
                "sample_invalid_tokens": str(sample_tokens) if sample_tokens else "None",
            })

    report = pd.DataFrame(type_rows)
    print(report.to_string(index=False))
    return coerced_df


def check_duplicates_and_timestamps(df: pd.DataFrame, primary_ts_col: str) -> None:
    """Check for full-row duplicates, duplicate timestamps, DST transitions, and gaps."""
    print("\n--- 2. Duplicates & Timestamp Integrity ---")
    full_dups = int(df.duplicated().sum())
    ts = df[primary_ts_col]
    ts_dups = int(ts.duplicated().sum())

    print(f"Exact full-row duplicates: {full_dups:,}")
    print(f"Duplicate '{primary_ts_col}' timestamps: {ts_dups:,}")

    valid_ts = ts.dropna()
    if valid_ts.empty:
        print("No valid timestamps found.")
        return

    start_ts, end_ts = valid_ts.min(), valid_ts.max()
    print(f"Time range: {start_ts} -> {end_ts}")

    # Check if duplicate timestamps occur only on October DST fall-back Sundays (02:00-02:45)
    if ts_dups > 0:
        dup_timestamps = valid_ts[valid_ts.duplicated(keep=False)]
        oct_dst_mask = (
            (dup_timestamps.dt.month == 10)
            & (dup_timestamps.dt.weekday == 6)
            & (dup_timestamps.dt.hour == 2)
        )
        dst_dup_rows = int(oct_dst_mask.sum())
        non_dst_dup_rows = int((~oct_dst_mask).sum())
        print(
            f"  -> Rows in duplicate timestamps during Oct DST fall-back (02:00-02:45): {dst_dup_rows:,}"
        )
        print(f"  -> Unexpected duplicate timestamp rows outside Oct DST: {non_dst_dup_rows:,}")

    # Check for missing 15-minute intervals against a naive wall-clock grid
    expected_grid = pd.date_range(start=start_ts, end=end_ts, freq="15min")
    missing_wall_clock = expected_grid.difference(valid_ts.unique())
    march_dst_mask = (
        (missing_wall_clock.month == 3)
        & (missing_wall_clock.weekday == 6)
        & (missing_wall_clock.hour == 2)
    )
    spring_forward_count = int(march_dst_mask.sum())
    unexpected_gaps = int((~march_dst_mask).sum())
    print(
        f"Missing 15-min wall-clock timestamps: {len(missing_wall_clock):,} "
        f"(March DST spring-forward: {spring_forward_count:,}, unexpected gaps: {unexpected_gaps:,})"
    )


def check_missing_data(raw_df: pd.DataFrame, coerced_df: pd.DataFrame, primary_ts_col: str) -> None:
    """Report missing data before and after coercing non-numeric tokens."""
    print("\n--- 3. Missing Data ---")
    total_rows = len(coerced_df)
    ts = coerced_df[primary_ts_col]

    missing_rows = []
    for col in coerced_df.columns:
        raw_na = int(raw_df[col].isna().sum())
        coerced_na = int(coerced_df[col].isna().sum())
        non_num_na = coerced_na - raw_na
        pct_missing = (coerced_na / total_rows) * 100.0 if total_rows else 0.0

        valid_mask = coerced_df[col].notna()
        first_valid = ts[valid_mask].min() if valid_mask.any() else None
        last_valid = ts[valid_mask].max() if valid_mask.any() else None

        missing_rows.append({
            "column": col,
            "raw_nulls": raw_na,
            "placeholder_nulls": non_num_na,
            "total_missing": coerced_na,
            "missing_%": round(pct_missing, 2),
            "first_valid_ts": str(first_valid) if first_valid is not None else "None",
            "last_valid_ts": str(last_valid) if last_valid is not None else "None",
        })

    report = pd.DataFrame(missing_rows)
    print(report.to_string(index=False))


def check_outliers_and_impossible_values(
    coerced_df: pd.DataFrame,
    timestamp_cols: list[str],
    dataset_name: str,
) -> None:
    """Compute distribution stats, IQR/Z-score outliers, and domain-specific impossible values."""
    print("\n--- 4. Outliers & Impossible / Implausible Values ---")
    value_cols = [c for c in coerced_df.columns if c not in timestamp_cols]

    stats_rows = []
    for col in value_cols:
        s = coerced_df[col].dropna()
        if s.empty:
            continue
        q01, q25, q50, q75, q99 = s.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        iqr = q75 - q25
        iqr_outliers = int(((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum())
        extreme_iqr = int(((s < (q25 - 3.0 * iqr)) | (s > (q75 + 3.0 * iqr))).sum())
        std = s.std()
        z_outliers = int((((s - s.mean()).abs() / std) > 3.0).sum()) if std > 0 else 0

        stats_rows.append({
            "column": col,
            "min": round(float(s.min()), 2),
            "p01": round(float(q01), 2),
            "median": round(float(q50), 2),
            "mean": round(float(s.mean()), 2),
            "p99": round(float(q99), 2),
            "max": round(float(s.max()), 2),
            "iqr_1.5x": iqr_outliers,
            "iqr_3.0x": extreme_iqr,
            "z_gt_3": z_outliers,
        })

    stats_report = pd.DataFrame(stats_rows)
    print(stats_report.to_string(index=False))

    # Domain-specific impossible / implausible value checks
    print("\nDomain-specific sanity checks:")
    if dataset_name == "Actual Consumption":
        for col in ["grid load [MWh]", "Grid load incl. hydro pumped storage [MWh]"]:
            if col in coerced_df.columns:
                non_positive = int((coerced_df[col] <= 0).sum())
                print(f"  - {col} <= 0 (impossible non-positive grid load): {non_positive:,}")
        if "Hydro pumped storage [MWh]" in coerced_df.columns:
            neg_hydro = int((coerced_df["Hydro pumped storage [MWh]"] < 0).sum())
            print(f"  - Hydro pumped storage [MWh] < 0: {neg_hydro:,}")

    elif dataset_name == "Day-Ahead Prices":
        target_col = "Germany/Luxembourg [€/MWh]"
        if target_col in coerced_df.columns:
            s = coerced_df[target_col].dropna()
            neg_prices = int((s < 0).sum())
            below_cap = int((s < -500).sum())
            above_cap = int((s > 4000).sum())
            print(
                f"  - {target_col}: negative prices (< 0 €/MWh, valid market phenomenon): {neg_prices:,}"
            )
            print(
                f"  - {target_col}: outside SDAC market caps (<-500 or >4000 €/MWh): "
                f"{below_cap + above_cap:,}"
            )

    elif dataset_name == "Open-Meteo Weather":
        checks = [
            ("temperature_2m (°C)", lambda x: (x < -40) | (x > 50), "outside [-40, 50] °C"),
            ("dew_point_2m (°C)", lambda x: (x < -50) | (x > 45), "outside [-50, 45] °C"),
            ("relative_humidity_2m (%)", lambda x: (x < 0) | (x > 100), "outside [0, 100] %"),
            ("precipitation (mm)", lambda x: x < 0, "< 0 mm"),
            ("shortwave_radiation (W/m²)", lambda x: (x < 0) | (x > 1500), "outside [0, 1500] W/m²"),
            ("direct_radiation (W/m²)", lambda x: (x < 0) | (x > 1500), "outside [0, 1500] W/m²"),
            ("sunshine_duration (s)", lambda x: (x < 0) | (x > 900), "outside [0, 900] s per 15-min"),
            ("wind_speed_10m (km/h)", lambda x: (x < 0) | (x > 200), "outside [0, 200] km/h"),
            ("weather_code (wmo code)", lambda x: (x < 0) | (x > 99), "outside [0, 99] WMO range"),
        ]
        for col, rule, desc in checks:
            if col in coerced_df.columns:
                s = coerced_df[col].dropna()
                violations = int(rule(s).sum())
                print(f"  - {col} {desc}: {violations:,}")

        if {"temperature_2m (°C)", "dew_point_2m (°C)"}.issubset(coerced_df.columns):
            temp = coerced_df["temperature_2m (°C)"]
            dew = coerced_df["dew_point_2m (°C)"]
            dew_gt_temp = int((dew > temp + 1e-6).sum())
            print(f"  - dew_point_2m > temperature_2m (physically impossible): {dew_gt_temp:,}")


def run_exploration() -> None:
    """Run all exploration checks across the three datasets."""
    datasets = load_raw_datasets()

    timestamp_config = {
        "Actual Consumption": (["Start date", "End date"], "Start date"),
        "Day-Ahead Prices": (["Start date", "End date"], "Start date"),
        "Open-Meteo Weather": (["time"], "time"),
    }

    for name, raw_df in datasets.items():
        print("\n" + "=" * 80)
        print(f"DATASET: {name}")
        print("=" * 80)
        ts_cols, primary_ts = timestamp_config[name]
        coerced_df = check_data_types(raw_df, ts_cols)
        check_duplicates_and_timestamps(coerced_df, primary_ts)
        check_missing_data(raw_df, coerced_df, primary_ts)
        check_outliers_and_impossible_values(coerced_df, ts_cols, name)


if __name__ == "__main__":
    run_exploration()

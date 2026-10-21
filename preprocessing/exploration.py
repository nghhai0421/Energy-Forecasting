"""Explore raw datasets, create BW holiday dataset, align timestamps, merge, and analyze correlations.

Steps performed:
1. Explore raw datasets (Actual Consumption, Day-Ahead Prices, Open-Meteo Weather) for:
   - Incorrect data types and non-numeric placeholder tokens ('-', '#NUM!')
   - Duplicates and timestamp integrity (DST summertime/wintertime transitions)
   - Missing values
   - Outliers and domain-specific impossible values
2. Generate Baden-Württemberg (BW) holiday dataset and save to Dataset/df_holiday.xlsx
3. Align timestamps across all datasets (handling Europe/Berlin DST vs. fixed UTC+2 in Open-Meteo)
4. Merge the key variables defined in Dataset/README.md at 15-minute resolution
5. Summarize the target variable ('grid load [MWh]')
6. Compute and display the correlation matrix across all merged variables
"""

from datetime import timedelta, timezone
from pathlib import Path

import holidays
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
HOLIDAY_FILE = DATASET_DIR / "df_holiday.xlsx"

# Key variables specified in Dataset/README.md
TARGET_COL = "grid load [MWh]"
PRICE_COL = "Germany/Luxembourg [€/MWh]"
WEATHER_COLS = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "shortwave_radiation (W/m²)",
    "wind_speed_10m (km/h)",
    "dew_point_2m (°C)",
    "weather_code (wmo code)",
]


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
        if PRICE_COL in coerced_df.columns:
            s = coerced_df[PRICE_COL].dropna()
            neg_prices = int((s < 0).sum())
            below_cap = int((s < -500).sum())
            above_cap = int((s > 4000).sum())
            print(
                f"  - {PRICE_COL}: negative prices (< 0 €/MWh, valid market phenomenon): {neg_prices:,}"
            )
            print(
                f"  - {PRICE_COL}: outside SDAC market caps (<-500 or >4000 €/MWh): "
                f"{below_cap + above_cap:,}"
            )

    elif dataset_name == "Open-Meteo Weather":
        checks = [
            ("temperature_2m (°C)", lambda x: (x < -40) | (x > 50), "outside [-40, 50] °C"),
            ("dew_point_2m (°C)", lambda x: (x < -50) | (x > 45), "outside [-50, 45] °C"),
            ("relative_humidity_2m (%)", lambda x: (x < 0) | (x > 100), "outside [0, 100] %"),
            ("precipitation (mm)", lambda x: x < 0, "< 0 mm"),
            ("shortwave_radiation (W/m²)", lambda x: (x < 0), "<0 W/m²"),
            ("direct_radiation (W/m²)", lambda x: (x < 0),  "<0 W/m²"),
            ("sunshine_duration (s)", lambda x: (x < 0) | (x > 900), "outside [0, 900] s per 15-min"),
            ("wind_speed_10m (km/h)", lambda x: (x < 0),  "<0 km/h"),
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


def create_bw_holiday_dataset(
    years: range = range(2017, 2027),
    save_path: Path = HOLIDAY_FILE,
) -> pd.DataFrame:
    """Create a Baden-Württemberg (BW) holiday dataset and save it in Dataset/."""
    print("\n" + "=" * 80)
    print("HOLIDAY DATASET: Baden-Württemberg (BW)")
    print("=" * 80)
    bw_holidays = holidays.Germany(years=years, subdiv="BW")
    df_holidays = (
        pd.DataFrame(list(bw_holidays.items()), columns=["Datum", "Feiertag"])
        .sort_values("Datum")
        .reset_index(drop=True)
    )
    df_holidays["Datum"] = pd.to_datetime(df_holidays["Datum"])
    df_holidays.to_excel(save_path, index=False)
    print(f"Saved BW holiday dataset ({len(df_holidays)} holidays) to: {save_path}")
    print(df_holidays.head(10).to_string(index=False))
    return df_holidays


def align_and_merge_datasets(
    datasets: dict[str, pd.DataFrame],
    df_holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Match timestamps (addressing DST summertime vs. wintertime) and merge key variables.

    Timezone / DST alignment details:
    - SMARD (Actual Consumption & Day-Ahead Prices): recorded in local 'Europe/Berlin' wall-clock
      time (CET UTC+1 in winter, CEST UTC+2 in summer). Localizing with ambiguous='infer' resolves
      the 36 repeated October fall-back timestamps, and converting to UTC yields a strictly unique,
      continuous 15-minute UTC index.
    - Open-Meteo Weather: exported with a fixed UTC+2 offset year-round (utc_offset_seconds = 7200).
      We localize to fixed UTC+2 and convert to UTC so winter and summer observations match the
      exact physical point in time.
    - Holidays: matched against the local 'Europe/Berlin' calendar date.
    """
    print("\n" + "=" * 80)
    print("TIMESTAMP ALIGNMENT & MERGING (15-min Resolution)")
    print("=" * 80)

    # 1. Consumption: keep 'grid load [MWh]'
    cons_raw = datasets["Actual Consumption"]
    cons_berlin = parse_timestamp_series(cons_raw["Start date"]).dt.tz_localize(
        "Europe/Berlin", ambiguous="infer"
    )
    cons_df = pd.DataFrame({
        "timestamp_utc": cons_berlin.dt.tz_convert("UTC"),
        "timestamp_berlin": cons_berlin,
        TARGET_COL: pd.to_numeric(cons_raw[TARGET_COL], errors="coerce"),
    })

    # 2. Day-Ahead Prices: keep 'Germany/Luxembourg [€/MWh]' and repeat hourly value 4x if needed
    prices_raw = datasets["Day-Ahead Prices"]
    prices_berlin = parse_timestamp_series(prices_raw["Start date"]).dt.tz_localize(
        "Europe/Berlin", ambiguous="infer"
    )
    price_series = pd.to_numeric(prices_raw[PRICE_COL], errors="coerce")
    # Ensure hourly values fill all four 15-minute intervals within the same hour
    price_series = price_series.ffill(limit=3)
    prices_df = pd.DataFrame({
        "timestamp_utc": prices_berlin.dt.tz_convert("UTC"),
        PRICE_COL: price_series,
    })

    # 3. Open-Meteo Weather: keep key variables from Dataset/README.md
    weather_raw = datasets["Open-Meteo Weather"]
    fixed_utc_plus_2 = timezone(timedelta(hours=2))
    weather_ts_utc = (
        parse_timestamp_series(weather_raw["time"])

        .dt.tz_localize(fixed_utc_plus_2)
        .dt.tz_convert("UTC")
    )
    weather_df = pd.DataFrame({"timestamp_utc": weather_ts_utc})
    for col in WEATHER_COLS:
        weather_df[col] = pd.to_numeric(weather_raw[col], errors="coerce")

    # 4. Merge on unique 15-minute UTC timestamps
    merged = (
        cons_df.merge(prices_df, on="timestamp_utc", how="inner")
        .merge(weather_df, on="timestamp_utc", how="inner")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    # 5. Add BW holiday indicator based on local Europe/Berlin date
    holiday_dates = set(pd.to_datetime(df_holidays["Datum"]).dt.date)
    berlin_dates = merged["timestamp_berlin"].dt.date
    merged["is_holiday"] = berlin_dates.isin(holiday_dates).astype(int)

    print(f"Merged dataset shape (full overlap): {merged.shape[0]:,} rows x {merged.shape[1]} columns")
    print(f"UTC range:    {merged['timestamp_utc'].min()} -> {merged['timestamp_utc'].max()}")
    print(f"Berlin range: {merged['timestamp_berlin'].min()} -> {merged['timestamp_berlin'].max()}")
    print(f"Duplicate UTC timestamps after alignment: {merged['timestamp_utc'].duplicated().sum()}")

    print("\nMissing values in merged dataset (2017-2025 overlap):")
    missing_summary = pd.DataFrame({
        "missing_count": merged.isna().sum(),
        "missing_%": (merged.isna().mean() * 100).round(2),
    })
    print(missing_summary.to_string())

    complete_cases = merged.dropna()
    print(
        f"\nComplete rows with all variables present (from {complete_cases['timestamp_berlin'].min()} "
        f"to {complete_cases['timestamp_berlin'].max()}): {len(complete_cases):,}"
    )
    merged.to_csv(MERGED_FILE, index=False)
    print(f"Saved merged dataset to: {MERGED_FILE}")
    return merged


def summarize_target_variable(merged: pd.DataFrame) -> None:
    """Provide a detailed statistical and temporal summary of 'grid load [MWh]'."""
    print("\n" + "=" * 80)
    print(f"TARGET VARIABLE SUMMARY: {TARGET_COL}")
    print("=" * 80)

    target = merged[TARGET_COL].dropna()
    q01, q05, q25, q50, q75, q95, q99 = target.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    overall_stats = pd.Series({
        "Count": f"{len(target):,}",
        "Missing": f"{int(merged[TARGET_COL].isna().sum()):,}",
        "Mean [MWh]": f"{target.mean():.2f}",
        "Std [MWh]": f"{target.std():.2f}",
        "Min [MWh]": f"{target.min():.2f}",
        "1% Quantile [MWh]": f"{q01:.2f}",
        "5% Quantile [MWh]": f"{q05:.2f}",
        "25% Quantile [MWh]": f"{q25:.2f}",
        "Median (50%) [MWh]": f"{q50:.2f}",
        "75% Quantile [MWh]": f"{q75:.2f}",
        "95% Quantile [MWh]": f"{q95:.2f}",
        "99% Quantile [MWh]": f"{q99:.2f}",
        "Max [MWh]": f"{target.max():.2f}",
        "IQR [MWh]": f"{(q75 - q25):.2f}",
        "Skewness": f"{target.skew():.3f}",
        "Kurtosis": f"{target.kurtosis():.3f}",
    })
    print("\n--- Overall Distribution Statistics ---")
    print(overall_stats.to_string())

    # Breakdown by Year (local Berlin time)
    df_temp = merged[["timestamp_berlin", TARGET_COL, "is_holiday"]].dropna(subset=[TARGET_COL]).copy()
    df_temp["year"] = df_temp["timestamp_berlin"].dt.year
    df_temp["weekday"] = df_temp["timestamp_berlin"].dt.weekday
    df_temp["day_type"] = np.where(
        df_temp["is_holiday"] == 1,
        "Holiday (BW)",
        np.where(df_temp["weekday"] >= 5, "Weekend", "Workday"),
    )

    print("\n--- Target Summary by Year ---")
    yearly = (
        df_temp.groupby("year")[TARGET_COL]
        .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")
        .round(2)
    )
    print(yearly.to_string())

    print("\n--- Target Summary by Day Type (Workday vs. Weekend vs. BW Holiday) ---")
    by_day_type = (
        df_temp.groupby("day_type")[TARGET_COL]
        .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")
        .round(2)
    )
    print(by_day_type.to_string())


def analyze_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute and display Pearson and Spearman correlations across all merged variables."""
    print("\n" + "=" * 80)
    print("CORRELATION OVERVIEW BETWEEN VARIABLES")
    print("=" * 80)

    feature_cols = [TARGET_COL, PRICE_COL, *WEATHER_COLS, "is_holiday"]
    corr_df = merged[feature_cols].dropna()
    print(f"Correlation computed on {len(corr_df):,} complete 15-minute observations.")

    pearson_corr = corr_df.corr(method="pearson")
    spearman_corr = corr_df.corr(method="spearman")

    target_corr = pd.DataFrame({
        "Pearson_r": pearson_corr[TARGET_COL],
        "Spearman_rho": spearman_corr[TARGET_COL],
        "Abs_Pearson": pearson_corr[TARGET_COL].abs(),
    }).sort_values("Abs_Pearson", ascending=False).drop(columns=["Abs_Pearson"])

    print(f"\n--- Correlation with Target ({TARGET_COL}) ---")
    print(target_corr.round(4).to_string())

    print("\n--- Full Pearson Correlation Matrix ---")
    print(pearson_corr.round(3).to_string())

    return pearson_corr


def run_exploration() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run dataset exploration, BW holiday creation, timestamp alignment, merging, and correlation analysis."""
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

    df_holidays = create_bw_holiday_dataset()
    merged_df = align_and_merge_datasets(datasets, df_holidays)
    summarize_target_variable(merged_df)
    corr_matrix = analyze_correlations(merged_df)
    return merged_df, corr_matrix


if __name__ == "__main__":
    merged_data, correlation_matrix = run_exploration()

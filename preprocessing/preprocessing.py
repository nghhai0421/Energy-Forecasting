"""Prepare the merged dataset for the period with available prices and build 1y/5y 15min and hourly subsets."""

from datetime import timedelta, timezone
import locale
from pathlib import Path

import holidays
import pandas as pd

# Set Paths
project_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = project_start
for candidate in (project_start, *project_start.parents):
    if (candidate / "Dataset").is_dir():
        PROJECT_ROOT = candidate
        break

DATASET_DIR = PROJECT_ROOT / "Dataset"
RAW_DIR = DATASET_DIR / "raw"
PREPROCESSED_DIR = DATASET_DIR / "preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CONSUMPTION_FILE = RAW_DIR / "Actual_consumption_201701010000_202601010000_Quarterhour.xlsx"
PRICES_FILE = RAW_DIR / "Day-ahead_prices_201701010000_202601010000_Quarterhour.xlsx"
WEATHER_FILE = RAW_DIR / "open-meteo-48.83N9.15E293m.xlsx"
HOLIDAY_FILE = RAW_DIR / "df_holiday.xlsx"

MERGED_FILE = PREPROCESSED_DIR / "merged_dataset.csv"

PREPROCESSED_1Y_15MIN_FILE = PREPROCESSED_DIR / "preprocessed_dataset_1y_15min.csv"
PREPROCESSED_1Y_HOURLY_FILE = PREPROCESSED_DIR / "preprocessed_dataset_1y_hourly.csv"

PREPROCESSED_5Y_15MIN_FILE = PREPROCESSED_DIR / "preprocessed_dataset_5y_15min.csv"
PREPROCESSED_5Y_HOURLY_FILE = PREPROCESSED_DIR / "preprocessed_dataset_5y_hourly.csv"

# Set Timeframe Constants (Europe/Berlin calendar years)
FIVE_YEAR_RANGE = (2019, 2023)
ONE_YEAR = 2023

# Set Columns
TARGET_COL = "grid load [MWh]"
PRICE_COL = "Germany/Luxembourg [€/MWh]"
WEATHER_CODE_COL = "weather_code (wmo code)"
WEATHER_COLS = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "shortwave_radiation (W/m²)",
    "wind_speed_10m (km/h)",
    "dew_point_2m (°C)",
    WEATHER_CODE_COL,
]
TIMESTAMP_COLS = ["timestamp_utc", "timestamp_berlin"]


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse timestamps consistently across datetime, Excel serial, or English string formats."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="D", origin="1899-12-30").dt.round("min")
    # Use C locale so English month names ('Mar', 'May', 'Oct') and 'AM'/'PM' parse on German Windows
    locale.setlocale(locale.LC_TIME, "C")
    return pd.to_datetime(series, format="%b %d, %Y %I:%M %p", errors="coerce")


def build_merged_dataset_from_raw(save_path: Path = MERGED_FILE) -> pd.DataFrame:
    """Build and save the merged 15-minute dataset from raw Excel files (matching exploration.ipynb)."""
    print("Loading raw datasets from:", RAW_DIR)
    raw_consumption = pd.read_excel(CONSUMPTION_FILE, skiprows=9)
    raw_prices = pd.read_excel(PRICES_FILE, skiprows=9)
    raw_weather = pd.read_excel(WEATHER_FILE, skiprows=3)

    # Generate & save Baden-Württemberg (BW) holiday table
    bw_holidays = holidays.Germany(years=range(2017, 2027), subdiv="BW")
    df_holidays = (
        pd.DataFrame(list(bw_holidays.items()), columns=["Datum", "Feiertag"])
        .sort_values("Datum")
        .reset_index(drop=True)
    )
    df_holidays["Datum"] = pd.to_datetime(df_holidays["Datum"])
    df_holidays.to_excel(HOLIDAY_FILE, index=False)

    # 1. Consumption
    cons_berlin = parse_timestamp_series(raw_consumption["Start date"]).dt.tz_localize(
        "Europe/Berlin", ambiguous="infer"
    )
    cons_df = pd.DataFrame({
        "timestamp_utc": cons_berlin.dt.tz_convert("UTC"),
        "timestamp_berlin": cons_berlin,
        TARGET_COL: pd.to_numeric(raw_consumption[TARGET_COL], errors="coerce"),
    })

    # 2. Day-Ahead Prices (repeat hourly value up to 4 quarter-hours)
    prices_berlin = parse_timestamp_series(raw_prices["Start date"]).dt.tz_localize(
        "Europe/Berlin", ambiguous="infer"
    )
    price_series = pd.to_numeric(raw_prices[PRICE_COL], errors="coerce").ffill(limit=3)
    prices_df = pd.DataFrame({
        "timestamp_utc": prices_berlin.dt.tz_convert("UTC"),
        PRICE_COL: price_series,
    })

    # 3. Open-Meteo Weather (fixed UTC+2 offset)
    fixed_utc_plus_2 = timezone(timedelta(hours=2))
    weather_ts_utc = (
        parse_timestamp_series(raw_weather["time"])
        .dt.tz_localize(fixed_utc_plus_2)
        .dt.tz_convert("UTC")
    )
    weather_df = pd.DataFrame({"timestamp_utc": weather_ts_utc})
    for col in WEATHER_COLS:
        weather_df[col] = pd.to_numeric(raw_weather[col], errors="coerce")

    # 4. Merge on UTC timestamp & add BW holiday indicator
    merged_df = (
        cons_df.merge(prices_df, on="timestamp_utc", how="inner")
        .merge(weather_df, on="timestamp_utc", how="inner")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    holiday_dates = set(df_holidays["Datum"].dt.date)
    merged_df["is_holiday"] = merged_df["timestamp_berlin"].dt.date.isin(holiday_dates).astype(int)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(save_path, index=False)
    print(f"Saved merged dataset ({merged_df.shape[0]:,} rows x {merged_df.shape[1]} cols) to: {save_path}")
    return merged_df


def load_merged_dataset(path: Path = MERGED_FILE) -> pd.DataFrame:
    """Load a merged CSV path, generating it from raw files if absent; return timezone-aware rows."""
    if not path.is_file():
        print(f"{path} not found. Generating merged dataset from raw files...")
        df = build_merged_dataset_from_raw(save_path=path)
    else:
        print(f"Loading merged dataset from: {path}")
        df = pd.read_csv(path)

    # Parse Timestamps
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_berlin"] = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    return df


def check_na_distribution(df: pd.DataFrame, stage_label: str) -> pd.DataFrame:
    """Print missingness for timestamped rows and a stage label; return the report."""
    print(f"\nNA distribution ({stage_label})")

    # Count Missing Values
    total_rows = len(df)
    timestamps = df["timestamp_berlin"]
    summary_rows = []
    for column in df.columns:
        if column in TIMESTAMP_COLS:
            continue
        na_mask = df[column].isna()
        na_count = int(na_mask.sum())
        na_pct = (na_count / total_rows) * 100.0 if total_rows else 0.0
        if na_count > 0:
            na_timestamps = timestamps[na_mask]
            first_na = str(na_timestamps.min())
            last_na = str(na_timestamps.max())
            # Count NA Blocks
            block_starts = na_mask & ~na_mask.shift(1, fill_value=False)
            na_blocks = int(block_starts.sum())
        else:
            first_na = "None"
            last_na = "None"
            na_blocks = 0
        summary_rows.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "na_count": na_count,
            "na_%": round(na_pct, 4),
            "contiguous_na_blocks": na_blocks,
            "first_na_berlin": first_na,
            "last_na_berlin": last_na,
        })

    # Print Report
    report = pd.DataFrame(summary_rows)
    print(report.to_string(index=False))
    any_na_mask = df.drop(columns=TIMESTAMP_COLS).isna().any(axis=1)
    print(f"\nTotal rows with at least one NA: {int(any_na_mask.sum()):,} / {total_rows:,}")
    return report


def impute_prior_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric NAs using the causal mean of all prior values; leave weather_code as-is."""
    out = df.copy()
    numeric_cols = [
        col for col in out.columns
        if col not in (*TIMESTAMP_COLS, WEATHER_CODE_COL, "is_holiday")
    ]
    for col in numeric_cols:
        na_count = int(out[col].isna().sum())
        if na_count > 0:
            prior_mean = out[col].expanding(min_periods=1).mean().shift(1)
            out[col] = out[col].fillna(prior_mean)
            print(f"Imputed {na_count:,} NAs in '{col}' using causal prior mean.")
    return out


def resample_to_hourly(df_15min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a 15-minute preprocessed DataFrame to hourly ('1h') resolution on UTC timestamps."""
    categories = df_15min[WEATHER_CODE_COL].cat.categories if hasattr(df_15min[WEATHER_CODE_COL], "cat") else None

    def _mode_or_first(series: pd.Series):
        valid = series.dropna()
        if valid.empty:
            return pd.NA
        modes = valid.mode()
        return modes.iloc[0] if not modes.empty else valid.iloc[0]

    agg_spec = {
        TARGET_COL: "sum",
        PRICE_COL: "mean",
        "temperature_2m (°C)": "mean",
        "relative_humidity_2m (%)": "mean",
        "precipitation (mm)": "sum",
        "shortwave_radiation (W/m²)": "mean",
        "wind_speed_10m (km/h)": "mean",
        "dew_point_2m (°C)": "mean",
        WEATHER_CODE_COL: _mode_or_first,
        "is_holiday": "max",
    }

    hourly = (
        df_15min.set_index("timestamp_utc")
        .resample("1h")
        .agg(agg_spec)
        .reset_index()
    )
    hourly.insert(1, "timestamp_berlin", hourly["timestamp_utc"].dt.tz_convert("Europe/Berlin"))

    # Restore categorical dtype for weather_code (wmo code)
    codes = pd.to_numeric(hourly[WEATHER_CODE_COL], errors="coerce").round().astype("Int64")
    if categories is not None:
        hourly[WEATHER_CODE_COL] = pd.Categorical(codes, categories=categories)
    else:
        hourly[WEATHER_CODE_COL] = codes.astype("category")

    hourly["is_holiday"] = hourly["is_holiday"].astype(int)
    return hourly


def slice_berlin_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Return rows whose Europe/Berlin timestamp falls within [start_year, end_year] inclusive."""
    years = df["timestamp_berlin"].dt.year
    mask = (years >= start_year) & (years <= end_year)
    return df.loc[mask].copy().reset_index(drop=True)


def run_preprocessing() -> dict[str, pd.DataFrame]:
    """Read the merged CSV, preprocess price-period rows, and save 1-year and 5-year (15min & hourly) datasets."""
    # Load Data
    df = load_merged_dataset()
    check_na_distribution(df, stage_label="Before Chopping to Price Timeframe")

    # Trim Price Period
    valid_prices = df[PRICE_COL].notna()
    if not valid_prices.any():
        raise ValueError(f"Column '{PRICE_COL}' has no valid observations.")
    start = df.loc[valid_prices, "timestamp_utc"].min()
    end = df.loc[valid_prices, "timestamp_utc"].max()
    in_period = (df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)
    print(f"\nRows before chopping: {len(df):,}")
    df = df.loc[in_period].copy().reset_index(drop=True)
    print(f"Rows after chopping:  {len(df):,}")
    print(f"Chopped Berlin range: {df['timestamp_berlin'].min()} -> {df['timestamp_berlin'].max()}")

    # Convert Weather Codes
    # Nullable integers preserve missing codes before categorical conversion.
    codes = pd.to_numeric(df[WEATHER_CODE_COL], errors="coerce").round().astype("Int64")
    df[WEATHER_CODE_COL] = codes.astype("category")
    print(f"New dtype for '{WEATHER_CODE_COL}': {df[WEATHER_CODE_COL].dtype}")
    categories = list(df[WEATHER_CODE_COL].cat.categories)
    print(f"Categories ({len(categories)}): {categories}")

    # Check Missing Values & Impute Prior Mean
    check_na_distribution(df, stage_label="After Chopping to Price Timeframe")
    df = impute_prior_mean(df)

    # Build Hourly Dataset from 15-Minute Preprocessed Dataset
    df_hourly = resample_to_hourly(df)

    # Slice 1-Year and 5-Year (2019-2023) Subsets for 15-min and Hourly Resolutions
    start_5y, end_5y = FIVE_YEAR_RANGE
    df_5y_15min = slice_berlin_years(df, start_5y, end_5y)
    df_5y_hourly = slice_berlin_years(df_hourly, start_5y, end_5y)

    df_1y_15min = slice_berlin_years(df, ONE_YEAR, ONE_YEAR)
    df_1y_hourly = slice_berlin_years(df_hourly, ONE_YEAR, ONE_YEAR)

    # Save Datasets
    outputs = [
        (f"5 Years {start_5y}-{end_5y} (15min)", df_5y_15min, PREPROCESSED_5Y_15MIN_FILE),
        (f"5 Years {start_5y}-{end_5y} (hourly)", df_5y_hourly, PREPROCESSED_5Y_HOURLY_FILE),
        (f"1 Year {ONE_YEAR} (15min)", df_1y_15min, PREPROCESSED_1Y_15MIN_FILE),
        (f"1 Year {ONE_YEAR} (hourly)", df_1y_hourly, PREPROCESSED_1Y_HOURLY_FILE),
    ]

    print("\nSaved Preprocessed Datasets:")
    for label, subset_df, path in outputs:
        subset_df.to_csv(path, index=False)
        berlin_min = subset_df["timestamp_berlin"].min()
        berlin_max = subset_df["timestamp_berlin"].max()
        print(f"  - {label:<28}: {len(subset_df):>7,} rows | {berlin_min} -> {berlin_max} | {path.name}")

    return {
        "5y_15min": df_5y_15min,
        "5y_hourly": df_5y_hourly,
        "1y_15min": df_1y_15min,
        "1y_hourly": df_1y_hourly,
    }


if __name__ == "__main__":
    datasets = run_preprocessing()

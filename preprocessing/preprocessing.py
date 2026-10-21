"""Prepare the merged dataset for the period with available prices."""

from pathlib import Path

import pandas as pd

# Set Paths
project_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = project_start
for candidate in (project_start, *project_start.parents):
    if (candidate / "Dataset").is_dir():
        PROJECT_ROOT = candidate
        break
DATASET_DIR = PROJECT_ROOT / "Dataset"
MERGED_FILE = DATASET_DIR / "merged_dataset.csv"
PREPROCESSED_FILE = DATASET_DIR / "preprocessed_dataset.csv"

# Set Columns
TARGET_COL = "grid load [MWh]"
PRICE_COL = "Germany/Luxembourg [€/MWh]"
WEATHER_CODE_COL = "weather_code (wmo code)"
TIMESTAMP_COLS = ["timestamp_utc", "timestamp_berlin"]


def load_merged_dataset(path: Path = MERGED_FILE) -> pd.DataFrame:
    """Load a merged CSV path, generating it if absent; return timezone-aware rows."""
    # Load Data
    if not path.is_file():
        print(f"{path} not found. Generating merged dataset via exploration.py...")
        import sys

        prep_dir = str(PROJECT_ROOT / "preprocessing")
        if prep_dir not in sys.path:
            sys.path.insert(0, prep_dir)
        from exploration import (
            align_and_merge_datasets,
            create_bw_holiday_dataset,
            load_raw_datasets,
        )

        raw_datasets = load_raw_datasets()
        df_holidays = create_bw_holiday_dataset()
        df = align_and_merge_datasets(raw_datasets, df_holidays)
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


def run_preprocessing() -> pd.DataFrame:
    """Read the merged CSV; save and return price-period rows with categorical weather."""
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
    print(f"Rows before chopping: {len(df):,}")
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

    # Check Missing Values
    check_na_distribution(df, stage_label="After Chopping to Price Timeframe")

    # Save Data
    df.to_csv(PREPROCESSED_FILE, index=False)
    print(f"\nSaved preprocessed dataset ({len(df):,} rows) to: {PREPROCESSED_FILE}")
    return df


if __name__ == "__main__":
    preprocessed_df = run_preprocessing()

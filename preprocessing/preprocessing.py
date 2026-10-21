"""Preprocess the merged energy forecasting dataset.

Steps performed:
1. Load the merged dataset (Dataset/merged_dataset.csv)
2. Chop the dataframe to the timeframe where day-ahead price data ('Germany/Luxembourg [€/MWh]') is available
3. Convert 'weather_code (wmo code)' to categorical dtype
4. Inspect how NAs are distributed across columns and timestamps
5. Save the preprocessed dataset to Dataset/preprocessed_dataset.csv
"""

from pathlib import Path

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

MERGED_FILE = DATASET_DIR / "merged_dataset.csv"
PREPROCESSED_FILE = DATASET_DIR / "preprocessed_dataset.csv"

TARGET_COL = "grid load [MWh]"
PRICE_COL = "Germany/Luxembourg [€/MWh]"
WEATHER_CODE_COL = "weather_code (wmo code)"
TIMESTAMP_COLS = ["timestamp_utc", "timestamp_berlin"]


def load_merged_dataset(path: Path = MERGED_FILE) -> pd.DataFrame:
    """Load the saved merged dataset, or generate it if not yet saved."""
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

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_berlin"] = (
        df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    )
    return df


def check_na_distribution(df: pd.DataFrame, stage_label: str) -> pd.DataFrame:
    """Report how NAs are distributed across columns and timestamps."""
    print("\n" + "=" * 80)
    print(f"NA DISTRIBUTION CHECK ({stage_label})")
    print("=" * 80)

    total_rows = len(df)
    ts = df["timestamp_berlin"]
    summary_rows = []

    for col in df.columns:
        if col in TIMESTAMP_COLS:
            continue
        na_mask = df[col].isna()
        na_count = int(na_mask.sum())
        na_pct = (na_count / total_rows) * 100.0 if total_rows else 0.0

        if na_count > 0:
            na_ts = ts[na_mask]
            first_na = str(na_ts.min())
            last_na = str(na_ts.max())
            # Count contiguous NA blocks
            na_blocks = int((na_mask & ~na_mask.shift(1, fill_value=False)).sum())
        else:
            first_na = "None"
            last_na = "None"
            na_blocks = 0

        summary_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "na_count": na_count,
            "na_%": round(na_pct, 4),
            "contiguous_na_blocks": na_blocks,
            "first_na_berlin": first_na,
            "last_na_berlin": last_na,
        })

    report = pd.DataFrame(summary_rows)
    print(report.to_string(index=False))

    any_na_mask = df.drop(columns=TIMESTAMP_COLS).isna().any(axis=1)
    print(f"\nTotal rows with at least one NA: {int(any_na_mask.sum()):,} / {total_rows:,}")
    return report


def chop_to_price_timeframe(df: pd.DataFrame) -> pd.DataFrame:
    """Chop dataframe to the timeframe where day-ahead price data is available."""
    print("\n" + "=" * 80)
    print(f"CHOPPING TO PRICE TIMEFRAME ({PRICE_COL})")
    print("=" * 80)

    valid_price_mask = df[PRICE_COL].notna()
    if not valid_price_mask.any():
        raise ValueError(f"Column '{PRICE_COL}' has no valid observations.")

    start_ts = df.loc[valid_price_mask, "timestamp_utc"].min()
    end_ts = df.loc[valid_price_mask, "timestamp_utc"].max()

    chopped = (
        df[(df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts)]
        .copy()
        .reset_index(drop=True)
    )
    print(f"Rows before chopping: {len(df):,}")
    print(f"Rows after chopping:  {len(chopped):,}")
    print(
        f"Chopped Berlin range: {chopped['timestamp_berlin'].min()} -> "
        f"{chopped['timestamp_berlin'].max()}"
    )
    return chopped


def convert_weather_code_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'weather_code (wmo code)' to categorical dtype while preserving NAs."""
    print("\n" + "=" * 80)
    print(f"CONVERTING '{WEATHER_CODE_COL}' TO CATEGORICAL")
    print("=" * 80)

    out = df.copy()
    # Use nullable Int64 first so WMO codes become clean integer categories (e.g. 0, 1, 2, 3...)
    wmo_int = pd.to_numeric(out[WEATHER_CODE_COL], errors="coerce").round().astype("Int64")
    out[WEATHER_CODE_COL] = wmo_int.astype("category")

    print(f"New dtype for '{WEATHER_CODE_COL}': {out[WEATHER_CODE_COL].dtype}")
    print(f"Categories ({len(out[WEATHER_CODE_COL].cat.categories)}): "
          f"{list(out[WEATHER_CODE_COL].cat.categories)}")
    return out


def run_preprocessing() -> pd.DataFrame:
    """Execute the full preprocessing pipeline and save the preprocessed dataset."""
    df = load_merged_dataset()

    # 1. Check NA distribution in the full merged dataset before chopping
    check_na_distribution(df, stage_label="Before Chopping to Price Timeframe")

    # 2. Chop dataframe to the timeframe where day-ahead price data is available
    df = chop_to_price_timeframe(df)

    # 3. Change weather_code (wmo code) to categorical
    df = convert_weather_code_to_categorical(df)

    # 4. Check how NAs are distributed in the price-timeframe dataset
    check_na_distribution(df, stage_label="After Chopping to Price Timeframe")

    # 5. Save preprocessed dataframe
    df.to_csv(PREPROCESSED_FILE, index=False)
    print(f"\nSaved preprocessed dataset ({len(df):,} rows) to: {PREPROCESSED_FILE}")
    return df


if __name__ == "__main__":
    preprocessed_df = run_preprocessing()

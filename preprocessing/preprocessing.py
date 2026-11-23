"""Preprocess the 2023-2024 and 2025 consumption datasets by interpolating missing values in Residual load [MWh]."""

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
RAW_CSV_DIR = DATASET_DIR / "raw" / "csv"
RAW_XLSX_DIR = DATASET_DIR / "raw" / "xlsx"
PREPROCESSED_DIR = DATASET_DIR / "preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CONSUMPTION_2023_2024_NAME = "Actual_consumption_2023_2024_Quarterhour.csv"
CONSUMPTION_2025_NAME = "Actual_consumption_2025_Quarterhour.csv"

PREPROCESSED_2023_2024_FILE = PREPROCESSED_DIR / CONSUMPTION_2023_2024_NAME
PREPROCESSED_2025_FILE = PREPROCESSED_DIR / CONSUMPTION_2025_NAME

TARGET_COL = "Residual load [MWh]"
NUMERIC_COLS = [
    "grid load [MWh]",
    "Grid load incl. hydro pumped storage [MWh]",
    "Hydro pumped storage [MWh]",
    TARGET_COL,
]


def load_raw_consumption(filename: str) -> pd.DataFrame:
    """Load a raw consumption dataset from CSV (or Excel fallback if CSV is absent).

    Parameters
    ----------
    filename : str
        CSV filename in Dataset/raw/csv/.

    Returns
    -------
    pandas.DataFrame
        Raw consumption dataframe.
    """
    csv_path = RAW_CSV_DIR / filename
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    xlsx_path = RAW_XLSX_DIR / Path(filename).with_suffix(".xlsx")
    if xlsx_path.is_file():
        return pd.read_excel(xlsx_path, skiprows=9)
    raise FileNotFoundError(f"Could not find {csv_path} or {xlsx_path}")


def preprocess_consumption_dataframe(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Coerce numeric columns and interpolate missing values in Residual load [MWh].

    Parameters
    ----------
    df : pandas.DataFrame
        Raw consumption rows for a given timeframe.
    label : str
        Dataset timeframe label for console logging.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with missing values in Residual load [MWh] interpolated.
    """
    out = df.copy()
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    na_mask = out[TARGET_COL].isna()
    na_count = int(na_mask.sum())
    print(f"\n[{label}] Total rows: {len(out):,} | Missing in '{TARGET_COL}': {na_count}")

    if na_count > 0:
        missing_indices = out.index[na_mask].tolist()
        out[TARGET_COL] = out[TARGET_COL].interpolate(method="linear").bfill().ffill()
        for idx in missing_indices:
            ts = out.loc[idx, "Start date"] if "Start date" in out.columns else idx
            val = out.loc[idx, TARGET_COL]
            print(f"  - Interpolated row index {idx} ({ts}): '{df.loc[idx, TARGET_COL]}' -> {val}")

    remaining_na = int(out[TARGET_COL].isna().sum())
    print(f"[{label}] Remaining missing values in '{TARGET_COL}': {remaining_na}")
    return out


def run_preprocessing() -> dict[str, pd.DataFrame]:
    """Preprocess and save the 2023-2024 and 2025 consumption datasets to Dataset/preprocessed/.

    Returns
    -------
    dict of str to pandas.DataFrame
        Preprocessed '2023_2024' and '2025' consumption dataframes.
    """
    specs = [
        ("2023_2024", "2023-2024", CONSUMPTION_2023_2024_NAME, PREPROCESSED_2023_2024_FILE),
        ("2025", "2025", CONSUMPTION_2025_NAME, PREPROCESSED_2025_FILE),
    ]

    results = {}
    for key, label, raw_name, save_path in specs:
        raw_df = load_raw_consumption(raw_name)
        clean_df = preprocess_consumption_dataframe(raw_df, label=label)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Saved preprocessed {label} dataset ({len(clean_df):,} rows) to: {save_path}")
        results[key] = clean_df

    return results


if __name__ == "__main__":
    datasets = run_preprocessing()

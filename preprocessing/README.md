# Preprocessing

Use the project environment in [environment.yml](../environment.yml). Run Python examples from the repository root.

| File | Purpose and inputs | Outputs |
| --- | --- | --- |
| [split_raw_datasets.ipynb](split_raw_datasets.ipynb) | Split full-period workbooks in `Dataset/raw/xlsx/`; generate BW holidays using `holidays`. | Six subsets for 2023-2024 and Jan-Nov 2025, plus holidays for 2023-2025, in the same folder. Existing workbooks are overwritten. |
| [convert_raw_to_csv.py](convert_raw_to_csv.py) | Convert the six subset workbooks and holiday workbook; remove metadata preambles. | Matching filenames in `Dataset/raw/csv/`. Existing CSVs are skipped. |
| [add_seasonal_features.py](add_seasonal_features.py) | Generate calendar features from a timestamp column. | A DataFrame copy with eight sine/cosine pairs and year immediately after the timestamp column. |
| [exploration.ipynb](exploration.ipynb) | Inspect missing values, duplicates, timestamps, and relationships in raw data. | Notebook tables and plots. |
| [preprocessing.py](preprocessing.py) | Load the merged CSV, building it from raw Excel if absent; impute using prior observations and aggregate hourly. | Merged data and 1-year (2023) / 5-year (2019-2023) variants at 15-minute and hourly resolutions in `Dataset/preprocessed/`. |

## Usage

Run notebook cells from top to bottom. Python entry points:

```python
from preprocessing.convert_raw_to_csv import convert_raw_to_csv
from preprocessing.preprocessing import run_preprocessing

counts = convert_raw_to_csv()
datasets = run_preprocessing()
```

Calendar features work independently during training and inference:

```python
import pandas as pd
from preprocessing.add_seasonal_features import add_seasonal_features

df = pd.DataFrame({"timestamp_utc": ["2025-01-01 00:00:00"]})
featured = add_seasonal_features(df, "timestamp_utc", input_timezone="UTC")
```

## Key rules

- Calendar features default to `timezone="Europe/Berlin"`. Naive timestamps require `input_timezone`; normalize mixed offsets to UTC first. Invalid/missing timestamps, ambiguous DST times, and column collisions raise errors.
- Calendar formulas follow [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series/blob/main/tabpfn_time_series/features/basic_features.py), including its period-minus-one convention. Original data and index are preserved. Holidays are joined separately.
- CSV conversion preserves values and timestamps, uses commas and UTF-8 with BOM, and skips existing files. Remove a generated CSV explicitly to refresh it. Missing inputs, unexpected sheet/header layouts, and formulas raise errors. Existing `energy_daily.csv` and `energy_hourly.csv` use semicolons.
- Local timestamps use Berlin time for SMARD and fixed UTC+2 for Open-Meteo. Normalize sources before combining; match holidays by Berlin calendar date.

See the [dataset overview](../Dataset/README.md) and [data inspection notes](../docs/data_inspection.md).

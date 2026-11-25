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

See the [dataset overview](../Dataset/README.md).

## Final split-file GBT preparation

The final experiment uses `src/models/gradient_boosting/process.py`, independently
of the other model runners. It follows Chronos-2's source timezone conventions
and causal filling policy and calls the shared `add_seasonal_features()` routine.
Development loading opens only the 2023–2024 consumption, weather and price files.
Demo loading is a separate call made after final model weights have been saved.
Missing original target labels remain missing for training/scoring, even when
historical inputs are filled from strictly preceding valid observations.

The shared holiday CSV contains only 2023–2024; the GBT loader generates absent
calendar years with `holidays.Germany(subdiv="BW")`. These dates are known future
calendar inputs and are bundled with the checkpoint. Raw files are preserved.

Two lags each for load, temperature, wind, dew point and price use effective offsets
of 25 and 169 hours (nominal 24/168 hours plus the configured one-hour availability
buffer). The 17 shared calendar encodings and holiday flag bring the input count
to 28. `outputs/runs/20261113T123354683990Z_gradient-boosting-final-grid-load-15min-evaluation_395ad9bfa94a/artifacts/development_features.csv` contains 69,500 usable training
rows after the 676-row warm-up. `preprocessing/gbt_final_demo.ipynb` demonstrates
the same preparation and frozen inference functions used by the final evaluation.

Incoming demo history must be a continuous, UTC-indexed, prepared quarter-hour
table containing the configured lag fields. Keep at least 676 rows ending just
before the forecast origin. `forecast()` discards any rows at/after the origin;
future labels, weather and prices cannot become features. Calendar inputs are
generated from delivery timestamps. Missing raw observations require the same
causal preprocessing policy and state as development; do not backfill from future
rows. The buffer does not verify historical weather publication vintages.


See the [final GBT setup and comparison](../reports/transnetbw_2025_final_comparison.md) for the full feature list and training commands.

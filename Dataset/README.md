# Dataset Overview (`Dataset/`)

This directory contains both the **raw input datasets** (`Dataset/raw/`) and the **preprocessed datasets** (`Dataset/preprocessed/`) used for energy load forecasting in Baden-Württemberg (TransnetBW control area).

All paths in this document are relative to `Dataset/` unless prefixed with
`Dataset/`. Run project commands from the repository root.

## TransnetBW comparison inputs

The supplied SMARD benchmark exports are stored in `benchmarks/transnetbw/`:

| Repository-relative path | Contents |
| --- | --- |
| `Dataset/benchmarks/transnetbw/actual_consumption.csv` | Actual grid-load export for 2021–2025 |
| `Dataset/benchmarks/transnetbw/forecasted_consumption.csv` | Published forecast export for 2021–2025 |

These are unchanged copies of the supplied English, semicolon-separated SMARD
quarter-hour exports. Both use `grid load [MWh] Original resolutions` and Berlin
local timestamps. The TransnetBW region comes from the download selection.
The comparison selects only delivery intervals covered by the model predictions.
Keep these files in the documented location when sharing the checkout; no Downloads
folder or personal username is needed. Input hashes are recorded in the generated
comparison summary under `outputs/runs/20261113T123354683990Z_gradient-boosting-final-grid-load-15min-evaluation_395ad9bfa94a/metrics/`.

---

## 1. Raw Datasets (`Dataset/raw/`)

Excel sources live in `raw/xlsx/`. CSV files live in `raw/csv/`.
`preprocessing/convert_raw_to_csv.py` exports the six 2023?2024 and 2025
consumption, price, and weather tables plus `df_holiday.xlsx`, using the original
filenames with `.csv`. Existing CSV exports are skipped.
Exports omit the metadata preambles. The Excel sources retain the metadata,
including Open-Meteo's fixed UTC+2 offset. Existing energy CSVs retain their
semicolon delimiter; the six new exports use commas and UTF-8 with BOM.
See the [conversion guide](../preprocessing/README.md#usage).

### 1.1 Actual Electricity Consumption (SMARD)
- **Source**: [SMARD Market Data Download Center](https://www.smard.de/en/downloadcenter/download-market-data/)
- **Region**: TransnetBW, Baden-Württemberg
- **Resolution**: 15 minutes (`Quarterhour`), local time (`Europe/Berlin`)
- **Key Variables**:
  - `grid load [MWh]` *(Target variable)*
  - Additional columns in raw file: `Start date`, `End date`, `Grid load incl. hydro pumped storage [MWh]`, `Hydro pumped storage [MWh]`, `Residual load [MWh]`
- **Files**:
  - `raw/xlsx/Actual_consumption_201701010000_202601010000_Quarterhour.xlsx`: Full period (`01.01.201` – `31.12.2025`)
  - `raw/xlsx/Actual_consumption_2023_2024_Quarterhour.xlsx`: 2023–2024 split (`01.01.2023` – `31.12.2024`)
  - `raw/xlsx/Actual_consumption_2025_Quarterhour.xlsx`: 2025 Jan–Nov split (`01.01.2025` – `30.11.2025`)

### 1.2 Day-Ahead Electricity Prices (SMARD)
- **Source**: [SMARD Market Data Download Center](https://www.smard.de/en/downloadcenter/download-market-data/)
- **Region**: Germany / Luxembourg (`DE/LU`) 
- **Resolution**: 15-minute grid (`Quarterhour`, 4 times original resolution `60min`), local time (`Europe/Berlin`)
- **Key Variables**:
  - `Germany/Luxembourg [€/MWh]` *(available from `01.10.2018` – `31.12.2025`)*
- **Files**:
  - `raw/xlsx/Day-ahead_prices_201701010000_202601010000_Quarterhour.xlsx`: Full period (`01.01.2017` – `31.12.2025`)
  - `raw/xlsx/Day-ahead_prices_2023_2024_Quarterhour.xlsx`: 2023–2024 split (`01.01.2023` – `31.12.2024`)
  - `raw/xlsx/Day-ahead_prices_2025_Quarterhour.xlsx`: 2025 Jan–Nov split (`01.01.2025` – `30.11.2025`)

### 1.3 Historical Weather Data (Open-Meteo)
- **Source**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Region**: Stuttgart
- **Resolution**: 15 minutes (`UTC+2` fixed offset)
- **Key Variables**:
  - `temperature_2m (°C)`: Temperature 2 meters above ground
  - `relative_humidity_2m (%)`: Relative humidity 2 meters above ground
  - `precipitation (mm)`: Total precipitation (rain, showers, snow) sum of the preceding 15 minutes
  - `shortwave_radiation (W/m²)`: Shortwave solar radiation as average of the preceding 15 minutes (solar panels primarily work with shortwave radiation)
  - `wind_speed_10m (km/h)`: Wind speed at 10 meters above ground (standard level)
  - `dew_point_2m (°C)`: Dew point temperature at 2 meters above ground
  - `weather_code (wmo code)`:
    - `"0"`: `"Clear sky"`
    - `"1"`: `"Mainly clear"`
    - `"2"`: `"Partly cloudy"`
    - `"3"`: `"Overcast"`
    - `"45"`: `"Fog"`
    - `"48"`: `"Depositing rime fog"`
    - `"51"`: `"Light drizzle"`
    - `"53"`: `"Moderate drizzle"`
    - `"55"`: `"Dense drizzle"`
    - `"56"`: `"Light freezing drizzle"`
    - `"57"`: `"Dense freezing drizzle"`
    - `"61"`: `"Slight rain"`
    - `"63"`: `"Moderate rain"`
    - `"65"`: `"Heavy rain"`
    - `"66"`: `"Light freezing rain"`
    - `"67"`: `"Heavy freezing rain"`
    - `"71"`: `"Slight snowfall"`
    - `"73"`: `"Moderate snowfall"`
    - `"75"`: `"Heavy snowfall"`
    - `"77"`: `"Snow grains"`
    - `"80"`: `"Slight rain showers"`
    - `"81"`: `"Moderate rain showers"`
    - `"82"`: `"Violent rain showers"`
    - `"85"`: `"Slight snow showers"`
    - `"86"`: `"Heavy snow showers"`
    - `"95"`: `"Thunderstorm"`
    - `"96"`: `"Thunderstorm with slight hail"`
    - `"97"`: `"Heavy thunderstorm"`
    - `"99"`: `"Thunderstorm with heavy hail"`
  - Additional columns in raw file: `direct_radiation (W/m²)`, `sunshine_duration (s)`
- **Files**:
  - `raw/xlsx/open-meteo-48.83N9.15E293m.xlsx`: Full period (`2017-01-01` – `2025-12-13`)
  - `raw/xlsx/open-meteo-48.83N9.15E293m_2023_2024.xlsx`: 2023–2024 split (`2023-01-01` – `2024-12-31`)
  - `raw/xlsx/open-meteo-48.83N9.15E293m_2025.xlsx`: 2025 Jan–Nov split (`2025-01-01` – `2025-11-30`)

### 1.4 Public Holiday Data (Baden-Württemberg)
- **File**: `raw/xlsx/df_holiday.xlsx`
- **Source**: Generated in [`preprocessing/split_raw_datasets.ipynb`](../preprocessing/split_raw_datasets.ipynb) via Python `holidays` library (`holidays.Germany(years=range(2023, 2026), subdiv="BW")`)
- **Region**: Baden-Württemberg (`BW`)
- **Columns & Variables**:
  - Raw columns: `Datum` (date), `Feiertag` (holiday name)
  - Mapped to `is_holiday` (`0` = regular day, `1` = BW public holiday) in the merged/preprocessed datasets

### 1.5 Additional SMARD Calculated-Resolution CSV Datasets (2019–2024)
Semicolon-delimited (`;`) SMARD exports covering `01.01.2019` – `01.01.2025`
- `raw/csv/energy_hourly.csv`: Hourly resolution (`52,608` rows)
- `raw/csv/energy_daily.csv`: Daily resolution (`2,192` rows)

### 1.6 Dataset Splitting & Holiday Generation Notebook
- **File**: [`preprocessing/split_raw_datasets.ipynb`](../preprocessing/split_raw_datasets.ipynb)
- **Purpose**: Splits the 3 full-period raw Excel datasets into the 6 subsets for **`2023–2024`** (`70,176` rows each) and **`2025 (Jan–Nov)`** (`32,064` rows each) while preserving their original metadata preamble rows (`skiprows=9` for SMARD, `skiprows=3` for Open-Meteo), and generates `raw/xlsx/df_holiday.xlsx`.

---

## 2. Preprocessed Datasets (`Dataset/preprocessed/`)
Generated by `preprocessing/preprocessing.py`:
- `preprocessed/merged_dataset.csv`: Full merged 15-minute dataset (`2017-01-01` – `2025-12-13`, `313,820` rows)
- `preprocessed/preprocessed_dataset_5y_15min.csv`: 5-year (`2019–2023`) 15-minute dataset (`175,296` rows)
- `preprocessed/preprocessed_dataset_5y_hourly.csv`: 5-year (`2019–2023`) hourly dataset (`43,824` rows)
- `preprocessed/preprocessed_dataset_1y_15min.csv`: 1-year (`2023`) 15-minute dataset (`35,040` rows)
- `preprocessed/preprocessed_dataset_1y_hourly.csv`: 1-year (`2023`) hourly dataset (`8,760` rows)
- `preprocessed/preprocessed_dataset_hourly.csv`: 2-year (`2023–2024`) hourly dataset (`17,544` rows)

## Final GBT development table

`outputs/runs/20261113T123354683990Z_gradient-boosting-final-grid-load-15min-evaluation_395ad9bfa94a/artifacts/development_features.csv` is generated from the separate 2023–2024
raw CSVs by `configs/experiments/gradient_boosting.yaml`. It contains 28 input
features, the original `grid load [MWh]` target and a UTC interval-start timestamp.
There are 69,500 usable rows after 676 rows of lag context. It contains no 2025
observations. The 2025 source files are loaded separately for the frozen-model demo
and cover January 1 through November 30 (32,064 quarter-hours).

The checked-in `raw/csv/df_holiday.csv` currently contains only 2023–2024 holidays,
despite the broader generation range described above. Final GBT preparation adds
known BW calendar dates for missing years in memory and stores them in the model
checkpoint. It does not alter the supplied raw CSV.

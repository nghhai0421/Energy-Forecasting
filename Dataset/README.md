# Dataset Overview (`Dataset/`)

This directory contains both the **raw input datasets** (`Dataset/raw/`) and the **preprocessed datasets** (`Dataset/preprocessed/`) used for energy load forecasting in Baden-Württemberg (TransnetBW control area).

---

## 1. Raw Datasets (`Dataset/raw/`)

### 1.1 Actual Electricity Consumption (SMARD)
- **Source**: [SMARD Market Data Download Center](https://www.smard.de/en/downloadcenter/download-market-data/)
- **Region**: TransnetBW, Baden-Württemberg
- **Resolution**: 15 minutes (`Quarterhour`), local time (`Europe/Berlin`)
- **Key Variables**:
  - `grid load [MWh]` *(Target variable)*
  - Additional columns in raw file: `Start date`, `End date`, `Grid load incl. hydro pumped storage [MWh]`, `Hydro pumped storage [MWh]`, `Residual load [MWh]`
- **Files**:
  - `raw/Actual_consumption_201701010000_202601010000_Quarterhour.xlsx`: Full period (`01.01.201` – `31.12.2025`)
  - `raw/Actual_consumption_2023_2024_Quarterhour.xlsx`: 2023–2024 split (`01.01.2023` – `31.12.2024`)
  - `raw/Actual_consumption_2025_Quarterhour.xlsx`: 2025 Jan–Nov split (`01.01.2025` – `30.11.2025`)

### 1.2 Day-Ahead Electricity Prices (SMARD)
- **Source**: [SMARD Market Data Download Center](https://www.smard.de/en/downloadcenter/download-market-data/)
- **Region**: Germany / Luxembourg (`DE/LU`) 
- **Resolution**: 15-minute grid (`Quarterhour`, 4 times original resolution `60min`), local time (`Europe/Berlin`)
- **Key Variables**:
  - `Germany/Luxembourg [€/MWh]` *(available from `01.10.2018` – `31.12.2025`)*
- **Files**:
  - `raw/Day-ahead_prices_201701010000_202601010000_Quarterhour.xlsx`: Full period (`01.01.2017` – `31.12.2025`)
  - `raw/Day-ahead_prices_2023_2024_Quarterhour.xlsx`: 2023–2024 split (`01.01.2023` – `31.12.2024`)
  - `raw/Day-ahead_prices_2025_Quarterhour.xlsx`: 2025 Jan–Nov split (`01.01.2025` – `30.11.2025`)

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
  - `raw/open-meteo-48.83N9.15E293m.xlsx`: Full period (`2017-01-01` – `2025-12-13`)
  - `raw/open-meteo-48.83N9.15E293m_2023_2024.xlsx`: 2023–2024 split (`2023-01-01` – `2024-12-31`)
  - `raw/open-meteo-48.83N9.15E293m_2025.xlsx`: 2025 Jan–Nov split (`2025-01-01` – `2025-11-30`)

### 1.4 Public Holiday Data (Baden-Württemberg)
- **File**: `raw/df_holiday.xlsx`
- **Source**: Generated in `raw/split_raw_datasets.ipynb` via Python `holidays` library (`holidays.Germany(years=range(2023, 2027), subdiv="BW")`)
- **Region**: Baden-Württemberg (`BW`)
- **Columns & Variables**:
  - Raw columns: `Datum` (date), `Feiertag` (holiday name)
  - Mapped to `is_holiday` (`0` = regular day, `1` = BW public holiday) in the merged/preprocessed datasets

### 1.5 Additional SMARD Calculated-Resolution CSV Datasets (2019–2024)
Semicolon-delimited (`;`) SMARD exports covering `01.01.2019` – `01.01.2025`
- `raw/energy_hourly.csv`: Hourly resolution (`52,608` rows)
- `raw/energy_daily.csv`: Daily resolution (`2,192` rows)

### 1.6 Dataset Splitting & Holiday Generation Notebook
- **File**: `raw/split_raw_datasets.ipynb`
- **Purpose**: Splits the 3 full-period raw Excel datasets into the 6 subsets for **`2023–2024`** (`70,176` rows each) and **`2025 (Jan–Nov)`** (`32,064` rows each) while preserving their original metadata preamble rows (`skiprows=9` for SMARD, `skiprows=3` for Open-Meteo), and generates `raw/df_holiday.xlsx`.

---

## 2. Preprocessed Datasets (`Dataset/preprocessed/`)
Generated by `preprocessing/preprocessing.py`:
- `preprocessed/merged_dataset.csv`: Full merged 15-minute dataset (`2017-01-01` – `2025-12-13`, `313,820` rows)
- `preprocessed/preprocessed_dataset_5y_15min.csv`: 5-year (`2019–2023`) 15-minute dataset (`175,296` rows)
- `preprocessed/preprocessed_dataset_5y_hourly.csv`: 5-year (`2019–2023`) hourly dataset (`43,824` rows)
- `preprocessed/preprocessed_dataset_1y_15min.csv`: 1-year (`2023`) 15-minute dataset (`35,040` rows)
- `preprocessed/preprocessed_dataset_1y_hourly.csv`: 1-year (`2023`) hourly dataset (`8,760` rows)
- `preprocessed/preprocessed_dataset_hourly.csv`: 2-year (`2023–2024`) hourly dataset (`17,544` rows)

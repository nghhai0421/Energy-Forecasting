# Exploration steps (`preprocessing/exploration.ipynb`)
1. Loading raw datasets from `Dataset/raw/`
2. Explore raw datasets (`Actual Consumption`, `Day-Ahead Prices`, `Open-Meteo Weather`) for:
   - Incorrect data types and non-numeric placeholder tokens (`'-'`, `'#NUM!'`)
   - Duplicates and timestamp integrity (DST summertime/wintertime transitions)
   - Missing values
   - Outliers and domain-specific impossible values
      - Notice, negative energy prices are possible (https://www.eurelectric.org/wp-content/uploads/2024/11/Eurelectric-explainer-on-negative-prices.pdf)
3. Generate Baden-Württemberg (BW) holiday dataset and save to `Dataset/raw/df_holiday.xlsx`
   - Add `is_holiday` indicator matched against `Europe/Berlin` calendar date
4. Align timestamps across all datasets (handling `Europe/Berlin` DST vs. fixed `UTC+2` in Open-Meteo)
   - timestap for us should be "timestamp_utc", "timestamp_berlin" still included to understand holiday variable
5. Merge the key variables (defined in `Dataset/README.md`) at 15-minute resolution 
   - saved to `Dataset/preprocessed/merged_dataset.csv`
   - Hourly price repeated 4 times per hour
6. Explore the target variable (`grid load [MWh]` from `Actual Consumption`) across overall quantiles and years
7. Compute and display the Pearson and Spearman correlation matrices across all variables
   - Only high correlation is `temperature_2m (°C)` and `dew_point_2m (°C)` ($r = 0.853$)

# General preprocessing steps (`preprocessing/preprocessing.py`)
1. Load `Dataset/preprocessed/merged_dataset.csv`
2. Check how NAs are distributed and slice the merged dataframe to the timeframe where day-ahead price data (`Germany/Luxembourg [€/MWh]`) is available (`2018-10-01` to `2025-12-13`)
3. Convert `weather_code (wmo code)` to `category` dtype
4. Inspect how NAs are distributed across columns and timestamps after chopping and impute numeric NAs using the causal prior mean 
   - No NAs remain from `2018-10-01` to `2025-12-13`
5. Aggregate the 15-minute preprocessed data to hourly (`1h`) resolution:
   - `sum` for `grid load [MWh]` and `precipitation (mm)`
   - `mean` for `Germany/Luxembourg [€/MWh]` and continuous weather variables
   - Hourly mode for `weather_code (wmo code)` and `max` for `is_holiday`
6. Slice and save the **1-year (`2023`)** and **5-year (`2019–2023`)** datasets for both **15-minute** and **hourly** resolutions in `Dataset/preprocessed/`:
   - `Dataset/preprocessed/preprocessed_dataset_5y_15min.csv` (`175,296` rows, `2019–2023`)
   - `Dataset/preprocessed/preprocessed_dataset_5y_hourly.csv` (`43,824` rows, `2019–2023`)
   - `Dataset/preprocessed/preprocessed_dataset_1y_15min.csv` (`35,040` rows, `2023`)
   - `Dataset/preprocessed/preprocessed_dataset_1y_hourly.csv` (`8,760` rows, `2023`)

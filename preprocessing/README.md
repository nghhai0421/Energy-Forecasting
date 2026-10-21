# Dataset preparation (planned)

Prepare reproducible variants here: univariate, target-derived features, and
exogenous features. Record raw input, transformations, feature availability and
variant version; do not transform the supplied raw files in place.

Processed rows will expose timestamp, target, and optional feature columns.
Start with one target series; choose the source target explicitly after reviewing
its meaning. Add series_id if multiple series are supported later. See
../docs/data_inspection.md for verified raw columns and timestamp caveats.

Every input feature must be available at its forecast origin. Lags and rolling
statistics must not use future target observations; realized future exogenous
values are not automatically available inputs. Frequency regularization, timezone
and daylight-saving ambiguity, missing values/hours and duplicates require an
explicit policy before folds are built. Do not silently discard repeated wall
clock hours or interpolate without documenting the decision.

Model-specific scaling, encoding and window construction belong in each model's
process.py. Fit preprocessing on that fold's training data only and reuse it for
validation/prediction. Future shared folds use the same processed row indices
across models and candidates.

# Exploration
1. Changed Excel serial date to pd.to_datetime for all raw dataframe
2. Explore datasets (Actual Consumption, Day-Ahead Prices, Open-Meteo Weather) for:
   - Incorrect data types and non-numeric placeholder tokens ('-', '#NUM!')
   - Duplicates and timestamp integrity (DST summertime/wintertime transitions)
   - Missing values
   - Outliers and domain-specific impossible values
3. Generate Baden-Württemberg (BW) holiday dataset and save to Dataset/df_holiday.xlsx
4. Align timestamps across all datasets (handling Europe/Berlin DST vs. fixed UTC+2)
5. Merge the key variables defined in Dataset/README.md at 15-minute resolution
    - Hourly price = 4 times
6. Explore the target variable ('grid load [MWh]')
7. Compute and display the correlation matrix across all merged variables
    - Only high correlation is temperature and dew_point_2m (makes sense, I guess we just drop dew_point_2m)

# Preprocessing
1. Chop the merged dataframe to the timeframe where day-ahead price data ('Germany/Luxembourg [€/MWh]') is available
2. Convert 'weather_code (wmo code)' to categorical dtype
3. Inspect how NAs are distributed across columns and timestamps
    - No NAs from 2018.10.01 - 2025.12.13

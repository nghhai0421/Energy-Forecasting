# Energy Forecasting — Evaluation Overview

## 1. Evaluation Choice and Assumptions

We assume **TransnetBW issues a rolling 24-hour residual-load forecast every hour**, following the same schedule as Chronos-2 and GBT in this project. We interpret each forecast as an estimate of residual-load requirements for future delivery intervals, relevant to generation and supply planning.

For **forecast direction and asymmetric weighted loss**, we evaluate only the prediction at **horizon step 96 (24 hours ahead)** from each hourly forecast:

- **Reason:** We want to assess how accurately each model anticipates residual load for a delivery interval at a consistent 24-hour lead time, under the assumed TransnetBW forecasting schedule.
- **Selection:** Each forecast produces 96 predictions, but only its last prediction contributes to forecast-direction metrics and asymmetric weighted loss.
- **Aggregation:** Calculate the error and loss for each selected prediction, then aggregate across forecast origins using a simple arithmetic mean where appropriate.
- **Overall accuracy:** MAE and RMSE use all 96 steps of every forecast to measure general prediction accuracy.
- **Interpretation:** Step 96 corresponds to **one 15-minute interval at the end of the 24-hour horizon**, not the sum of energy required during the entire day. Selected target intervals are one hour apart because forecasts are issued hourly.
- **TransnetBW assumptions:** Use an hourly rolling 24-hour forecasting schedule for this comparison. Assume the benchmark forecasts include a safety margin against underestimating residual-load requirements.

## 2. Forecasting Setup

- **Models:** Chronos-2 and Gradient Boosting Tree (GBT).
- **Benchmark:** TransnetBW residual-load forecasts.
- **Target:** Actual residual load.
- **Data and prediction resolution:** 15 minutes.
- **Lookback:** 672 observations (previous 7 days).
- **Forecast horizon:** 96 observations (next 24 hours).
- **Forecast step:** 4 observations (one new forecast every hour).
- **TransnetBW forecast frequency:** Assumed hourly for this evaluation.
- **Forecast output:** 96 predictions per forecast origin.
- **Evaluation period:** Two weeks in January 2025, matching the demo period; use the same actual target values for all three models.

Each forecast uses the preceding 672 observations to predict the next 96 intervals. After one hour, four new observations become available and another forecast is issued. **No hourly aggregation is performed.**

- **MAE and RMSE:** All 96 horizon steps.
- **Forecast direction and asymmetric weighted loss:** Only step 96 from each forecast.
- **Saved identifiers:** `forecast_origin`, `horizon_step`, `target_timestamp`, prediction, and actual value.

## 3. Evaluation Metrics

### A. Forecast Accuracy — All 96 Horizon Steps

**Notation used in this section:**

- **Forecast origin ( $o$ ):** The time a new 96-step forecast is issued; origins are spaced one hour apart.
- **Horizon step ( $h$ ):** Position within a forecast, from 1 to 96; each step represents 15 minutes.
- **Actual residual load ( $y_{o,h}$ ):** Actual value for the target interval at step $h$ from origin $o$.
- **Predicted residual load ( $\hat{y}_{o,h}$ ):** Forecast for the same target interval.
- **Signed error ( $e_{o,h}$ ):** Prediction minus actual; positive indicates overprediction.
- **Evaluation pairs ( $\mathcal{P}$ ):** Set of valid forecast-origin and horizon-step pairs; $N=|\mathcal{P}|$ is its size.

$$
e_{o,h}=\hat{y}_{o,h}-y_{o,h}
$$

- **Mean Absolute Error (MAE):** Average magnitude of errors across all 96 forecast steps. Lower is better.

$$
\mathrm{MAE}=\frac{1}{N}\sum_{(o,h)\in\mathcal{P}}|e_{o,h}|
$$

- **Root Mean Squared Error (RMSE):** Square root of the mean squared error. Larger errors have a greater effect on RMSE. Lower is better.

$$
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{(o,h)\in\mathcal{P}}e_{o,h}^{2}}
$$

### B. Forecast Direction — Horizon Step 96 Only

For each **forecast origin ( $o$ )**, evaluate only the prediction at **step 96**. Define its **signed error ( $e_o$ )** as:

$$
e_o=\hat{y}_{o,96}-y_{o,96}
$$

- **Overprediction ( $e_o>0$ ):** The prediction exceeds actual residual load.
- **Underprediction ( $e_o<0$ ):** The prediction is below actual residual load.
- **Exact prediction ( $e_o=0$ ):** No forecasting error.
- **Number of evaluated origins ( $M$ ):** Hourly forecast origins with an available prediction and actual value at step 96. All sums in this section run over these origins.

**Direction metrics:**

- **Mean Bias:** Average signed error at the 24-hour lead time. Positive bias indicates an overall tendency to overpredict; negative bias indicates an overall tendency to underpredict.

$$
\mathrm{Bias}_{24h}=\frac{1}{M}\sum_o e_o
$$

- **Overprediction Rate (%):** Percentage of selected forecasts above the actual value.

$$
\mathrm{Overprediction\ Rate}_{24h}=\frac{\#\{o:e_o>0\}}{M}\times100\%
$$

- **Underprediction Rate (%):** Percentage of selected forecasts below the actual value.

$$
\mathrm{Underprediction\ Rate}_{24h}=\frac{\#\{o:e_o<0\}}{M}\times100\%
$$

- **Total Overprediction:** Sum of positive error magnitudes for the selected forecasts.

$$
\mathrm{Total\ Overprediction}_{24h}=\sum_o\max(e_o,0)
$$

- **Total Underprediction:** Sum of underprediction magnitudes, reported as positive quantities.

$$
\mathrm{Total\ Underprediction}_{24h}=\sum_o\max(-e_o,0)
$$

Zero-error predictions count toward neither direction rate. These totals represent **only the selected final-step intervals**, not the entire daily energy excess or shortfall.

### C. Asymmetric Weighted Loss — Horizon Step 96 Only

**Assumption:** Underprediction is **twice as costly** as overprediction.

- **Underprediction magnitude:** $\max(y_{o,96}-\hat{y}_{o,96},0)$; positive only when predicted residual load is below actual load.
- **Overprediction magnitude:** $\max(\hat{y}_{o,96}-y_{o,96},0)$; positive only when predicted residual load exceeds actual load.
- **Weight ( $\alpha$ ):** Set $\alpha=\frac{2}{3}$ for underprediction and $1-\alpha=\frac{1}{3}$ for overprediction, giving a **2:1 penalty ratio**.
- **Weighted loss ( $L_o$ ):** Penalty for the selected final-step prediction from origin $o$.

$$
L_o=\alpha\max(y_{o,96}-\hat{y}_{o,96},0)
+(1-\alpha)\max(\hat{y}_{o,96}-y_{o,96},0)
$$

With $\alpha=\frac{2}{3}$:

$$
L_o=\frac{2}{3}\max(y_{o,96}-\hat{y}_{o,96},0)
+\frac{1}{3}\max(\hat{y}_{o,96}-y_{o,96},0)
$$

- **Average Weighted Loss:** Simple arithmetic mean of the selected final-step losses across the $M$ forecast origins. Lower is better.

$$
\overline{L}_{24h}=\frac{1}{M}\sum_o L_o
$$

The weighted loss is a **hypothetical cost-sensitive evaluation score**, not a monetary market-cost simulation.

## 4. Evaluation Procedure

1. Generate a 96-step forecast at 15-minute resolution every hour for Chronos-2 and GBT.
2. Match each prediction to the actual residual load of its target interval.
3. Compute **MAE and RMSE across all 96 forecast steps**.
4. Filter to **`horizon_step == 96`**, keeping one final-step prediction per forecast origin.
5. Compute **forecast-direction metrics and asymmetric weighted loss** for the filtered predictions.
6. Compare models using aligned target intervals and lead times under the assumed TransnetBW hourly rolling 24-hour forecasting schedule.

## 5. Final Model Comparison

| Metric | Evaluation Scope | Chronos-2 | GBT | TransnetBW |
|---|---|---:|---:|---:|
| MAE | All 96 horizon steps | — | — | — |
| RMSE | All 96 horizon steps | — | — | — |
| Mean Bias | Step 96 only | — | — | — |
| Overprediction Rate (%) | Step 96 only | — | — | — |
| Underprediction Rate (%) | Step 96 only | — | — | — |
| Total Overprediction | Step 96 only | — | — | — |
| Total Underprediction | Step 96 only | — | — | — |
| **Average Weighted Loss (2:1)** | **Step 96 only** | — | — | — |

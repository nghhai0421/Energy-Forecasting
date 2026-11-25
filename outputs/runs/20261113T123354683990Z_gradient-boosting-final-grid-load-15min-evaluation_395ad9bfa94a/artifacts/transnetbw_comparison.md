# GBT comparison with the published TransnetBW forecast

Evaluated 2026-11-25 using saved predictions; no retraining or test-set tuning.

## Coverage and matching

- Source run: `20261113T123354683990Z_gradient-boosting-final-grid-load-15min-evaluation_395ad9bfa94a`.
- 32,064 identical quarter-hour intervals, 2025-01-01T00:00:00+01:00 through 2025-11-30T23:45:00+01:00 (interval starts).
- 334 GBT forecast origins; each predicts 96 intervals.
- 0 saved prediction rows excluded for missing data.
- Downloaded actual grid load matches saved test labels exactly within 1e-8 MWh.
- SMARD files use MWh per quarter-hour already; no power-to-energy conversion applied.
- UTC joins disambiguate repeated autumn local times. No interpolation of benchmark forecasts.
- Both exports use `grid load [MWh] Original resolutions`. The TransnetBW region
  comes from the supplied download selection; the CSV headers do not encode it.

## Results

| Forecast | MAE (MWh/interval) | RMSE (MWh/interval) | Underprediction total (MWh) | Overprediction total (MWh) |
| --- | ---: | ---: | ---: | ---: |
| GBT | 58.36 | 79.08 | 808,253.49 | 1,063,021.11 |
| Published TransnetBW forecast | 113.55 | 135.40 | 185,020.43 | 3,455,978.70 |

GBT minus published forecast relative error: **-48.61% MAE** and **-41.60% RMSE**
(positive means GBT has higher error). GBT has lower absolute error in
78.24% of scored intervals; this is descriptive, not a significance test.
Underprediction is max(actual minus forecast, 0); overprediction is the reverse.
These measure forecasting errors, not actual purchases or balancing-account positions.

The published series overpredicts in 89.70% of intervals,
with signed mean error (actual minus forecast) -102.01 MWh.
GBT's signed mean error is -7.95 MWh.
This systematic difference deserves investigation; its cause is not established here.

## Monthly MAE

| Month | GBT MAE | Published forecast MAE | Intervals |
| --- | ---: | ---: | ---: |
| 2025-01 | 80.80 | 139.57 | 2,976 |
| 2025-02 | 54.24 | 103.65 | 2,688 |
| 2025-03 | 66.44 | 115.37 | 2,972 |
| 2025-04 | 67.02 | 139.62 | 2,880 |
| 2025-05 | 56.93 | 100.53 | 2,976 |
| 2025-06 | 57.66 | 115.14 | 2,880 |
| 2025-07 | 42.86 | 108.47 | 2,976 |
| 2025-08 | 59.97 | 105.79 | 2,976 |
| 2025-09 | 48.85 | 126.46 | 2,880 |
| 2025-10 | 53.84 | 105.70 | 2,980 |
| 2025-11 | 52.73 | 88.34 | 2,880 |

Coverage is limited to the dates above; no forecasts are extrapolated into missing months.

## What this answers

The saved GBT does forecast held-out 2025 load: weights were fitted on 2023-2024,
and prior 2025 observations refresh lagged inputs at later daily origins without
retraining. Its accuracy can be scored against the published forecast on the same
actual load and delivery timestamps. This report provides that preliminary comparison.
It does not establish superiority under identical forecast-time information.

## Limits to the operational comparison

- GBT issues a rolling next-24-hour forecast at 23:00 UTC (midnight in winter,
  01:00 in summer); the benchmark is a day-ahead delivery-day forecast. The model
  may have more recent information. Lead times and issuance times are not matched.
- CSVs contain delivery timestamps but no forecast issue time or revision identifier.
  We have not established that the downloaded benchmark is the original forecast
  available at the intended cutoff.
- GBT uses lagged historical temperature, not archived weather forecasts. Different
  available feature sets are a limitation when interpreting an operational comparison.
- GBT's lag features include a configured 60-minute availability buffer. This does not verify historical publication vintages or later revisions.
- The comparison covers GBT and the published TransnetBW forecast.
  ARIMA and Chronos were not evaluated on this identical 2025 setup here.
- Forecast accuracy is not procurement savings; no reBAP or cost model is used.

Next: preserve this preliminary result, align issuance/horizons and data availability,
add archived weather forecasts, and rerun all models under the common protocol.
Once results from 2025 inform changes, further 2025 comparisons are exploratory;
reserve another untouched period for a confirmatory final test.

SMARD identifies forecasted consumption as a day-ahead total-load forecast.
ENTSO-E documents forecast revisions and subsequent updates of TransnetBW actual
load using settlement data. These support treating this as a retrospective
comparison rather than a reconstruction of the original information sets.
See the [SMARD user guide](https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf)
and [ENTSO-E load data definitions](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Actual-Total-Load-Day-ahead-Per-Bidding-Zone-6-1-A-6-1-B).

## Artifacts

Aligned predictions are in `predictions/transnetbw_aligned.csv`; metrics and input hashes
are in `metrics/transnetbw_summary.json` and `metrics/transnetbw_monthly.csv` under the run.
The comparison plot is saved to `outputs\runs\20261113T123354683990Z_gradient-boosting-final-grid-load-15min-evaluation_395ad9bfa94a\plots\transnetbw_monthly_comparison.png`.

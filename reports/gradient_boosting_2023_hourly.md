# Gradient boosting: 2023 hourly data

Evaluated on 2026-10-28. Full evaluation run: `20261028T191211673498Z_gradient-boosting-regional-2023-hourly-evaluation_5bd1b2e9581b`.
HPO run: `20261028T190215246255Z_gradient-boosting-regional-2023-hourly-hpo_95db45710093`.

## Changes and data

- Loads `Dataset/preprocessed/preprocessed_dataset_1y_hourly.csv` directly, without re-aggregation.
- Adds short comments to loading, timestamp validation, feature selection, calendar construction, and lagging.
- Keeps result formatting inside the gradient boosting runner; no import from another model runner.
- Uses 8,760 input hours from 2023, with the first 24 removed as lag warm-up (8,736 usable hours).
- Inputs: 24 load lags, hour, weekday, month, holidays, and five weather variables lagged 24 hours.
- Holds out the final 20%; uses 80% of development as each sliding training window.
- Forecasts 24 elapsed hours and refits every 24 hours: 58 development folds and 72 complete test folds.
- Scores 1,728 test hours; the final 19 hours are an incomplete horizon and are unscored.

## Learning-rate HPO

Five trials with TPESampler(seed=42), range [0.03, 0.2] on a log scale.
Selection minimizes mean development-fold RMSE. Test observations are excluded from HPO.
Other settings remain fixed: max_iter=50, max_leaf_nodes=15, min_samples_leaf=30,
l2_regularization=1.0, random_state=42, early_stopping=False.

Selected learning rate: **0.1821474442**.

| Trial (zero-based) | Learning rate | Mean development-fold RMSE (MWh) |
| --- | ---: | ---: |
| 0 | 0.06105316 | 260.31 |
| 1 | 0.18214744 | 223.27 |
| 2 | 0.12028666 | 228.08 |
| 3 | 0.09340308 | 235.91 |
| 4 | 0.04033354 | 339.68 |

## Held-out results

| Model | MAE (MWh) | RMSE (MWh) |
| --- | ---: | ---: |
| Gradient boosting with selected learning rate | 249.84 | 357.12 |
| Weekly seasonal naive | 483.63 | 761.44 |

Relative reduction against the weekly baseline: **48.34% MAE**
and **53.10% RMSE**.
The baseline uses load from 168 hours earlier on exactly the saved prediction timestamps.
Test RMSE is pooled across all scored hours; HPO uses the mean of per-fold RMSEs.

[HPO history](gradient_boosting_2023_hourly_hpo.png) · [First test forecast](gradient_boosting_2023_hourly_forecast.png)

## Validation and reproduction

Five unit tests passed: hourly values/DST, future-observation isolation, gaps and short lags,
non-hourly/duplicate input rejection, and identical rows for the calendar-only variant.
A separate 16-day smoke run passed HPO, evaluation, plotting, and saved-parameter reuse
with identical predictions. Full-run predictions, five completed trials, checkpoint reload,
seven plots, and agreement between HPO and evaluation CV scores were verified.

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.venv/Scripts/python.exe main.py --stage all --exp-cfg configs/experiments/gradient_boosting.yaml --mdl-cfg configs/models/gradient_boosting.yaml
.venv/Scripts/python.exe -m unittest discover -s tests
```

Versions: {"python": "3.11.6", "numpy": "2.4.6", "pandas": "3.0.6", "scikit_learn": "1.9.1", "sktime": "1.2.0", "optuna": "5.0.0"}.
Input CSV SHA-256: `c7222800f9a54f3d7071705d9141797683be4e830ef20fc8c2c739f6dafc53f4`.
Full metrics, predictions, config snapshots, and source/version hashes remain in the local run directory.
This report and its two images are outside the ignored outputs directory, ready to include in a Git commit.

## Limits

This is a 2023 experiment, not a direct repeat of the earlier 2024 dataset evaluation.
It uses lagged observed weather, assumes no publication delay, and does not isolate the benefit
of weather through an ablation. It does not implement archived weather forecasts,
auction-timed delivery days, or procurement-cost simulation. No comparison with the other
model runners is claimed; their existing configurations point to different datasets.
The installed sktime/scikit-learn combination emits non-fatal shape and future-default warnings.

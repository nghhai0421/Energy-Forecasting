# Energy Forecasting

A config-driven experiment setup for comparing models for Germany's electricity
demand forecasting. Run experiments through YAML configs, using a common Optuna
optimizer for HPO and rolling-window splits across models.

The project uses sktime to streamline time-series model fitting, cross-validation
splits, HPO, and evaluation. Each model's runner connects data loading to this
workflow.

Implemented Models:
- ARIMA
- Chronos-2 (pretrained, zero-shot)
- Gradient Boosting (HistGradientBoostingRegressor through sktime YfromX)

Under Development: Build a production-style forecasting demo using Azure Databricks and Azure Data Lake, replaying historical data to simulate incoming observations and trigger preprocessing, inference, and forecast storage.

## Setup

Create and activate the Conda environment:

```powershell
conda env create --file environment.yml
conda activate energy_forecasting
```

Without Conda, use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn matplotlib optuna pyyaml tzdata holidays nbformat nbclient ipykernel pmdarima sktime==1.2.0
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

On Linux/macOS, use `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## Train, evaluate, and tune

Run from the repository root with the environment active. The examples use
ARIMA. For Ridge, replace the experiment path with
`configs/experiments/ridge_prototype.yaml` and the model path with
`configs/models/ridge_regression.yaml`.

Commands and configuration paths are relative to the repository root. Markdown
links are relative to their document. Use `python` from the activated environment
on Windows, Linux or macOS; no personal checkout path is required.

Fit and evaluate with defaults (no HPO):

```powershell
python main.py --stage evaluate --exp-cfg configs/experiments/ARIMA.yaml
```

`evaluate` is the default stage: train/fit the model, run development CV, and
evaluate on the final test set. Add `--mdl-cfg` for optional parameter overrides;
otherwise defaults come from the model function.

HPO only (development CV; no final-test evaluation):

```powershell
python main.py --stage hpo --exp-cfg configs/experiments/ARIMA.yaml --mdl-cfg configs/models/ARIMA.yaml
```

HPO, then fit and evaluate with the selected parameters:

```powershell
python main.py --stage all --exp-cfg configs/experiments/ARIMA.yaml --mdl-cfg configs/models/ARIMA.yaml
```

Reuse saved HPO parameters without tuning again (replace `RUN_ID`):

```powershell
python main.py --stage evaluate --exp-cfg configs/experiments/ARIMA.yaml --hpo-run outputs/hpo_runs/RUN_ID
```

ARIMA/Ridge HPO requires a model YAML with search settings; GBT keeps these in its experiment YAML. `--hpo-run` loads the best
hyperparameters from a saved HPO run. Use `--run-name` to set the run name.


## Gradient boosting

For the retained grid-load experiment use [configs/experiments/gradient_boosting.yaml](configs/experiments/gradient_boosting.yaml), for the final experiment. It contains source files, splits, the selected model parameters and the HPO search space.

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python main.py --stage evaluate --exp-cfg configs/experiments/gradient_boosting.yaml
```

Evaluation uses the selected parameters. Use `--stage hpo` for development-only
search, `--stage all` for search followed by evaluation, or add `--hpo-run <directory>`
to evaluation to reuse a compatible saved search. Omit `--mdl-cfg`.

Like Chronos-2, the model has three modules: `run.py` orchestrates the experiment,
`process.py` prepares data, and `inference.py` loads a model and forecasts.
GBT is trained on separate 2023-2024 files and replays January-November 2025 with
fixed weights. Its 28 features use historical load, temperature, wind, dew point
and price at effective 25h/169h lags, shared calendar encodings and BW holidays.
Each origin predicts 96 quarter-hours. The target is grid load in MWh per interval;
Use [the matched comparison configuration](configs/experiments/gradient_boosting_chronos_comparison.yaml) for January residual load with hourly origins, matching Chronos-2. The [results index](reports/model_results.md) links both comparisons and their artifacts.

Read the [final setup and TransnetBW comparison](reports/transnetbw_2025_final_comparison.md)
for preprocessing, all features, HPO, saved artifacts and reproduction commands.
The [demo notebook](preprocessing/gbt_final_demo.ipynb) demonstrates frozen inference.
GBT evaluation produces the same standard plots as Chronos-2 inside its run. Comparison plots are saved there too.

## Outputs

Each run gets a unique folder under `outputs/hpo_runs/` or `outputs/runs/`.
The parent folder is set by `output_root` in the experiment YAML. The reviewed final
GBT snapshots are versioned so the demo notebook and report links work on a fresh
checkout. Newly generated run directories remain ignored by Git.

Saved files:

- `run_manifest.json`: run settings, status, timings, hardware, and summary.
- `artifacts/split_manifest.json`: development/test boundaries and fold details.
- `artifacts/selected_parameters.json`: model parameters and their source.
- `artifacts/hpo_trials.csv`: trial parameters and validation scores (with HPO).
- `predictions/cv_validation.csv`, `final_test.csv`: actuals, forecasts, and residuals.
- `metrics/cv_validation.json`, `final_test.json`: MAE/RMSE and directional errors overall and by horizon;
  CV also includes fold scores and their mean.
- `metrics/*_evaluation.csv`: per-fold scores and timings.
- `checkpoints/model.joblib`: fitted development model (Ridge/ARIMA/Gradient Boosting).
- `artifacts/development_features.csv`, `first_demo_features.csv`: GBT prepared input tables.
- `artifacts/training_manifest.json`: GBT training provenance.
- `logs/error.log`: details if output saving fails.

Forecast CSVs share exactly these columns: `split,fold,forecast_origin,cutoff_timestamp,target_timestamp,horizon_step,actual,prediction,residual`. Residual is actual minus prediction. Directional totals in standard metrics sum forecast rows: overlapping origins repeat delivery intervals. Use the non-overlapping comparison CSV for energy-volume totals. See [the results index and reproduction commands](reports/model_results.md).

Timing reports mean fit time, mean prediction time, and total evaluation time
separately for validation and test, in seconds. Means are per forecast origin;
total evaluation includes fitting, prediction, scoring, and overhead. For Chronos,
fit means context setup, not weight training. Hardware records the device, CPU
model, and logical core count. These details also appear in the shell summary.

Evaluation graphs for all models, including GBT, are saved in the run's `plots/` folder:

- `forecast_example_1.png` to `forecast_example_3.png`: up to three test forecasts
  against actuals, with recent history and the train/test boundary.
- `final_test_errors_by_horizon.png`: MAE and RMSE by forecast step.
- `final_test_residual_distribution.png`: histogram of actual minus prediction.
- `final_test_residuals_over_time.png`: one-step forecast residuals over time.
- `optuna_validation_history.png`: trial scores and best score so far (with HPO).

HPO-only runs save the manifest and artifacts. Evaluation runs save predictions, metrics and the graphs listed above. Chronos-2 records the
pretrained model ID, resolved revision when available, and inference settings
instead of saving another weights checkpoint.

## Add a model

1. Keep the experiment in `src/models/<model_name>/run.py`. Expose
   `run_experiment(exp_cfg, mdl_cfg=None, run_name=None, stage="evaluate", hpo_run=None)`.
2. Keep configuration in the experiment YAML where practical, following Chronos-2
   and GBT. ARIMA/Ridge also support separate model overrides.
3. Add a dispatch branch in `main.py` and any required dependencies.
4. Use the existing runners as examples for standardized results and call
   `save_run_outputs()`.
5. Save any native checkpoint directly under the returned run's `checkpoints/`
   folder, then call functions from `src/outputs/plot.py` separately.

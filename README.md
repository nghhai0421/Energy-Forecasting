# Energy Forecasting

A config-driven experiment setup for comparing models for Germany's electricity
demand forecasting. Run experiments through YAML configs, using a common Optuna
optimizer for HPO and rolling-window splits across models.

The project uses sktime to streamline time-series model fitting, cross-validation
splits, HPO, and evaluation. Each model's runner connects data loading to this
workflow.

Implemented Models:
- Ridge Regression
- ARIMA
- TCN - A recurrent CNN model

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
python -m pip install numpy pandas scikit-learn matplotlib optuna pyyaml tzdata pmdarima sktime==1.2.0
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

On Linux/macOS, use `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## Train, evaluate, and tune

Run from the repository root with the environment active. The examples use
ARIMA. For Ridge, replace the experiment path with
`configs/experiments/ridge_prototype.yaml` and the model path with
`configs/models/ridge_regression.yaml`.

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

HPO requires a model YAML with search settings. `--hpo-run` loads the best
hyperparameters from a saved HPO run. Use `--run-name` to set the run name.

## Outputs

Each run gets a unique folder under `outputs/hpo_runs/` or `outputs/runs/`.
The parent folder is set by `output_root` in the experiment YAML.

Saved files:

- `run_manifest.json`: run settings, status, timings, and summary.
- `artifacts/split_manifest.json`: development/test boundaries and fold details.
- `artifacts/selected_parameters.json`: model parameters and their source.
- `artifacts/hpo_trials.csv`: trial parameters and validation scores (with HPO).
- `predictions/cv_validation.csv`, `final_test.csv`: actuals, forecasts, and residuals.
- `metrics/cv_validation.json`, `final_test.json`: MAE/RMSE overall and by horizon;
  CV also includes fold scores and their mean.
- `metrics/*_evaluation.csv`: per-fold scores and timings.
- `checkpoints/model.joblib`: fitted development model.
- `logs/error.log`: details if output saving fails.

Graphs in `plots/`:

- `forecast_example_1.png` to `forecast_example_3.png`: up to three test forecasts
  against actuals, with recent history and the train/test boundary.
- `final_test_errors_by_horizon.png`: MAE and RMSE by forecast step.
- `final_test_residual_distribution.png`: histogram of actual minus prediction.
- `final_test_residuals_over_time.png`: one-step forecast residuals over time.
- `optuna_validation_history.png`: trial scores and best score so far (with HPO).

HPO-only runs save the manifest, artifacts, and tuning graph. Evaluation runs
save predictions, metrics, the checkpoint, and forecast graphs too.

## Add a model

1. Keep the experiment in `src/models/<model_name>/run.py`. Expose
   `run_experiment(exp_cfg, mdl_cfg=None, run_name=None, stage="evaluate", hpo_run=None)`.
2. Define model defaults inside the runner function. A model YAML supplies
   optional evaluation overrides and is required when starting HPO.
3. Add a dispatch branch in `main.py` and any required dependencies.
4. Use the existing runners as examples for standardized results and call
   `save_run_outputs()`.
5. Save any native checkpoint directly under the returned run's `checkpoints/`
   folder, then call functions from `src/outputs/plot.py` separately.

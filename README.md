# Energy Forecasting

Compare energy forecasting models with single-file runners. Ridge regression
and ARIMA are implemented. Forecast horizons and steps are in observations;
their duration depends on the dataset frequency.

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

## Run a model

Run from the repository root with the environment active. The experiment YAML
defines data, splits, and forecast settings. A model YAML is optional; model
defaults live inside the model's runner function.

Evaluate Ridge defaults without HPO:

```powershell
python main.py --exp-cfg configs/experiments/ridge_prototype.yaml
```

Use `--mdl-cfg configs/models/ridge_regression.yaml` for optional model/HPO
overrides. Evaluation is the default stage.

Run HPO explicitly with a model YAML containing the search space, trial count,
and seed:

```powershell
python main.py --stage hpo --exp-cfg configs/experiments/ridge_prototype.yaml --mdl-cfg configs/models/ridge_regression.yaml
```

Then evaluate its saved parameters:

```powershell
python main.py --stage evaluate --exp-cfg configs/experiments/ridge_prototype.yaml --hpo-run outputs/hpo_runs/<hpo-run-id>
```

Use `--stage all` for HPO followed by evaluation. HPO saves to
`outputs/hpo_runs/`; evaluation saves to `outputs/runs/`. Use `--output-root`
and `--run-name` to override these settings. Saved HPO runs use plain parameter
names and `hpo_trials.csv`; older Ridge output formats are no longer supported.

## Run ARIMA

Evaluate the function's default order `(1, 1, 1)` without HPO or a model YAML:

```powershell
python main.py --exp-cfg configs/experiments/ARIMA.yaml
```

Tune the candidate orders from the model YAML:

```powershell
python main.py --stage all --exp-cfg configs/experiments/ARIMA.yaml --mdl-cfg configs/models/ARIMA.yaml
```

The supplied ARIMA experiment reads `Dataset/preprocessed_dataset_1y.csv`:
35,136 observations at 15-minute intervals, with no duplicate timestamps or
missing targets. Its horizon and step of 24 both represent six hours. The data
is not resampled. Use equivalent data periods, frequencies, and evaluation
settings when comparing models.

ARIMA reads a comma-separated processed CSV with UTC timestamps and a numeric
target. It uses the full fold training window; Ridge's `lookback` setting does
not apply. ARIMA's native checkpoint is saved as `checkpoints/model.joblib`,
and its prediction, metric, and HPO files follow the common result contract.

Run the focused integration tests on a small synthetic series:

```powershell
python -m unittest discover -s tests -v
```

They include default/override evaluation, a one-trial HPO smoke test, saved-HPO
reuse, checkpoints, plots, and CLI dispatch for both models.

## Add a model

1. Keep the experiment in `src/models/<model_name>/run.py`. Expose
   `run_experiment(exp_cfg, mdl_cfg=None, output_root=None, run_name=None, stage="evaluate", hpo_run=None)`.
2. Define model defaults inside the runner function. A model YAML supplies
   optional evaluation overrides and is required when starting HPO.
3. Add a dispatch branch in `main.py` and any required dependencies.
4. Build the result dict below and call `save_run_outputs()`.
5. Save any native checkpoint directly under the returned run's `checkpoints/`
   folder, then call functions from `src/outputs/plot.py` separately.

A pretrained foundation-model runner can load its chosen model and predict
without model YAML, HPO, or exporting another copy of its weights. It still
uses experiment data/split/horizon settings. Record the pretrained model ID,
revision, and inference settings in run metadata. A Chronos runner is not
implemented here.

## Result dict

Use plain dictionaries and DataFrames; no result class is required:

```python
results = {
    "predictions": predictions,
    "split_manifest": split_manifest,
    "selected_parameters": {
        "parameters": parameters,
        "source": "defaults",
    },
}
run_path = save_run_outputs(results, metadata={"model": model_name})
```

`predictions` is a nonempty DataFrame with `split`, `forecast_origin`,
`target_timestamp`, `horizon_step`, `actual`, and `prediction` columns.
Supported splits are `cv_validation` and `final_test`; CV rows also require
`fold`. A final-test-only result is valid. The writer calculates residuals
and shared MAE/RMSE metrics. Future multi-series runs should include `series_id`.

`split_manifest` records the model runner's actual split boundaries. Forecast
plotting expects `final_test.evaluation_range.start` and a separate timestamped
target series.

Optional result fields:

- `selected_parameters`: a dict with `parameters` and `source`
  (`defaults`, `model_config`, or `hpo`).
- `trials`: a DataFrame with `trial`, `score`, and optional `state`;
  saved as `artifacts/hpo_trials.csv`.
- `tables`: named DataFrames saved under `metrics/`.
- `checkpoint_path`: an exported file or folder, copied into `checkpoints/`.
  Use this only for an existing export. Otherwise the model runner can save
  directly into the returned run folder, using joblib, PyTorch, or its own API.
  Omit this field when no checkpoint is needed.

The manifest's `completed` status covers shared result saving. Native
checkpoints and plots are separate runner steps and propagate their own errors.

HPO results contain `trials`, `split_manifest`, and `selected_parameters`.
The selection records `parameters`, `validation_score`, `metric`, and
`direction` (`minimize` or `maximize`). Tuning plots use these standard fields.

Fit preprocessing on training data only, tune on development data, and keep
final-test scores out of hyperparameter selection. Without HPO, CV reports
the fixed model's performance and does not select parameters.

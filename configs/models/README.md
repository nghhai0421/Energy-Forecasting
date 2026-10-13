# Ridge model YAML

Keep files directly here, such as ridge_regression.yaml.

The experiment selects the model; this file contains model settings and its
optional HPO settings:

```yaml
solver: cholesky
hpo:
  n_trials: 10
  seed: 42
  alpha:
    low: 0.0001
    high: 10000
    log: true
```

`hpo.alpha` contains Optuna FloatDistribution options. sktime's
ForecastingOptunaSearchCV owns the trial loop and CV scoring; it tunes the
Ridge parameter estimator__ridge__alpha inside the reduction/pipeline.
The objective is mean CV-fold RMSE. Ridge uses an intercept and the cholesky
solver. StandardScaler fits on each fold's training windows only.

The output saves sktime's cv_results_ table as optuna_trials.csv, including
trial numbers, scores, timestamps, durations, states, parameter values and
ranks when provided by the installed version. Selected parameters and the
best validation score go into selected_parameters.json. Per-trial fold scores
and an Optuna study object are not exposed by this search API.

Use `main.py --stage hpo` to search and save the selected parameters. Use
`main.py --stage evaluate --hpo-run <run-folder>` to load that saved selection.
Both stages use this same model config; the selected parameters are an output
artifact, not another config file. `--stage all` runs both stages in sequence.
The model's run.py calls sktime directly. There is no custom CV adapter or
Optuna objective function. Deep-learning adapters are deferred. See
docs/model_workflow.md for the working path.

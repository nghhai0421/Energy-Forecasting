"""Gradient boosting: load regional data, tune learning rate, evaluate and save outputs."""

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
import sktime
from optuna.distributions import FloatDistribution
from sklearn.ensemble import HistGradientBoostingRegressor
from sktime.forecasting.compose import DirectReductionForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ForecastingOptunaSearchCV
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.split import SlidingWindowSplitter, temporal_train_test_split

from src.config import load_run_settings
from src.models.gradient_boosting.process import load_data
from src.models.ridge_regression.run import format_results
from src.outputs.output import save_run_outputs
from src.outputs.plot import create_forecast_plots, create_tuning_plot


def run_experiment(exp_cfg, mdl_cfg=None, run_name=None,
                   stage="evaluate", hpo_run=None):
    """Fit/evaluate gradient boosting or tune its learning rate.

    Input: experiment YAML path, optional model YAML path and run name.
    stage: evaluate (default), hpo, or all; hpo_run: saved HPO folder for evaluate.
    Output: run folder Path; all returns a dict with hpo and evaluation Paths.
    """
    if stage not in {"all", "hpo", "evaluate"}:
        raise ValueError(f"Unknown run stage: {stage}")
    if stage != "evaluate" and hpo_run is not None:
        raise ValueError("hpo_run is only used with stage='evaluate'.")

    # Load Configs
    experiment, settings = load_run_settings(exp_cfg, mdl_cfg)
    if stage in {"hpo", "all"} and "hpo" not in settings:
        raise ValueError("HPO requires --mdl-cfg with an hpo section.")
    dataset = experiment["data"]
    lookback = experiment["forecast"]["lookback"]
    horizon = experiment["forecast"]["horizon"]
    step = experiment["forecast"]["step_length"]
    output_root = experiment["output_root"]
    run_name = run_name if run_name is not None else experiment["run_name"]

    # Set Model Defaults
    defaults = dict(learning_rate=0.1, max_iter=50, max_leaf_nodes=15,
                    min_samples_leaf=30, l2_regularization=1.0, random_state=42)
    parameters = {key: settings.get(key, value) for key, value in defaults.items()}
    selection = {
        "parameters": parameters,
        "source": "model_config" if defaults.keys() & settings.keys() else "defaults",
    }
    hpo_path = None
    trials = None
    timing = {}

    # Load Data
    y, X, series, preparation = load_data(dataset, horizon)

    # Split Data
    # Hold out the final test period before tuning.
    test_size = int(len(y) * experiment["split"]["test_fraction"])
    development, test = temporal_train_test_split(y, test_size=test_size)
    window_length = int(len(development) * experiment["split"]["train_window_fraction"])
    fh = np.arange(1, horizon + 1)

    # Build Model
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)
    if window_length <= lookback + horizon or min(len(test), len(development) - window_length) < horizon:
        raise ValueError("Insufficient data for the requested training window and horizon.")
    forecaster = DirectReductionForecaster(
        HistGradientBoostingRegressor(**parameters, early_stopping=False),
        window_length=lookback, X_treatment="concurrent", impute_method=None,
    )
    development_X = X.loc[development.index]
    rmse = MeanSquaredError(square_root=True)

    # Set Run Metadata
    metadata = {
        "model": experiment["model"],
        "sktime_version": sktime.__version__,
        "config_files": {
            "experiment": str(Path(exp_cfg).resolve()),
            "model": str(Path(mdl_cfg).resolve()) if mdl_cfg is not None else None,
        },
        "data": dataset,
        "preparation": preparation,
        "forecast": experiment["forecast"],
        "timing": timing,
    }
    split_manifest = {
        "fractions": experiment["split"],
        "development_observations": len(development),
        "test_observations": len(test),
        "window_length": window_length,
        "splitter": "SlidingWindowSplitter",
        "strategy": "refit",
    }

    # Run HPO
    if stage in {"hpo", "all"}:
        hpo_settings = settings["hpo"]
        search = ForecastingOptunaSearchCV(
            forecaster=forecaster, cv=cv,
            param_grid={"estimator__learning_rate": FloatDistribution(**hpo_settings["learning_rate"])},
            scoring=rmse, strategy="refit", refit=False,
            n_evals=hpo_settings["n_trials"],
            sampler=optuna.samplers.TPESampler(seed=hpo_settings["seed"]),
            backend="threading", error_score="raise",
        )
        started = perf_counter()
        search.fit(development, X=development_X, fh=fh)
        timing["hpo_seconds"] = perf_counter() - started
        trials = search.cv_results_.rename(columns={
            "number": "trial", f"mean_test_{rmse.name}": "score",
        })
        selection = {
            "parameters": {**parameters, "learning_rate": search.best_params_["estimator__learning_rate"]},
            "source": "hpo",
            "trial": int(search.cv_results_.iloc[search.best_index_]["number"]),
            "validation_score": float(search.best_score_),
            "metric": "rmse",
            "direction": "minimize",
        }
        hpo_path = save_run_outputs(
            {"trials": trials, "selected_parameters": selection, "split_manifest": split_manifest},
            output_root=output_root, run_name=f"{run_name}_hpo",
            metadata={**metadata, "run_stage": "hpo"}, stage="hpo",
        )
        create_tuning_plot(hpo_path)
        if stage == "hpo":
            return hpo_path

    # Load HPO Parameters
    if hpo_run is not None:
        hpo_path = Path(hpo_run).expanduser().resolve()
        manifest = json.loads((hpo_path / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("run_stage") != "hpo":
            raise ValueError(f"{hpo_path} is not a completed HPO run.")
        if manifest["model"] != experiment["model"]:
            raise ValueError("The HPO run belongs to a different model.")
        selection = json.loads((hpo_path / "artifacts/selected_parameters.json").read_text(encoding="utf-8"))
        trials = pd.read_csv(hpo_path / "artifacts/hpo_trials.csv")

    print("Fitting development model", flush=True)
    # Fit Model
    model = forecaster.clone().set_params(**{
        f"estimator__{key}": value for key, value in selection["parameters"].items()
    })
    started = perf_counter()
    model.fit(development, X=development_X, fh=fh)
    timing["fit_seconds"] = perf_counter() - started

    print(f"Evaluating {cv.get_n_splits(development)} development folds", flush=True)
    # Evaluate CV
    started = perf_counter()
    validation = evaluate(
        model.clone(), cv=cv, y=development, X=development_X, strategy="refit",
        scoring=[MeanAbsoluteError(), rmse], return_data=True, error_score="raise",
    )
    timing["validation_seconds"] = perf_counter() - started

    # Evaluate Test
    # Start with the last development window and retain original row indices.
    test_history = y.iloc[len(development) - window_length:]
    test_cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)
    print(f"Evaluating {test_cv.get_n_splits(test_history)} test folds", flush=True)
    started = perf_counter()
    final = evaluate(
        model.clone(), cv=test_cv, y=test_history, X=X.loc[test_history.index], strategy="refit",
        scoring=[MeanAbsoluteError(), rmse], return_data=True, error_score="raise",
    )
    timing["test_seconds"] = perf_counter() - started

    # Format Results
    results = format_results(
        {"cv_validation": validation, "final_test": final}, series, split_manifest,
    )
    results["selected_parameters"] = selection
    results["trials"] = trials
    metadata["run_stage"] = "evaluation"
    if hpo_path is not None:
        metadata["hpo_source"] = str(hpo_path)

    # Save Outputs
    evaluation_path = save_run_outputs(
        results, output_root=output_root, run_name=f"{run_name}_evaluation",
        metadata=metadata,
    )
    joblib.dump(model, evaluation_path / "checkpoints/model.joblib")

    # Plot Results
    if trials is not None:
        create_tuning_plot(evaluation_path)
    create_forecast_plots(evaluation_path, series)
    if stage == "all":
        return {"hpo": hpo_path, "evaluation": evaluation_path}
    return evaluation_path

"""ARIMA: load data, tune order (p, d, q), evaluate and save outputs."""

import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
import sktime
from optuna.distributions import CategoricalDistribution
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ForecastingOptunaSearchCV
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.split import SlidingWindowSplitter, temporal_train_test_split

from src.config import load_run_settings
from src.outputs.output import save_run_outputs


def load_series(dataset: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed CSV (or legacy SMARD CSV) and return series DataFrame, dates, and y."""
    path = Path(dataset["path"])
    ts_col = dataset["timestamp_column"]
    target_col = dataset["target_column"]

    raw = pd.read_csv(path)
    if ts_col not in raw.columns:
        # Fallback for semicolon-delimited legacy SMARD CSV files
        raw = pd.read_csv(path, sep=";", decimal=",", thousands=".")
        dates = pd.to_datetime(raw[ts_col], format="%d.%m.%Y %H:%M")
        dates = dates.dt.tz_localize("Europe/Berlin", ambiguous="infer").dt.tz_convert("UTC")
    else:
        dates = pd.to_datetime(raw[ts_col], utc=True)

    values = pd.to_numeric(raw[target_col]).to_numpy(dtype=float)
    series = pd.DataFrame({"timestamp": dates, "target": values})
    y = pd.Series(values, name=target_col)
    return series, dates, y


def run_experiment(exp_cfg, mdl_cfg, output_root=None, run_name=None,
                   stage="all", hpo_run=None):
    """Run HPO, evaluate a saved HPO result, or run both."""
    if stage not in {"all", "hpo", "evaluate"}:
        raise ValueError(f"Unknown run stage: {stage}")
    if stage == "evaluate" and hpo_run is None:
        raise ValueError("Evaluation requires an HPO run directory.")

    # Configs
    experiment, settings = load_run_settings(exp_cfg, mdl_cfg)
    dataset = experiment["data"]  # Adjust paths and columns in the experiment YAML.
    lookback = experiment["forecast"]["lookback"]
    horizon = experiment["forecast"]["horizon"]
    step = experiment["forecast"]["step_length"]
    output_root = output_root if output_root is not None else experiment["output_root"]
    run_name = run_name if run_name is not None else experiment["run_name"]
    if dataset.get("covariates"):
        raise NotImplementedError("This ARIMA runner uses target history only.")

    # Data loading
    series, dates, y = load_series(dataset)

    # Split the final test period off before HPO.
    test_size = int(len(y) * experiment["split"]["test_fraction"])
    development, test = temporal_train_test_split(y, test_size=test_size)
    window_length = int(len(development) * experiment["split"]["train_window_fraction"])
    fh = np.arange(1, horizon + 1)
    if (
        window_length < lookback + horizon
        or len(development) - window_length < horizon
        or len(test) < horizon
    ):
        raise ValueError("The train, validation and test ranges need complete forecast windows.")

    # CV refits on a fixed-size window that slides forward by step observations.
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)

    # Model: sktime ARIMA with order=(p, d, q).
    default_order = tuple(settings.get("order", [1, 1, 1]))
    forecaster = ARIMA(
        order=default_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        suppress_warnings=True,
    )
    rmse = MeanSquaredError(square_root=True)

    # Run information shared by HPO and evaluation.
    metadata = {
        "model": experiment["model"],
        "sktime_version": sktime.__version__,
        "config_files": {
            "experiment": str(Path(exp_cfg).resolve()),
            "model": str(Path(mdl_cfg).resolve()),
        },
        "data": {
            "path": dataset["path"],
            "target_column": dataset["target_column"],
            "covariates": dataset.get("covariates", []),
        },
        "lookback": lookback,
        "forecast_horizon": horizon,
        "step_length": step,
        "cpu": get_cpu_info(),
    }
    split_manifest = {
        "fractions": experiment["split"],
        "development_observations": len(development),
        "development_range": {
            "first_row": 0, "last_row": len(development) - 1,
            "start": dates.iloc[0].isoformat(),
            "end": dates.iloc[len(development) - 1].isoformat(),
        },
        "cv_window_length": window_length,
        "cv_splitter": "SlidingWindowSplitter",
        "cv_strategy": "refit",
        "step_length": step,
    }

    # HPO uses development data only. Candidate (p, d, q) orders are defined in configs/models/ARIMA.yaml.
    if stage in {"hpo", "all"}:
        hpo_settings = settings["hpo"]
        candidate_orders = [tuple(order) for order in hpo_settings["orders"]]
        search = ForecastingOptunaSearchCV(
            forecaster=forecaster, cv=cv,
            param_grid={"order": CategoricalDistribution(candidate_orders)},
            scoring=rmse, strategy="refit",
            refit=False,  # The evaluation stage fits the selected order on development data.
            n_evals=hpo_settings["n_trials"],
            sampler=optuna.samplers.TPESampler(seed=hpo_settings["seed"]),
            backend="threading", error_score="raise",
        )
        started = perf_counter()
        search.fit(development, fh=fh)
        search_seconds = perf_counter() - started
        hpo_path = save_run_outputs(
            {"search": search, "split_manifest": split_manifest},
            output_root=output_root, run_name=f"{run_name}_hpo",
            metadata={**metadata, "run_stage": "hpo",
                      "timing": {"hpo_search_seconds": search_seconds}},
            stage="hpo",
        )
        if stage == "hpo":
            return hpo_path
    else:
        hpo_path = Path(hpo_run).expanduser().resolve()

    # Load the selected parameters from the completed HPO run.
    selection_path = hpo_path / "artifacts" / "selected_parameters.json"
    hpo_manifest_path = hpo_path / "run_manifest.json"
    if not selection_path.is_file() or not hpo_manifest_path.is_file():
        raise FileNotFoundError(f"No completed HPO result found in {hpo_path}.")
    hpo_manifest = json.loads(hpo_manifest_path.read_text(encoding="utf-8"))
    if hpo_manifest.get("status") != "completed" or hpo_manifest.get("run_stage") != "hpo":
        raise ValueError(f"{hpo_path} is not a completed HPO run.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    # Convert JSON list back to tuple for ARIMA's order=(p, d, q) parameter.
    selected_params = dict(selection["parameters"])
    if "order" in selected_params and isinstance(selected_params["order"], list):
        selected_params["order"] = tuple(selected_params["order"])

    # Model fit using HPO hyperparam.
    model = forecaster.clone().set_params(**selected_params)
    started = perf_counter()
    model.fit(development, fh=fh)
    checkpoint_fit_seconds = perf_counter() - started

    # Validation, record time.
    started = perf_counter()
    validation = evaluate(
        model.clone(), cv=cv, y=development, strategy="refit",
        scoring=[MeanAbsoluteError(), rmse], return_data=True, error_score="raise",
    )
    validation_seconds = perf_counter() - started

    # Test Using Sliding Window split, update the starting fit window position.
    test_history_start = len(development) - window_length
    test_history = y.iloc[test_history_start:]  # Keep row indices for output timestamps.
    test_cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)
    started = perf_counter()
    final = evaluate(
        model.clone(), cv=test_cv, y=test_history, strategy="refit",
        scoring=[MeanAbsoluteError(), rmse], return_data=True, error_score="raise",
    )
    test_seconds = perf_counter() - started

    # Record and report the test ranges and rolling-window settings.
    split_manifest.update({
        "test_observations": len(test),
        "test_range": {
            "first_row": len(development), "last_row": len(series) - 1,
            "start": dates.iloc[len(development)].isoformat(),
            "end": dates.iloc[-1].isoformat(),
        },
        "test_window_length": window_length,
        "test_splitter": "SlidingWindowSplitter",
        "test_strategy": "refit",
    })
    metadata.update({
        "run_stage": "evaluation",
        "hpo_source": {
            "run_id": hpo_manifest["run_id"],
            "run_path": str(hpo_path),
            "selected_parameters": selection["parameters"],
            "validation_score": selection["validation_score"],
        },
        "timing": {
            "hpo_search_seconds": hpo_manifest.get("timing", {}).get("hpo_search_seconds"),
            "development_checkpoint_fit_seconds": checkpoint_fit_seconds,
            "cv_validation": summarize_evaluation_time(validation, validation_seconds),
            "final_test": summarize_evaluation_time(final, test_seconds),
        },
        "checkpoint_fit_observations": len(development),
        "splitting": {
            "test_fraction": experiment["split"]["test_fraction"],
            "train_window_fraction": experiment["split"]["train_window_fraction"],
            "development_observations": len(development),
            "test_observations": len(test),
            "cv_validation": {
                "splitter": "SlidingWindowSplitter",
                "training_window_type": "fixed_length_sliding",
                "training_window_observations": window_length,
                "model_update_strategy": "refit",
                "fold_count": len(validation),
            },
            "final_test": {
                "splitter": "SlidingWindowSplitter",
                "training_window_type": "fixed_length_sliding",
                "training_window_observations": window_length,
                "initial_training_window_observations": window_length,
                "initial_training_first_row": test_history_start,
                "initial_training_last_row": len(development) - 1,
                "first_test_row": len(development),
                "model_update_strategy": "refit",
                "fold_count": len(final),
            },
        },
    })

    # Save the checkpoint, predictions, metrics and plots through the output module.
    results = {
        "evaluations": {"cv_validation": validation, "final_test": final},
        "model": model,
        "series": series,
        "split_manifest": split_manifest,
        "selected_parameters": {**selection, "source_hpo_run_id": hpo_manifest["run_id"]},
    }
    evaluation_path = save_run_outputs(
        results, output_root=output_root, run_name=f"{run_name}_evaluation",
        metadata=metadata, stage="evaluation", hpo_run_path=hpo_path,
    )
    if stage == "all":
        return {"hpo": hpo_path, "evaluation": evaluation_path}
    return evaluation_path


def summarize_evaluation_time(evaluation, elapsed):
    """Use sktime's fit/prediction times; elapsed also includes scoring and overhead."""
    return {
        "evaluation_seconds": elapsed,
        "mean_fold_evaluation_seconds": elapsed / len(evaluation),
        "total_fit_seconds": float(evaluation["fit_time"].sum()),
        "total_prediction_seconds": float(evaluation["pred_time"].sum()),
        "mean_fold_fit_seconds": float(evaluation["fit_time"].mean()),
        "mean_fold_prediction_seconds": float(evaluation["pred_time"].mean()),
    }


def get_cpu_info():
    """Return a small CPU description without adding a system-info package."""
    cpu = {
        "name": platform.processor() or None,
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
    }
    if platform.system() == "Windows":
        import winreg
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu["name"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return cpu

"""Chronos-2: foundational model, load data, forecast with pretrained weights, and evaluate."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sktime
import torch
from sktime.forecasting.chronos2 import Chronos2Forecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.split import SlidingWindowSplitter, temporal_train_test_split

from src.config import load_run_settings
from src.outputs.output import save_run_outputs
from src.outputs.plot import create_forecast_plots


def run_experiment(exp_cfg, mdl_cfg=None, run_name=None,
                   stage="evaluate", hpo_run=None):
    """Evaluate pretrained Chronos-2.

    Input: experiment YAML path and optional run name; stage must be evaluate.
    mdl_cfg and hpo_run are unused and must be omitted.
    Output: run folder Path containing predictions, metrics, metadata, and plots.
    """
    if stage != "evaluate" or hpo_run is not None:
        raise ValueError("Chronos-2 supports zero-shot evaluation only; omit HPO options.")
    if mdl_cfg is not None:
        raise ValueError("Chronos-2 uses the experiment YAML only; omit --mdl-cfg.")

    # Load Configs
    experiment, _ = load_run_settings(exp_cfg)
    dataset = experiment["data"]
    lookback = experiment["forecast"]["lookback"]
    horizon = experiment["forecast"]["horizon"]
    step = experiment["forecast"]["step_length"]
    output_root = experiment["output_root"]
    run_name = run_name if run_name is not None else experiment["run_name"]
    covariates = dataset.get("covariates", {})
    historical_columns = covariates.get("historical", [])
    future_columns = covariates.get("future", [])
    covariate_columns = historical_columns + future_columns

    # Load Data
    raw = pd.read_csv(dataset["path"], sep=";", decimal=",", thousands=".")
    dates = pd.to_datetime(raw[dataset["timestamp_column"]], format="%d.%m.%Y %H:%M")
    # SMARD uses Berlin time; infer repeated autumn hours in file order.
    dates = dates.dt.tz_localize("Europe/Berlin", ambiguous="infer").dt.tz_convert("UTC")
    values = pd.to_numeric(raw[dataset["target_column"]]).to_numpy(dtype=float)
    series = pd.DataFrame({"timestamp": dates, "target": values})
    y = pd.Series(values, name=dataset["target_column"])
    X = raw[covariate_columns].apply(pd.to_numeric).astype(float) if covariate_columns else None

    # Split Data
    test_size = int(len(y) * experiment["split"]["test_fraction"])
    development, test = temporal_train_test_split(y, test_size=test_size)
    window_length = int(len(development) * experiment["split"]["train_window_fraction"])
    # Keep the full requested context.
    if window_length < lookback:
        raise ValueError("The rolling window must contain at least forecast.lookback observations.")
    # Leave at least one complete CV and test forecast.
    if test_size < horizon or len(development) - window_length < horizon:
        raise ValueError("Development validation and final test each need a full forecast horizon.")
    fh = np.arange(1, horizon + 1)
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)

    # Load Model
    parameters = {
        "model_path": "amazon/chronos-2",
        "context_length": lookback,
        "device_map": "cpu",
        "torch_dtype": "float32",
        "limit_prediction_length": True,
        "seed": 42,
    }
    forecaster = Chronos2Forecaster(
        model_path=parameters["model_path"], seed=parameters["seed"],
        config={
            "context_length": lookback,
            "device_map": parameters["device_map"],
            "torch_dtype": torch.float32,
            "limit_prediction_length": parameters["limit_prediction_length"],
        },
    )
    started = perf_counter()
    # fit loads cached pretrained weights and sets context
    development_X = X.loc[development.index] if X is not None else None
    forecaster.fit(development, X=development_X, fh=fh)
    timing = {"load_seconds": perf_counter() - started}
    revision = getattr(forecaster.model_pipeline.model.config, "_commit_hash", None)

    # Set Run Metadata
    metadata = {
        "model": experiment["model"],
        "sktime_version": sktime.__version__,
        "config_files": {"experiment": str(Path(exp_cfg).resolve()), "model": None},
        "inference_mode": "zero_shot",
        "model_revision": revision,
        "data": dataset,
        "forecast": experiment["forecast"],
        "timing": timing,
        "run_stage": "evaluation",
    }
    split_manifest = {
        "fractions": experiment["split"],
        "development_observations": len(development),
        "test_observations": len(test),
        "window_length": window_length,
        "context_length": lookback,
        "splitter": "SlidingWindowSplitter",
        "strategy": "refit",
    }

    # Evaluate CV
    # Reset history at each origin; pretrained weights stay unchanged.
    started = perf_counter()
    validation = evaluate_folds(forecaster, cv, development, X, future_columns)
    timing["validation_seconds"] = perf_counter() - started

    # Evaluate Test
    # Start at the test boundary using the latest development history.
    test_history = y.iloc[len(development) - window_length:]
    test_cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)
    started = perf_counter()
    final = evaluate_folds(forecaster, test_cv, test_history, X, future_columns)
    timing["test_seconds"] = perf_counter() - started

    # Save Outputs
    results = format_results(
        {"cv_validation": validation, "final_test": final}, series, split_manifest,
    )
    results["selected_parameters"] = {"parameters": parameters, "source": "defaults"}
    evaluation_path = save_run_outputs(
        results, output_root=output_root, run_name=f"{run_name}_evaluation",
        metadata=metadata,
    )

    # Plot Results
    # The pretrained model stays in its download cache; no checkpoint copy is needed.
    create_forecast_plots(evaluation_path, series)
    return evaluation_path


def evaluate_folds(forecaster, cv, y, X=None, future_columns=None):
    """Evaluate y/X folds, passing only future_columns at prediction; return fold rows."""
    mae = MeanAbsoluteError()
    rmse = MeanSquaredError(square_root=True)
    evaluations = []
    for train_positions, test_positions in cv.split(y):
        # Set History
        training = y.iloc[train_positions]
        actual = y.iloc[test_positions]
        history_X = X.loc[training.index] if X is not None else None
        model = forecaster.clone()
        started = perf_counter()
        model.fit(training, X=history_X, fh=cv.fh)
        fit_time = perf_counter() - started

        # Forecast
        # Past values include both groups; only known-future columns enter the horizon.
        future_X = X.loc[actual.index, future_columns] if future_columns else None
        started = perf_counter()
        prediction = model.predict(fh=cv.fh, X=future_X)
        pred_time = perf_counter() - started
        evaluations.append({
            f"test_{mae.name}": float(mae(actual, prediction)),
            f"test_{rmse.name}": float(rmse(actual, prediction)),
            "fit_time": fit_time, "pred_time": pred_time,
            "len_train_window": len(training), "cutoff": int(training.index[-1]),
            "y_train": training, "y_test": actual, "y_pred": prediction,
        })
    return pd.DataFrame(evaluations)


def format_results(evaluations, series, split_manifest):
    """Convert rolling evaluations and timestamps to standard rows, tables, and splits."""
    # Format Predictions
    predictions = []
    tables = {}
    split_manifest = split_manifest.copy()
    dates = series["timestamp"]
    for split, evaluation in evaluations.items():
        split_rows = []
        folds = []
        for number, (_, result) in enumerate(evaluation.iterrows(), start=1):
            cutoff = int(result["cutoff"])
            forecast = result["y_pred"]
            training = result["y_train"]
            rows = pd.DataFrame({
                "split": split,
                "fold": pd.array([number if split == "cv_validation" else None] * len(forecast), dtype="Int64"),
                "forecast_origin": dates.iloc[cutoff + 1],
                "cutoff_timestamp": dates.iloc[cutoff],
                "target_timestamp": dates.iloc[forecast.index].to_numpy(),
                "horizon_step": forecast.index.to_numpy() - cutoff,
                "actual": result["y_test"].reindex(forecast.index).to_numpy(),
                "prediction": forecast.to_numpy(),
            })
            rows["residual"] = rows["actual"] - rows["prediction"]
            split_rows.append(rows)

            # Record Ranges
            fold = {"fold": number}
            context = training.index[-split_manifest["context_length"]:]
            for name, positions in (("fit_range", training.index),
                                    ("context_range", context),
                                    ("evaluation_range", forecast.index)):
                first, last = int(positions[0]), int(positions[-1])
                fold[name] = {
                    "first_row": first, "last_row": last, "n_observations": len(positions),
                    "start": dates.iloc[first].isoformat(),
                    "end": dates.iloc[last].isoformat(),
                }
            fold["forecast_origin"] = dates.iloc[cutoff + 1].isoformat()
            folds.append(fold)

        # Record Split
        rows = pd.concat(split_rows, ignore_index=True)
        predictions.append(rows)
        tables[f"{split}_evaluation"] = evaluation.drop(columns=["y_train", "y_test", "y_pred"])
        if split == "cv_validation":
            split_manifest["cv_folds"] = folds
        else:
            start = split_manifest["development_observations"]
            scored = int(rows["target_timestamp"].nunique())
            split_manifest["final_test"] = {
                "initial_fit_range": folds[0]["fit_range"],
                "evaluation_range": {
                    "first_row": start, "last_row": len(series) - 1,
                    "n_observations": len(series) - start,
                    "start": dates.iloc[start].isoformat(), "end": dates.iloc[-1].isoformat(),
                },
                "folds": folds,
                "forecast_origins": len(folds),
                "scored_observations": scored,
                "unscored_observations": split_manifest["test_observations"] - scored,
            }
    return {
        "predictions": pd.concat(predictions, ignore_index=True),
        "tables": tables,
        "split_manifest": split_manifest,
    }

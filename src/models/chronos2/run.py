"""Chronos-2: foundational model, load data, forecast with pretrained weights, and evaluate."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sktime
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.split import SlidingWindowSplitter, temporal_train_test_split

from src.config import load_run_settings
from src.models.chronos2.inference import forecast, load_forecaster
from src.models.chronos2.process import load_data, prepare_future_covariates
from src.outputs.output import save_run_outputs
from src.outputs.plot import create_forecast_plots


def run_experiment(exp_cfg, mdl_cfg=None, run_name=None,
                   stage="evaluate", hpo_run=None):
    """Evaluate pretrained Chronos-2 on chronological development and test folds.

    Parameters
    ----------
    exp_cfg : str or pathlib.Path
        Experiment YAML path defining data, splits, forecasts, and output root.
    mdl_cfg : None, default None
        Reserved for the common runner interface; must be omitted.
    run_name : str or None, default None
        Run label; defaults to the experiment configuration's run name.
    stage : {'evaluate'}, default 'evaluate'
        Run zero-shot evaluation.
    hpo_run : None, default None
        Reserved for the common runner interface; must be omitted.

    Returns
    -------
    pathlib.Path
        Run directory containing saved predictions, metrics, metadata, and plots.

    Raises
    ------
    ValueError
        If HPO options are supplied or a full forecast window is unavailable.
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

    # Load Data
    y, X, series, preparation = load_data(dataset)

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
    loaded = load_forecaster(development, X.loc[development.index], context_length=lookback)
    forecaster = loaded["forecaster"]
    parameters = loaded["parameters"]
    timing = {"load_seconds": loaded["load_seconds"]}

    # Set Run Metadata
    metadata = {
        "model": experiment["model"],
        "hardware": {"device": parameters["device_map"]},
        "sktime_version": sktime.__version__,
        "config_files": {"experiment": str(Path(exp_cfg).resolve()), "model": None},
        "inference_mode": "zero_shot",
        "model_revision": loaded["revision"],
        "data": dataset,
        "preparation": preparation,
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
    validation = evaluate_folds(forecaster, cv, development, X, series, dataset)
    timing["cv_validation"] = {
        "total_eval_seconds": perf_counter() - started,
        "mean_fit_seconds": float(validation["fit_time"].mean()),
        "mean_predict_seconds": float(validation["pred_time"].mean()),
    }

    # Evaluate Test
    # Start at the test boundary using the latest development history.
    test_history = y.iloc[len(development) - window_length:]
    test_cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)
    started = perf_counter()
    final = evaluate_folds(forecaster, test_cv, test_history, X, series, dataset)
    timing["final_test"] = {
        "total_eval_seconds": perf_counter() - started,
        "mean_fit_seconds": float(final["fit_time"].mean()),
        "mean_predict_seconds": float(final["pred_time"].mean()),
    }

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


def evaluate_folds(forecaster, cv, y, X, series, dataset):
    """Evaluate chronological folds using available historical and future covariates.

    Parameters
    ----------
    forecaster : sktime forecaster
        Loaded forecaster whose context is replaced for each fold.
    cv : sktime splitter
        Chronological splitter exposing split(y) and fh.
    y : pandas.Series
        Target observations with indices aligned to X.
    X : pandas.DataFrame
        Historical and known-future covariates indexed like y.
    series : pandas.DataFrame
        Original timestamp/target rows indexed like y. Missing targets are excluded
        from scoring while imputed y remains available as model context.
    dataset : dict
        Data settings used to generate calendar and holiday horizon inputs.

    Returns
    -------
    pandas.DataFrame
        Fold predictions, training/test targets, MAE/RMSE, cutoffs, and timings.

    Raises
    ------
    ValueError
        If a fold contains no observed targets for scoring.
    """
    mae = MeanAbsoluteError()
    rmse = MeanSquaredError(square_root=True)
    evaluations = []
    for train_positions, test_positions in cv.split(y):
        # Set History
        training = y.iloc[train_positions]
        test_index = y.iloc[test_positions].index
        actual = series.loc[test_index, "target"]
        history_X = X.loc[training.index]

        # Forecast
        future_X = prepare_future_covariates(series.loc[test_index, "timestamp"], dataset)
        result = forecast(forecaster, training, history_X, future_X)
        prediction = result["prediction"]
        observed = actual.notna()
        if not observed.any():
            raise ValueError("Each evaluation fold needs at least one observed target.")
        scored_actual = actual[observed].to_numpy()
        scored_prediction = prediction.loc[actual.index[observed]].to_numpy()
        evaluations.append({
            f"test_{mae.name}": float(mae(scored_actual, scored_prediction)),
            f"test_{rmse.name}": float(rmse(scored_actual, scored_prediction)),
            "fit_time": result["fit_seconds"], "pred_time": result["predict_seconds"],
            "len_train_window": len(training), "cutoff": int(training.index[-1]),
            "y_train": training, "y_test": actual, "y_pred": prediction,
        })
    return pd.DataFrame(evaluations)


def format_results(evaluations, series, split_manifest):
    """Convert rolling forecast evaluations into standardized results.

    Parameters
    ----------
    evaluations : dict of pandas.DataFrame
        Split names mapped to fold rows with cutoff, y_train, y_test, and y_pred.
    series : pandas.DataFrame
        Timestamp and target columns aligned with the evaluation row indices.
    split_manifest : dict
        Split counts and window settings; Chronos-2 also requires context_length.

    Returns
    -------
    dict
        Prediction rows, evaluation tables, and split boundaries in a copied manifest.
    """
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
            rows = rows.dropna(subset=["actual"])
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

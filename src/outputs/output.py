"""Create a run folder and save finished results through save_run_outputs()."""

import json
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Save figures on laptops and headless cloud machines.
import matplotlib.pyplot as plt

from src.evaluation.metrics import summarize_predictions


def save_run_outputs(results, output_root="outputs", run_name=None,
                     metadata=None, stage="evaluation", hpo_run_path=None):
    """Create a run, save its finished result dictionary and return the path.

    For an HPO stage, results contains search and split_manifest. For evaluation,
    it contains the fitted model, predictions, split_manifest and selected parameters.
    Saving failures are logged, marked failed and re-raised. Training happens
    before this call, so training failures do not create a run folder or log.
    """
    stage_folders = {"hpo": "hpo_runs", "evaluation": "runs"}
    if stage not in stage_folders:
        raise ValueError(f"Unsupported run output stage: {stage}")
    run_path = create_run_folder(Path(output_root) / stage_folders[stage], run_name)
    print(f"Run directory: {run_path}", flush=True)
    try:
        update_run_manifest(run_path, "saving", **(metadata or {}))
        if stage == "hpo":
            summary = save_hpo_results(run_path, **results)
        elif stage == "evaluation":
            summary = save_results(run_path, hpo_run_path=hpo_run_path, **results)
        update_run_manifest(run_path, "completed", summary=summary)
    except (Exception, KeyboardInterrupt) as error:
        (run_path / "logs/error.log").write_text(traceback.format_exc(), encoding="utf-8")
        update_run_manifest(run_path, "failed", error=str(error))
        raise
    print(json.dumps(summary, indent=2, allow_nan=False))
    return run_path


def create_run_folder(output_root, run_name):
    """Reserve a unique run folder; never reuse an existing directory."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    label = re.sub(r"[^a-z0-9]+", "-", (run_name or "run").lower()).strip("-")
    label = label[:64].rstrip("-") or "run"
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    while True:
        run_id = f"{timestamp}_{label}_{uuid4().hex[:12]}"
        run_path = root / run_id
        try:
            run_path.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        break
    for name in ("predictions", "metrics", "plots", "checkpoints", "artifacts", "logs"):
        (run_path / name).mkdir()
    save_json(run_path / "run_manifest.json", {
        "run_id": run_id,
        "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
        "status": "initialized",
        "run_name": run_name,
    })
    return run_path


def update_run_manifest(run_path, status, **metadata):
    """Update run status while preserving its original identity."""
    path = run_path / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(metadata)
    manifest["status"] = status
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    temporary = path.with_suffix(".tmp")
    save_json(temporary, manifest)
    temporary.replace(path)


def save_hpo_results(run_path, search, split_manifest):
    """Save the search results without creating evaluation predictions."""
    manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
    split_manifest = split_manifest.copy()
    split_manifest.update({
        "dataset": manifest["data"]["path"],
        "target_column": manifest["data"]["target_column"],
        "lookback": manifest["lookback"],
        "forecast_horizon": manifest["forecast_horizon"],
    })
    save_json(run_path / "artifacts/split_manifest.json", split_manifest)
    save_tuning_results(run_path, search)
    create_tuning_plot(run_path, f"mean_test_{search.scoring.name}")
    return {
        "selected_parameters": search.best_params_,
        "best_validation_rmse": float(search.best_score_),
        "trial_count": len(search.cv_results_),
    }


def save_results(run_path, evaluations, model, series, split_manifest,
                 selected_parameters, hpo_run_path):
    """Save sktime evaluation tables, prediction rows, metrics and the forecaster."""
    run_path = Path(run_path)
    scores = {}
    split_manifest = split_manifest.copy()
    manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
    split_manifest.update({
        "dataset": manifest["data"]["path"],
        "target_column": manifest["data"]["target_column"],
        "lookback": manifest["lookback"],
        "forecast_horizon": manifest["forecast_horizon"],
    })
    for split, evaluation in evaluations.items():
        rows, folds = format_evaluation(evaluation, series, split)
        scores[split] = summarize_predictions(rows)
        save_predictions_and_metrics(run_path, rows, scores[split])
        # Nested training/prediction Series stay in memory; save only scalar fields.
        table = evaluation.drop(columns=["y_train", "y_test", "y_pred"])
        table.to_csv(run_path / "metrics" / f"{split}_evaluation.csv", index=False)
        if split == "cv_validation":
            split_manifest["cv_folds"] = folds
        else:
            start = split_manifest["development_observations"]
            scored = int(rows["target_timestamp"].nunique())
            split_manifest["final_test"] = {
                "initial_fit_range": folds[0]["fit_range"],
                "evaluation_range": describe_range(series, np.arange(start, len(series))),
                "folds": folds,
                "forecast_origins": len(folds),
                "scored_observations": scored,
                "unscored_observations": split_manifest["test_observations"] - scored,
            }

    save_json(run_path / "artifacts/split_manifest.json", split_manifest)
    save_json(run_path / "artifacts/selected_parameters.json", selected_parameters)
    # Keep trial details and their plot with evaluation outputs, and retain the
    # originating HPO run separately for full provenance.
    source = Path(hpo_run_path)
    shutil.copy2(source / "artifacts/optuna_trials.csv", run_path / "artifacts/optuna_trials.csv")
    shutil.copy2(source / "plots/optuna_validation_history.png",
                 run_path / "plots/optuna_validation_history.png")
    joblib.dump(model, run_path / "checkpoints/ridge.joblib")
    create_forecast_plots(run_path, series)

    validation, test = scores["cv_validation"], scores["final_test"]
    return {
        "cv_mean_fold": validation["mean_fold"],
        "cv_mean_fold_out_of_range": validation["mean_fold_out_of_range"],
        "final_test": test["overall"],
        "final_test_out_of_range": test["out_of_range"],
        "selected_parameters": selected_parameters["parameters"],
        "best_validation_rmse": float(selected_parameters["validation_score"]),
    }


def describe_range(series, positions):
    """Describe inclusive observation positions without saving the dataset."""
    first, last = int(positions[0]), int(positions[-1])
    return {
        "first_row": first, "last_row": last, "n_observations": len(positions),
        "start": series["timestamp"].iloc[first].isoformat(),
        "end": series["timestamp"].iloc[last].isoformat(),
    }


def format_evaluation(evaluation, series, split):
    """Flatten sktime's per-cutoff Series into rows for saving and plotting."""
    predictions, folds = [], []
    dates = series["timestamp"]
    for number, (_, result) in enumerate(evaluation.iterrows(), start=1):
        cutoff = int(result["cutoff"])
        forecast = result["y_pred"]
        # Both CV and test refit at each origin; diagnostics use that fit's target range.
        training = result["y_train"]
        low, high = float(training.min()), float(training.max())
        rows = pd.DataFrame({
            "split": split,
            "fold": number if split == "cv_validation" else None,
            "forecast_origin": dates.iloc[cutoff + 1],
            "cutoff_timestamp": dates.iloc[cutoff],
            "target_timestamp": dates.iloc[forecast.index].to_numpy(),
            "horizon_step": forecast.index.to_numpy() - cutoff,
            "actual": result["y_test"].reindex(forecast.index).to_numpy(),
            "prediction": forecast.to_numpy(),
            "fit_target_min": low,
            "fit_target_max": high,
        })
        rows["residual"] = rows["actual"] - rows["prediction"]
        rows["out_of_range"] = (rows["actual"] < low) | (rows["actual"] > high)
        predictions.append(rows)
        folds.append({
            "fold": number,
            "fit_range": describe_range(series, training.index),
            "evaluation_range": describe_range(series, forecast.index),
            "forecast_origin": dates.iloc[cutoff + 1].isoformat(),
            "fit_target_min": low, "fit_target_max": high,
        })
    return pd.concat(predictions, ignore_index=True), folds


def save_json(path: Path, payload: dict) -> None:
    """Write strict JSON; NaN/Infinity cannot silently become invalid metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_predictions_and_metrics(run_path: Path, predictions: pd.DataFrame, metrics: dict) -> None:
    """Save each evaluation split separately in the existing run layout."""
    split = metrics["split"]
    if split not in {"cv_validation", "final_test"}:
        raise ValueError(f"Unsupported result split: {split}")
    if set(predictions["split"]) != {split}:
        raise ValueError("Prediction and metric splits must agree.")
    predictions.to_csv(run_path / "predictions" / f"{split}.csv", index=False)
    save_json(run_path / "metrics" / f"{split}.json", metrics)


def save_tuning_results(run_path: Path, search) -> None:
    """Save every column returned by sktime's Optuna search and the selection."""
    pd.DataFrame(search.cv_results_).to_csv(run_path / "artifacts/optuna_trials.csv", index=False)
    selection = {
        "parameters": search.best_params_,
        "trial": int(search.cv_results_.iloc[search.best_index_]["number"]),
        "validation_score": float(search.best_score_),
        "objective": "mean_fold_validation_rmse",
    }
    save_json(run_path / "artifacts/selected_parameters.json", selection)


def save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_forecast_plots(run_path: Path, series) -> None:
    """Plot final-test forecasts, horizon errors and residuals."""
    output = run_path / "plots"
    predictions = pd.read_csv(run_path / "predictions" / "final_test.csv")
    for column in ("forecast_origin", "target_timestamp"):
        predictions[column] = pd.to_datetime(predictions[column], utc=True)
    splits = json.loads((run_path / "artifacts" / "split_manifest.json").read_text(encoding="utf-8"))
    boundary = pd.Timestamp(splits["final_test"]["evaluation_range"]["start"])
    boundary_row = int(series["timestamp"].searchsorted(boundary))
    origins = predictions["forecast_origin"].drop_duplicates().sort_values().reset_index(drop=True)
    # Examples near the boundary keep both recent history and the boundary visible.
    positions = range(min(3, len(origins)))
    for example, position in enumerate(positions, start=1):
        origin = origins.iloc[position]
        forecast = predictions[predictions["forecast_origin"] == origin].sort_values("horizon_step")
        origin_row = int(series["timestamp"].searchsorted(origin))
        history = series.iloc[max(0, boundary_row - 48):origin_row]
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.plot(history["timestamp"], history["target"], label="Observed history", color="0.4")
        axis.plot(forecast["target_timestamp"], forecast["actual"], label="Actual", color="tab:blue")
        axis.plot(forecast["target_timestamp"], forecast["prediction"], label="Forecast", color="tab:orange")
        axis.axvline(boundary, linestyle="--", color="black", label="Train/test boundary")
        axis.axvline(origin, linestyle=":", color="tab:orange", label="Forecast origin")
        axis.set(title=f"Final-test example {example}: origin {origin.isoformat()}",
                 xlabel="Time", ylabel="Target")
        axis.legend(loc="best", fontsize=8)
        figure.autofmt_xdate()
        save_figure(figure, output / f"forecast_example_{example}.png")

    metrics = json.loads((run_path / "metrics" / "final_test.json").read_text(encoding="utf-8"))
    by_horizon = pd.DataFrame(metrics["by_horizon"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("mae", "rmse")):
        axis.plot(by_horizon["horizon_step"], by_horizon[metric], marker="o", markersize=3)
        axis.set(title=f"Final-test {metric.upper()}", xlabel="Forecast horizon step",
                 ylabel=metric.upper())
        axis.set_ylim(bottom=0)
    save_figure(figure, output / "final_test_errors_by_horizon.png")

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(predictions["residual"], bins=40, color="tab:blue")
    axis.axvline(0, color="black", linestyle="--")
    axis.set(title="Final-test residuals (all origin/horizon pairs)",
             xlabel="Actual minus prediction", ylabel="Prediction rows")
    save_figure(figure, output / "final_test_residual_distribution.png")

    one_step = predictions[predictions["horizon_step"] == 1]
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(one_step["target_timestamp"], one_step["residual"], linewidth=0.7)
    axis.axhline(0, color="black", linestyle="--")
    axis.set(title="Final-test residuals: horizon step 1", xlabel="Time",
             ylabel="Actual minus prediction")
    figure.autofmt_xdate()
    save_figure(figure, output / "final_test_residuals_over_time.png")


def create_tuning_plot(run_path: Path, score_column: str) -> None:
    trials = pd.read_csv(run_path / "artifacts/optuna_trials.csv")
    completed = trials[trials["state"] == "COMPLETE"].sort_values("number")
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(completed["number"], completed[score_column], "o-", label="Trial validation score")
    axis.plot(completed["number"], np.minimum.accumulate(completed[score_column]),
              linestyle="--", label="Best so far")
    axis.set(title="Optuna validation history", xlabel="Trial number",
             ylabel="Mean-fold validation RMSE")
    axis.legend()
    save_figure(figure, run_path / "plots/optuna_validation_history.png")

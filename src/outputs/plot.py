"""Plot saved forecasting results independently of output saving."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Save plots without a display.
import matplotlib.pyplot as plt


def create_forecast_plots(run_path: Path, series) -> None:
    """Read saved predictions at run_path and timestamped series; write PNG plots."""
    # Load Predictions
    run_path = Path(run_path)
    output = run_path / "plots"
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(run_path / "predictions" / "final_test.csv")
    for column in ("forecast_origin", "target_timestamp"):
        predictions[column] = pd.to_datetime(predictions[column], utc=True)
    splits = json.loads((run_path / "artifacts" / "split_manifest.json").read_text(encoding="utf-8"))
    boundary = pd.Timestamp(splits["final_test"]["evaluation_range"]["start"])
    boundary_row = int(series["timestamp"].searchsorted(boundary))
    origins = predictions["forecast_origin"].drop_duplicates().sort_values().reset_index(drop=True)

    # Plot Forecasts
    # Keep examples near the test boundary.
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
        figure.tight_layout()
        figure.savefig(output / f"forecast_example_{example}.png", dpi=160)
        plt.close(figure)

    # Plot Horizon Errors
    metrics = json.loads((run_path / "metrics" / "final_test.json").read_text(encoding="utf-8"))
    by_horizon = pd.DataFrame(metrics["by_horizon"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("mae", "rmse")):
        axis.plot(by_horizon["horizon_step"], by_horizon[metric], marker="o", markersize=3)
        axis.set(title=f"Final-test {metric.upper()}", xlabel="Forecast horizon step",
                 ylabel=metric.upper())
        axis.set_ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(output / "final_test_errors_by_horizon.png", dpi=160)
    plt.close(figure)

    # Plot Residuals
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(predictions["residual"], bins=40, color="tab:blue")
    axis.axvline(0, color="black", linestyle="--")
    axis.set(title="Final-test residuals (all origin/horizon pairs)",
             xlabel="Actual minus prediction", ylabel="Prediction rows")
    figure.tight_layout()
    figure.savefig(output / "final_test_residual_distribution.png", dpi=160)
    plt.close(figure)

    # Plot One-Step Residuals
    one_step = predictions[predictions["horizon_step"] == 1]
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(one_step["target_timestamp"], one_step["residual"], linewidth=0.7)
    axis.axhline(0, color="black", linestyle="--")
    axis.set(title="Final-test residuals: horizon step 1", xlabel="Time",
             ylabel="Actual minus prediction")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output / "final_test_residuals_over_time.png", dpi=160)
    plt.close(figure)


def create_tuning_plot(run_path: Path) -> None:
    """Read standardized trials and selection at run_path; write the tuning PNG."""
    # Load Trials
    run_path = Path(run_path)
    output = run_path / "plots"
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(run_path / "artifacts/hpo_trials.csv")
    selection_path = run_path / "artifacts/selected_parameters.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    completed = trials
    if "state" in completed:
        completed = completed[completed["state"].astype(str).str.upper() == "COMPLETE"]
    completed = completed.sort_values("trial")

    # Plot Scores
    figure, axis = plt.subplots(figsize=(8, 4))
    scores = completed["score"].to_numpy()
    if selection.get("direction", "minimize") == "maximize":
        best = np.maximum.accumulate(scores)
    else:
        best = np.minimum.accumulate(scores)
    axis.plot(completed["trial"], scores, "o-", label="Trial validation score")
    axis.plot(completed["trial"], best, linestyle="--", label="Best so far")
    axis.set(title="Optuna validation history", xlabel="Trial number",
             ylabel=selection.get("metric", "Validation score"))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "optuna_validation_history.png", dpi=160)
    plt.close(figure)

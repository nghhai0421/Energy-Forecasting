"""Forecast errors and out-of-range diagnostics."""

import numpy as np
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError


def calculate_mae_rmse(rows):
    if rows.empty:
        return {"n_predictions": 0, "mae": None, "rmse": None}
    return {
        "n_predictions": len(rows),
        "mae": float(MeanAbsoluteError()(rows["actual"].to_numpy(), rows["prediction"].to_numpy())),
        "rmse": float(MeanSquaredError(square_root=True)(rows["actual"].to_numpy(), rows["prediction"].to_numpy())),
    }


def calculate_metrics_by_horizon(rows):
    return [
        {"horizon_step": int(step), **calculate_mae_rmse(group)}
        for step, group in rows.groupby("horizon_step")
    ]


def average_fold_metrics(scores):
    """Average fold errors separately; skip undefined errors from empty subsets."""
    valid = [score for score in scores if score["n_predictions"] > 0]
    return {
        "mae": float(np.mean([score["mae"] for score in valid])) if valid else None,
        "rmse": float(np.mean([score["rmse"] for score in valid])) if valid else None,
    }


def summarize_predictions(predictions):
    rows = predictions.copy()
    rows["out_of_range"] = (
        (rows["actual"] < rows["fit_target_min"]) | (rows["actual"] > rows["fit_target_max"])
    )
    metrics = {
        "split": rows["split"].iloc[0],
        "overall": calculate_mae_rmse(rows),
        "by_horizon": calculate_metrics_by_horizon(rows),
        "out_of_range": calculate_mae_rmse(rows[rows["out_of_range"]]),
    }
    if metrics["split"] == "cv_validation":
        folds = []
        for number, group in rows.groupby("fold"):
            folds.append({
                "fold": int(number),
                "fit_target_min": float(group["fit_target_min"].iloc[0]),
                "fit_target_max": float(group["fit_target_max"].iloc[0]),
                "overall": calculate_mae_rmse(group),
                "out_of_range": calculate_mae_rmse(group[group["out_of_range"]]),
            })
        diagnostics = [fold["out_of_range"] for fold in folds]
        metrics["folds"] = folds
        metrics["mean_fold"] = average_fold_metrics([fold["overall"] for fold in folds])
        metrics["mean_fold_out_of_range"] = {
            **average_fold_metrics(diagnostics),
            "folds_with_cases": sum(score["n_predictions"] > 0 for score in diagnostics),
            "total_folds": len(folds),
        }
    return metrics

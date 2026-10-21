"""Forecast MAE and RMSE summaries."""

import numpy as np
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError


def summarize_predictions(predictions):
    """Summarize MAE and RMSE.

    Input: nonempty DataFrame with split, actual, prediction, and horizon_step
    columns for one split. cv_validation also needs fold.
    Output: dict with split, overall, and by_horizon metrics; cv_validation
    also includes folds and mean_fold (equal weight per fold).
    """
    mae = MeanAbsoluteError()
    rmse = MeanSquaredError(square_root=True)

    # Calc MAE/RMSE
    actual = predictions["actual"].to_numpy()
    prediction = predictions["prediction"].to_numpy()
    metrics = {
        "split": predictions["split"].iloc[0],
        "overall": {
            "n_predictions": len(predictions),
            "mae": float(mae(actual, prediction)),
            "rmse": float(rmse(actual, prediction)),
        },
    }

    # Calc Horizon Metrics
    by_horizon = []
    for step, rows in predictions.groupby("horizon_step"):
        actual = rows["actual"].to_numpy()
        prediction = rows["prediction"].to_numpy()
        by_horizon.append({
            "horizon_step": int(step),
            "n_predictions": len(rows),
            "mae": float(mae(actual, prediction)),
            "rmse": float(rmse(actual, prediction)),
        })
    metrics["by_horizon"] = by_horizon

    if metrics["split"] != "cv_validation":
        return metrics

    # Calc Fold Metrics
    folds = []
    for number, rows in predictions.groupby("fold"):
        actual = rows["actual"].to_numpy()
        prediction = rows["prediction"].to_numpy()
        folds.append({
            "fold": int(number),
            "overall": {
                "n_predictions": len(rows),
                "mae": float(mae(actual, prediction)),
                "rmse": float(rmse(actual, prediction)),
            },
        })
    metrics["folds"] = folds

    # Average Folds
    metrics["mean_fold"] = {
        "mae": float(np.mean([fold["overall"]["mae"] for fold in folds])) if folds else None,
        "rmse": float(np.mean([fold["overall"]["rmse"] for fold in folds])) if folds else None,
    }
    return metrics

"""Forecast MAE and RMSE summaries."""

import numpy as np
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError


def summarize_predictions(predictions):
    """Calculate overall, horizon, and validation-fold MAE and RMSE.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Nonempty rows for one split with actual, prediction, split, and
        horizon_step columns; cv_validation also requires fold.

    Returns
    -------
    dict
        Overall and per-horizon metrics, plus validation-fold metrics when applicable.

    Notes
    -----
    Validation mean_fold gives each fold equal weight.
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

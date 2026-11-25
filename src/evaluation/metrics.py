"""Forecast MAE and RMSE summaries."""

import numpy as np
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError


def directional_errors(actual, prediction):
    """Sum energy errors over supplied rows; callers must state overlap policy."""
    error = np.asarray(actual, dtype=float) - np.asarray(prediction, dtype=float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Directional metrics require nonempty finite actuals and forecasts.")
    return {
        "bias_actual_minus_prediction": float(error.mean()),
        "underprediction_total": float(np.maximum(error, 0).sum()),
        "overprediction_total": float(np.maximum(-error, 0).sum()),
        "underprediction_fraction": float((error > 0).mean()),
        "overprediction_fraction": float((error < 0).mean()),
    }


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
            **directional_errors(actual, prediction),
        },
    }
    metrics["aggregation"] = {
        "scope": "forecast origin/target pairs; overlapping delivery intervals count repeatedly",
        "direction": "residual = actual - prediction; positive means underprediction",
        "units": "same as target; totals sum supplied forecast rows",
        "unique_target_timestamps": int(predictions["target_timestamp"].nunique()),
        "forecast_origins": int(predictions["forecast_origin"].nunique()),
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
            **directional_errors(actual, prediction),
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
                **directional_errors(actual, prediction),
            },
        })
    metrics["folds"] = folds

    # Average Folds
    metrics["mean_fold"] = {
        "mae": float(np.mean([fold["overall"]["mae"] for fold in folds])) if folds else None,
        "rmse": float(np.mean([fold["overall"]["rmse"] for fold in folds])) if folds else None,
    }
    return metrics

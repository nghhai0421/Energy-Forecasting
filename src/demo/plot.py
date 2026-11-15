"""Draw replay forecasts and released actuals using notebook compute."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_demo(spark, settings):
    """Plot the latest forecast, earlier predictions, and horizon errors.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Databricks session using UTC.
    settings : dict
        Demo settings. Run at least one replay step before plotting.

    Returns
    -------
    tuple of matplotlib.figure.Figure and dict
        A 2x2 overview/detail/error figure with current one-hour-ahead scores.
    """
    from pyspark.sql import functions as F

    # Read Results
    tables = settings["tables"]
    dataset = settings["data"]
    timestamp = dataset["timestamp_column"]
    target = dataset["target_column"]
    model_id = settings["model"]["id"]
    timezone = settings["plot"]["timezone"]
    coverage = 0.8
    selected = (F.col("model_id") == model_id) & (F.col("interval_coverage") == coverage)
    forecasts = spark.table(tables["forecasts"]).where(selected)
    origins = forecasts.select("forecast_origin").distinct().orderBy(F.desc("forecast_origin")).limit(2).toPandas()
    latest = pd.to_datetime(origins["forecast_origin"], utc=True).iloc[0]
    simulated_time = latest + pd.Timedelta(dataset["frequency"])
    history_hours = settings["plot"]["history_hours"]
    history_windows = [20 * history_hours, 5 * history_hours, history_hours]
    start = latest - pd.Timedelta(hours=history_windows[0])
    actuals = (
        spark.table(tables["inputs"]).where(F.col(timestamp) >= F.lit(start.to_pydatetime()))
        .select(timestamp, F.col("observed_target").alias(target)).orderBy(timestamp).toPandas()
    )
    current = forecasts.where(F.col("forecast_origin") == F.lit(latest.to_pydatetime())).orderBy("target_timestamp").toPandas()
    steps = min(len(current), int(settings["plot"]["future_hours"] * 4))
    current = current.iloc[:steps]
    comparison = pd.to_datetime(origins["forecast_origin"], utc=True).iloc[-1]
    previous = forecasts.where(F.col("forecast_origin") == F.lit(comparison.to_pydatetime())).orderBy("target_timestamp").toPandas()
    end = latest + pd.Timedelta(hours=settings["plot"]["future_hours"])
    previous = previous.loc[pd.to_datetime(previous["target_timestamp"], utc=True).le(end)]
    metrics = spark.table(tables["metrics"]).where(selected).orderBy("horizon_step").toPandas()
    one_hour = metrics.loc[metrics["horizon_step"].eq(4)]
    status = {"model_id": model_id, "simulated_time": simulated_time,
              "forecast_origin": latest, "n_predictions": 0, "mae": None, "rmse": None}
    if not one_hour.empty:
        row = one_hour.iloc[0]
        status.update(n_predictions=int(row["n_predictions"]), mae=float(row["mae"]), rmse=float(row["rmse"]))

    # Plot Forecasts and Actuals
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()
    actual_dates = pd.to_datetime(actuals[timestamp], utc=True)
    current_dates = pd.to_datetime(current["target_timestamp"], utc=True).dt.tz_convert(timezone)
    previous_dates = pd.to_datetime(previous["target_timestamp"], utc=True).dt.tz_convert(timezone)
    for axis, hours in zip(axes[:3], history_windows):
        window_start = latest - pd.Timedelta(hours=hours)
        visible = actual_dates.ge(window_start)
        axis.plot(actual_dates[visible].dt.tz_convert(timezone), actuals.loc[visible, target], label="Actual", color="black")
        axis.plot(current_dates, current["prediction"], label="Latest forecast", color="tab:blue")
        axis.fill_between(
            current_dates, current["prediction_lower"].to_numpy(), current["prediction_upper"].to_numpy(),
            color="tab:blue", alpha=0.2, label="Prediction interval",
        )
        if comparison != latest:
            axis.plot(previous_dates, previous["prediction"], label="Previously issued forecast", color="tab:orange")
            axis.axvline(comparison.tz_convert(timezone), color="tab:orange", linestyle="--", label="Previous forecast origin")
        axis.axvline(simulated_time.tz_convert(timezone), color="grey", linestyle=":", label="Simulated current time")
        axis.set_xlim(window_start.tz_convert(timezone), end.tz_convert(timezone))
        axis.set_title(f"{model_id}: simulated time {simulated_time.tz_convert(timezone):%Y-%m-%d %H:%M %Z} | {hours:g}h history")
        axis.set_ylabel(target)
        axis.set_xlabel(f"Interval start ({timezone})")
        axis.legend()
        axis.grid(alpha=0.2)

    # Plot Horizon Errors
    if not metrics.empty:
        hours = metrics["horizon_step"] / 4
        axes[3].plot(hours, metrics["mae"], label="MAE")
        axes[3].plot(hours, metrics["rmse"], label="RMSE")
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, "Waiting for actual observations", ha="center", transform=axes[3].transAxes)
    axes[3].set_title("Accuracy by lead time across issued forecasts")
    axes[3].set_xlabel("Lead time from last observed interval start (hours)")
    axes[3].set_ylabel("Error (MWh per quarter-hour)")
    axes[3].grid(alpha=0.2)
    return figure, status

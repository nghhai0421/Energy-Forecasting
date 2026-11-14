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
        Combined forecast/error figure and current one-hour-ahead scores.
    """
    from pyspark.sql import functions as F

    # Read Results
    tables = settings["tables"]
    dataset = settings["data"]
    timestamp = dataset["timestamp_column"]
    target = dataset["target_column"]
    model_id = settings["model"]["id"]
    timezone = settings["plot"]["timezone"]
    forecasts = spark.table(tables["forecasts"]).where(F.col("model_id") == model_id)
    origins = forecasts.select("forecast_origin").distinct().orderBy(F.desc("forecast_origin")).limit(2).toPandas()
    latest = pd.to_datetime(origins["forecast_origin"], utc=True).iloc[0]
    simulated_time = latest + pd.Timedelta(dataset["frequency"])
    start = latest - pd.Timedelta(hours=settings["plot"]["history_hours"])
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
    metrics = spark.table(tables["metrics"]).where(F.col("model_id") == model_id).orderBy("horizon_step").toPandas()
    one_hour = metrics.loc[metrics["horizon_step"].eq(4)]
    status = {"model_id": model_id, "simulated_time": simulated_time,
              "forecast_origin": latest, "n_predictions": 0, "mae": None, "rmse": None}
    if not one_hour.empty:
        row = one_hour.iloc[0]
        status.update(n_predictions=int(row["n_predictions"]), mae=float(row["mae"]), rmse=float(row["rmse"]))

    # Plot Forecasts and Actuals
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].plot(pd.to_datetime(actuals[timestamp], utc=True).dt.tz_convert(timezone), actuals[target], label="Actual", color="black")
    axes[0].plot(pd.to_datetime(current["target_timestamp"], utc=True).dt.tz_convert(timezone), current["prediction"], label="Latest forecast", color="tab:blue")
    if comparison != latest:
        axes[0].plot(pd.to_datetime(previous["target_timestamp"], utc=True).dt.tz_convert(timezone), previous["prediction"], label="Previously issued forecast", color="tab:orange")
        axes[0].axvline(comparison.tz_convert(timezone), color="tab:orange", linestyle="--", label="Previous forecast origin")
    axes[0].axvline(simulated_time.tz_convert(timezone), color="grey", linestyle=":", label="Simulated current time")
    axes[0].set_title(f"{model_id}: simulated time {simulated_time.tz_convert(timezone):%Y-%m-%d %H:%M %Z}")
    axes[0].set_ylabel(target)
    axes[0].set_xlabel(f"Interval start ({timezone})")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Plot Horizon Errors
    if not metrics.empty:
        hours = metrics["horizon_step"] / 4
        axes[1].plot(hours, metrics["mae"], label="MAE")
        axes[1].plot(hours, metrics["rmse"], label="RMSE")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Waiting for actual observations", ha="center", transform=axes[1].transAxes)
    axes[1].set_title("Accuracy by lead time across issued forecasts")
    axes[1].set_xlabel("Lead time from last observed interval start (hours)")
    axes[1].set_ylabel("Error (MWh per quarter-hour)")
    axes[1].grid(alpha=0.2)
    return figure, status

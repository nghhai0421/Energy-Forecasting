"""Initialize, advance, and reset the hourly Chronos-2 demo."""

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml

from src.demo.storage import initialize_storage
from src.models.chronos2.inference import forecast, load_forecaster
from src.models.chronos2.process import load_data, prepare_future_covariates


def load_demo_config(path=None):
    """Load demo settings and resolve source paths.

    Parameters
    ----------
    path : str or pathlib.Path or None, default None
        Demo YAML; defaults to config.yaml beside this module.

    Returns
    -------
    dict
        Settings, shared Chronos data specification, and replay coverage end.
    """
    # Load Settings
    root = Path(__file__).resolve().parents[2]
    path = Path(path) if path is not None else Path(__file__).with_name("config.yaml")
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load((root / settings["experiment_config"]).read_text(encoding="utf-8"))
    settings["data"] = experiment["data"]
    sources = settings["data"]["sources"]
    for source in sources.values():
        for key in ("development", "demo", "path"):
            if key in source:
                source[key] = str(Path(settings["raw_path"]) / Path(source[key]).name)

    # Align Source Coverage
    ends = []
    for name, date_format in (("consumption", "%b %d, %Y %I:%M %p"), ("weather", "%Y-%m-%d %H:%M:%S")):
        source = sources[name]
        dates = pd.read_csv(source["demo"])[source["timestamp_column"]]
        dates = pd.to_datetime(dates, format=date_format)
        dates = dates.dt.tz_localize(source["timezone"], ambiguous="infer").dt.tz_convert("UTC")
        ends.append(dates.iloc[-1])
    settings["replay_end"] = min(ends)
    weather = sources["weather"]
    weather_files = [weather["development"], weather["demo"]]
    weather["development"] = weather_files
    weather["demo"] = weather_files

    # Extend Holidays
    source = sources["holidays"]
    dates = pd.to_datetime(pd.read_csv(source["path"])[source["date_column"]])
    end = settings["replay_end"] + pd.Timedelta(minutes=15 * settings["forecast"]["horizon"])
    missing_years = set(range(int(dates.dt.year.min()), end.year + 1)) - set(dates.dt.year)
    calendar = holidays.Germany(years=sorted(missing_years), subdiv=settings["holiday_subdivision"])
    source["additional_dates"] = [date.isoformat() for date in sorted(calendar)]
    return settings


def initialize_demo(spark, settings, reset=False):
    """Load historical observations into the demo tables.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Databricks session with Unity Catalog access.
    settings : dict
        Settings returned by load_demo_config.
    reset : bool, default False
        Replace historical inputs and clear forecasts.

    Returns
    -------
    dict
        Historical row count and last observed timestamp.
    """
    # Prepare History
    dataset = settings["data"]
    y, X, series, _ = load_data(dataset)
    baseline = X.copy()
    baseline[dataset["target_column"]] = y
    baseline[dataset["timestamp_column"]] = series["timestamp"]
    baseline["observed_target"] = series["target"]

    # Save Tables
    initialize_storage(spark, settings, baseline, reset)
    return {"rows": len(baseline), "forecast_origin": series["timestamp"].iloc[-1]}


def reset_demo(spark, settings):
    """Restore historical inputs and clear demo forecasts.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Databricks session.
    settings : dict
        Demo settings identifying the tables to reset.

    Returns
    -------
    dict
        Restored historical row count and timestamp.
    """
    return initialize_demo(spark, settings, reset=True)


def run_demo(spark, settings, model=None):
    """Release one hour, forecast the next horizon, and append the results.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Databricks session; initialize tables before calling.
    settings : dict
        Settings returned by load_demo_config.
    model : dict or None, default None
        Loaded model from a previous call, reused within the notebook session.

    Returns
    -------
    dict
        Status, simulated time, and reusable loaded model.

    Notes
    -----
    Each successful call advances one hour. If a step fails after writing data,
    reset the demo before replaying it. Use one writer at a time.
    """
    # Read Current Time
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    dataset = settings["data"]
    tables = settings["tables"]
    timestamp = dataset["timestamp_column"]
    frequency = pd.Timedelta(dataset["frequency"])
    origin = spark.sql(f"SELECT max(`{timestamp}`) FROM {tables['inputs']}").first()[0]
    origin = pd.to_datetime(origin, utc=True)
    until = origin + settings["forecast"]["advance_steps"] * frequency
    if until > settings["replay_end"]:
        return {"status": "exhausted", "simulated_time": origin + frequency, "model": model}

    # Release One Hour
    y, X, series, _ = load_data(dataset, period="demo", after=origin, until=until, strict=True)
    expected = pd.date_range(origin + frequency, periods=settings["forecast"]["advance_steps"], freq=frequency)
    if not pd.DatetimeIndex(series["timestamp"]).equals(expected):
        raise ValueError("The next batch must contain the configured contiguous observations.")
    batch = X.copy()
    batch[dataset["target_column"]] = y
    batch[timestamp] = series["timestamp"]
    batch["observed_target"] = series["target"]
    spark.createDataFrame(batch, spark.table(tables["inputs"]).schema).write.mode("append").saveAsTable(tables["inputs"])

    # Read Context
    lookback = settings["forecast"]["lookback"]
    context = spark.sql(
        f"SELECT * FROM {tables['inputs']} ORDER BY `{timestamp}` DESC LIMIT {lookback}"
    ).toPandas().sort_values(timestamp).reset_index(drop=True)
    y = context[dataset["target_column"]]
    features = dataset["covariates"]["historical"] + dataset["covariates"]["future"]
    X = context[features]
    origin = pd.to_datetime(context[timestamp], utc=True).iloc[-1]

    # Predict
    if model is None:
        model = load_forecaster(
            y, X, context_length=lookback, model_path=settings["model"]["model_path"],
            device_map=settings["model"]["device_map"], seed=settings["model"]["seed"],
        )
    horizon = settings["forecast"]["horizon"]
    dates = pd.date_range(origin + frequency, periods=horizon, freq=frequency)
    future_X = prepare_future_covariates(dates, dataset)
    result = forecast(model["forecaster"], y, X, future_X)

    # Save Forecasts
    predictions = pd.DataFrame({
        "split": "demo", "model_id": settings["model"]["id"],
        "model_path": settings["model"]["model_path"], "model_revision": model["revision"],
        "forecast_origin": origin, "target_timestamp": dates,
        "horizon_step": np.arange(1, horizon + 1), "prediction": result["prediction"].to_numpy(),
        "created_at": pd.Timestamp.now(tz="UTC"), "parameters_json": json.dumps(model["parameters"]),
    })
    spark.createDataFrame(predictions, spark.table(tables["forecasts"]).schema).write.mode("append").saveAsTable(tables["forecasts"])
    return {"status": "advanced", "simulated_time": origin + frequency, "model": model}

"""Create the demo Delta tables and evaluation views."""

from datetime import datetime, timezone
from pathlib import Path


def sync_dashboard_config(spark, settings):
    """Publish YAML settings and create dashboard views for all visuals.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Databricks session with table/view creation access.
    settings : dict
        Loaded demo YAML; the configuration table contains one current row.

    Returns
    -------
    None
        Configuration is saved without changing input or forecast rows.
    """
    # Save Settings
    tables = settings["tables"]
    row = (
        settings["model"]["id"], int(settings["plot"]["history_hours"]),
        int(settings["plot"]["future_hours"]), 0.8,
        settings["plot"]["timezone"], 15, datetime.now(timezone.utc),
    )
    schema = "model_id STRING, history_hours BIGINT, future_hours BIGINT, interval_coverage DOUBLE, display_timezone STRING, frequency_minutes BIGINT, updated_at TIMESTAMP"
    spark.createDataFrame([row], schema).write.format("delta").mode("overwrite").saveAsTable(tables["dashboard_config"])

    # Refresh Evaluation Views
    if spark.catalog.tableExists(tables["inputs"]) and spark.catalog.tableExists(tables["forecasts"]):
        timestamp = settings["data"]["timestamp_column"]
        spark.sql(f"""
            CREATE OR REPLACE VIEW {tables['actuals']} AS
            SELECT f.*, a.observed_target AS actual,
                   a.observed_target - f.prediction AS error,
                   abs(a.observed_target - f.prediction) AS absolute_error,
                   power(a.observed_target - f.prediction, 2) AS squared_error
            FROM {tables['forecasts']} f
            JOIN {tables['inputs']} a ON f.target_timestamp = a.`{timestamp}`
            WHERE a.observed_target IS NOT NULL
        """)
        spark.sql(f"""
            CREATE OR REPLACE VIEW {tables['metrics']} AS
            SELECT model_id, interval_coverage, horizon_step, count(*) AS n_predictions,
                   avg(absolute_error) AS mae, sqrt(avg(squared_error)) AS rmse
            FROM {tables['actuals']}
            GROUP BY model_id, interval_coverage, horizon_step
        """)
        queries = Path(__file__).with_name("dashboard.sql").read_text(encoding="utf-8")
        for query in queries.format(**tables).split(";"):
            if query.strip():
                spark.sql(query)


def initialize_storage(spark, settings, baseline, reset=False):
    """Save historical inputs and create empty forecasts and evaluation views.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Session with table/view creation access in the configured schema.
    settings : dict
        Demo settings containing table names and the timestamp column.
    baseline : pandas.DataFrame
        Prepared historical inputs and original observed targets.
    reset : bool, default False
        Overwrite inputs and recreate the empty forecast table.

    Returns
    -------
    None
        Tables and views are saved in Unity Catalog.

    Raises
    ------
    ValueError
        If demo tables already exist and reset is false.
    """
    # Save History
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    tables = settings["tables"]
    if not reset and any(spark.catalog.tableExists(tables[name]) for name in ("inputs", "forecasts")):
        raise ValueError("Demo tables already exist. Skip initialization or use reset_demo().")
    (
        spark.createDataFrame(baseline).replace(float("nan"), None).write.format("delta")
        .option("delta.columnMapping.mode", "name").option("overwriteSchema", "true")
        .mode("overwrite").saveAsTable(tables["inputs"])
    )

    # Create Forecast Table
    command = "CREATE OR REPLACE TABLE" if reset else "CREATE TABLE"
    spark.sql(f"""
        {command} {tables['forecasts']} (
            split STRING, model_id STRING, model_path STRING, model_revision STRING,
            forecast_origin TIMESTAMP, target_timestamp TIMESTAMP, horizon_step BIGINT,
            prediction DOUBLE, prediction_lower DOUBLE, prediction_upper DOUBLE,
            interval_coverage DOUBLE, created_at TIMESTAMP, parameters_json STRING
        ) USING DELTA
    """)

    sync_dashboard_config(spark, settings)

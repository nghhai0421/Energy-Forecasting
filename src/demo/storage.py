"""Create the demo Delta tables and evaluation views."""


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
    """
    # Save History
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    tables = settings["tables"]
    timestamp = settings["data"]["timestamp_column"]
    (
        spark.createDataFrame(baseline).replace(float("nan"), None).write.format("delta")
        .option("delta.columnMapping.mode", "name").option("overwriteSchema", "true")
        .mode("overwrite" if reset else "errorifexists").saveAsTable(tables["inputs"])
    )

    # Create Forecast Table
    command = "CREATE OR REPLACE TABLE" if reset else "CREATE TABLE"
    spark.sql(f"""
        {command} {tables['forecasts']} (
            split STRING, model_id STRING, model_path STRING, model_revision STRING,
            forecast_origin TIMESTAMP, target_timestamp TIMESTAMP, horizon_step BIGINT,
            prediction DOUBLE, created_at TIMESTAMP, parameters_json STRING
        ) USING DELTA
    """)

    # Create Evaluation Views
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
        SELECT model_id, horizon_step, count(*) AS n_predictions,
               avg(absolute_error) AS mae, sqrt(avg(squared_error)) AS rmse
        FROM {tables['actuals']}
        GROUP BY model_id, horizon_step
    """)

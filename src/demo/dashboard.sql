-- Status and YAML settings
CREATE OR REPLACE VIEW {dashboard_status} AS
WITH ranked_origins AS (
    SELECT model_id, interval_coverage, forecast_origin,
           dense_rank() OVER (
               PARTITION BY model_id, interval_coverage ORDER BY forecast_origin DESC
           ) AS origin_rank
    FROM (SELECT DISTINCT model_id, interval_coverage, forecast_origin FROM {forecasts})
), origins AS (
    SELECT model_id, interval_coverage,
           max(CASE WHEN origin_rank = 1 THEN forecast_origin END) AS latest_origin,
           max(CASE WHEN origin_rank = 2 THEN forecast_origin END) AS previous_origin
    FROM ranked_origins
    GROUP BY model_id, interval_coverage
), clock AS (
    SELECT max(timestamp_utc) AS latest_observation FROM {inputs}
)
SELECT c.*, o.latest_origin, o.previous_origin, clock.latest_observation,
       timestampadd(MINUTE, cast(c.frequency_minutes AS INT), clock.latest_observation) AS simulated_time,
       from_utc_timestamp(
           timestampadd(MINUTE, cast(c.frequency_minutes AS INT), clock.latest_observation),
           c.display_timezone
       ) AS simulated_time_local,
       from_utc_timestamp(o.latest_origin, c.display_timezone) AS latest_origin_local,
       from_utc_timestamp(o.previous_origin, c.display_timezone) AS previous_origin_local,
       timestampadd(HOUR, -cast(c.history_hours AS INT), clock.latest_observation) AS window_start,
       timestampadd(HOUR, cast(c.future_hours AS INT), clock.latest_observation) AS window_end
FROM {dashboard_config} c
CROSS JOIN clock
LEFT JOIN origins o ON o.model_id = c.model_id AND o.interval_coverage = c.interval_coverage;

-- Combined actual and forecast chart
CREATE OR REPLACE VIEW {dashboard_chart} AS
WITH points AS (
    SELECT a.timestamp_utc, a.observed_target AS actual,
           cast(NULL AS DOUBLE) AS latest_prediction,
           cast(NULL AS DOUBLE) AS previous_prediction,
           cast(NULL AS DOUBLE) AS prediction_lower,
           cast(NULL AS DOUBLE) AS prediction_upper
    FROM {inputs} a CROSS JOIN {dashboard_status} s
    WHERE a.timestamp_utc BETWEEN s.window_start AND s.latest_observation

    UNION ALL

    SELECT f.target_timestamp, NULL, f.prediction, NULL,
           f.prediction_lower, f.prediction_upper
    FROM {forecasts} f JOIN {dashboard_status} s
      ON f.model_id = s.model_id AND f.interval_coverage = s.interval_coverage
     AND f.forecast_origin = s.latest_origin
    WHERE f.target_timestamp BETWEEN s.window_start AND s.window_end

    UNION ALL

    SELECT f.target_timestamp, NULL, NULL, f.prediction, NULL, NULL
    FROM {forecasts} f JOIN {dashboard_status} s
      ON f.model_id = s.model_id AND f.interval_coverage = s.interval_coverage
     AND f.forecast_origin = s.previous_origin
    WHERE f.target_timestamp BETWEEN s.window_start AND s.window_end
)
SELECT p.timestamp_utc,
       from_utc_timestamp(p.timestamp_utc, s.display_timezone) AS timestamp_local,
       max(p.actual) AS actual,
       max(p.latest_prediction) AS latest_prediction,
       max(p.previous_prediction) AS previous_prediction,
       max(p.prediction_lower) AS prediction_lower,
       max(p.prediction_upper) AS prediction_upper,
       s.model_id, s.interval_coverage,
       s.latest_origin_local, s.previous_origin_local, s.simulated_time_local
FROM points p CROSS JOIN {dashboard_status} s
GROUP BY p.timestamp_utc, s.display_timezone, s.model_id, s.interval_coverage,
         s.latest_origin_local, s.previous_origin_local, s.simulated_time_local;

-- Accuracy for the configured model and fixed interval level
CREATE OR REPLACE VIEW {dashboard_metrics} AS
SELECT m.*, m.horizon_step * s.frequency_minutes / 60.0 AS lead_hours
FROM {metrics} m JOIN {dashboard_config} s
  ON m.model_id = s.model_id AND m.interval_coverage = s.interval_coverage;

# Chronos-2 demo implementation

## Relevant files

The demo runs repository Python code on Databricks compute. Its notebook provides replay controls and charts.

```text
src/demo/
  config.yaml                          # Volume paths, tables, model and replay settings
  run.py                               # Load settings, initialize, advance, reset
  storage.py                           # Create Delta tables and evaluation views
  plot.py                              # Actuals, forecasts and error charts
  dashboard.sql                        # YAML-backed AI/BI status, chart and metric views
src/models/chronos2/
  process.py                           # Existing preparation with batch selection
  inference.py                         # Existing model loading and prediction
preprocessing/add_seasonal_features.py  # Existing calendar features
configs/experiments/chronos2.yaml        # Shared target and covariate specification
requirements.txt                       # Pip packages mirrored from environment.yml
notebooks/energy_demo.py                # Databricks controls and visualization
```

## Data and defaults

Source files remain in `/Volumes/energy_prediction/default/raw_data`. All models can select their required columns and context from the prepared input table.

| Setting | Value |
|---|---|
| Initial history | 2023–2024 consumption and weather |
| Replay | 2025 consumption and weather |
| Target | `Residual load [MWh]` |
| Frequency | 15 minutes |
| Advance per call | One hour: four observations |
| Lookback | Seven days: 672 observations |
| Forecast | Next 24 hours: 96 observations |
| Main chart | Previous three hours of actuals and next three hours of forecasts |

The historical baseline uses the existing causal prior-mean imputation. Original targets are preserved in `observed_target` for scoring. Incoming batches use strict validation.

Weather uses fixed UTC+02:00 and consumption uses Europe/Berlin. Both weather files are read to cover the year boundary; only released observations enter preparation. Replay stops at the common coverage end. Missing holiday years are generated using the existing German/Baden-Württemberg calendar.

## Tables

Initialization fills only the input table. Forecasts and evaluation results start empty.

| Object in `energy_prediction.default` | Contents |
|---|---|
| `demo_model_inputs` | UTC timestamp, prepared target/covariates, original target |
| `demo_forecasts` | Model ID/path/revision, origin, target timestamp, horizon step, median prediction, 10th/90th percentile bounds, nominal coverage, creation time, parameters |
| `demo_forecast_actuals` | View joining forecasts to released actuals and calculating errors |
| `demo_metrics` | View reporting MAE, RMSE and scored count by model, interval level and horizon |
| `dashboard_config` | One current row containing YAML model, window, interval and timezone settings |
| `dashboard_status` | YAML settings, current simulated time and latest/previous matching forecast origins |
| `dashboard_chart` | Actuals, latest/previous predictions and latest bounds within the YAML display window |
| `dashboard_metrics` | Accuracy for the YAML-selected model and fixed interval level |

Issued forecasts remain unchanged when actuals arrive. Metrics exclude unavailable actuals and group overlapping forecasts by lead time. Values retain MWh per quarter-hour units.

The demo uses a fixed central 80% prediction interval from the 10th and 90th percentiles. Bounds are saved in `prediction_lower` and `prediction_upper`; the point prediction remains the median. Chronos produces all three quantiles in one native pipeline call. The interval level is recorded internally as 0.8. There are no empirical-coverage calculations, coverage cards or coverage charts.

## Core workflow

`run_demo(spark, settings, model=None)` performs one sequential step:

1. Read the latest input timestamp.
2. Select and prepare the next hour from the replay files.
3. Append those four observations to the input table.
4. Read the latest 672 input rows.
5. Generate future calendar/holiday features and predict 96 steps.
6. Append the forecast rows.

The first forecast follows the first released 2025 hour. Repeated calls resume from the latest input timestamp. Pass the returned model to reuse loaded weights in the current session.

Use one writer and complete each step before running the next. Writes are plain appends. If a step fails after writing data, reset before replaying it. `reset_demo()` restores historical inputs and recreates the empty forecast table; evaluation views then return no results.

`forecast_origin` is the last observed interval's start timestamp. The first prediction is one interval later. Simulated current time is the last interval's end; charts display Europe/Berlin time.

## Run in Databricks

Open `notebooks/energy_demo.py` in the repository's Databricks Git folder. Its setup assumes the notebook working directory is `notebooks/`. Attach compute with volume access, table/view creation permissions, and the dependencies listed in the notebook.

The notebook runs `%pip install -r ../requirements.txt` and restarts Python before importing the demo. The requirements file mirrors all package dependencies and pip versions/index settings from `environment.yml`. Databricks supplies Python, pip, Spark, and Delta. Keep the two dependency files synchronized when packages change.

Run setup and initialize once, then repeat the advance cell. Skip initialization when resuming an existing demo. A minimal equivalent is:

```python
from src.demo.run import load_demo_config, initialize_demo, run_demo
from src.demo.plot import plot_demo
import matplotlib.pyplot as plt

settings = load_demo_config()
initialize_demo(spark, settings)
result = run_demo(spark, settings)
figure, status = plot_demo(spark, settings)
print(status)
plt.show()
```

Subsequent calls use `run_demo(spark, settings, model=result["model"])`. Use the notebook's reset cell to start over, including after changing model or forecast settings.

The Replay Hours cell saves each completed step's figure before clearing the displayed output. Set `replay_steps` to the desired number of simulated hours. PNGs are written at 150 DPI with fixed figure dimensions to `<raw_path>/demo_frames/<UTC recording ID>/forecast_00001.png`, `forecast_00002.png`, and so on. Each loop execution uses a new folder; an exhausted source creates no duplicate final frame. The volume requires write permission. Frames persist after the Python session and can be downloaded in filename order for video creation. Resetting the demo tables leaves saved frames intact.

The main chart combines recent actuals (black), the latest forecast (blue), and the previous origin's forecast (orange). A dashed orange line marks the previous origin; a dotted grey line marks simulated current time. Both forecasts use the configured future display window. A second chart shows errors by horizon. One-hour-ahead score cards use `horizon_step=4`.

The latest forecast has a light blue band labeled "Prediction interval". The earlier uncertainty update requires a reset if the base forecast table still lacks interval columns. The YAML/dashboard configuration update adds a table and views and does not require a reset of already compatible base tables.

These are notebook charts using notebook compute. Analytics → Dashboards is the separate AI/BI interface; it can query the same Delta tables/views through a SQL warehouse. That dashboard has not been created by this implementation.

## Connect the existing AI/BI dashboard to YAML

The notebook syncs settings during setup and each replay call. To apply a YAML edit without advancing the demo, run:

```python
settings = load_demo_config()
sync_dashboard_config(spark, settings)
```

The sync writes one current configuration row and refreshes the three dashboard views from `src/demo/dashboard.sql`. Refresh the dashboard afterward. YAML edits alone do not update the database; existing Python settings must be reloaded. The views use the table bindings in the YAML when created. The configuration is shared by this demo's dashboard users.

Remove the Model ID and Window input controls and their dataset parameters. Connect the existing dashboard visuals as follows; they all read YAML values through the views.

| Dashboard dataset | Source | Use |
|---|---|---|
| Status | `energy_prediction.default.dashboard_status` | Model, simulated time and origin cards; marker positions |
| Forecast chart | `energy_prediction.default.dashboard_chart` | `timestamp_local`, `actual`, `latest_prediction`, `previous_prediction`, lower/upper bounds |
| Accuracy | `energy_prediction.default.dashboard_metrics` | Score cards at `horizon_step=4`; error chart by `lead_hours` |

Use local timestamp columns without applying another timezone conversion. Keep the stored metric values at their existing model/interval/horizon grain. Display MAE, RMSE and scored counts, and label the shaded band "Prediction interval".

Paste this instruction into the existing dashboard's Genie authoring panel after syncing the configuration:

> Preserve the layout and remove the Model ID and Window controls, their parameters, all hardcoded model/window filters, and all coverage cards/charts. Replace the status dataset with SELECT * FROM energy_prediction.default.dashboard_status, the main chart dataset with SELECT * FROM energy_prediction.default.dashboard_chart ORDER BY timestamp_utc, and the accuracy/error dataset with SELECT * FROM energy_prediction.default.dashboard_metrics ORDER BY horizon_step. These views already apply YAML settings to every visual. Plot actual in black, latest_prediction in blue, previous_prediction in orange, and prediction_lower/prediction_upper as the latest forecast band labeled Prediction interval. Use timestamp_local directly. Use simulated_time_local, latest_origin_local and model_id for status cards and previous_origin_local for the origin marker. Accuracy cards select horizon_step=4 and show mae, rmse and n_predictions. Plot mae/rmse against lead_hours. Do not display coverage percentages or average RMSE across horizons. Do not add replacement input controls or modify the source tables. Keep unavailable forecasts and scores empty.

Changing the display window affects the main chart; accuracy charts summarize all scored forecasts for the selected model and interval. The current runner remains Chronos-2; changing the model ID labels/selects its recorded results and does not select a different model implementation.

## After pulling the update

Run the dependency cell, Python restart and Load Project. Setup automatically writes the YAML configuration and loads `src/demo/dashboard.sql` to create the dashboard views. Skip initialization/reset when the existing base tables already have prediction bounds.

Add the three dashboard views to the existing dashboard and apply the Genie instruction above once. Refresh the dashboard, run Advance One Hour, then refresh again. Later YAML edits need only a settings reload, configuration sync and dashboard refresh; the dashboard SQL is loaded by the notebook.

## Verification

Use small manual checks for baseline counts, four-row advances, 96-row forecasts, correct time boundaries, and reset. Existing Chronos preparation/inference checks cover shared behavior; no dedicated demo test file is required.

Delta persistence and pretrained execution require a Databricks smoke run. Local Matplotlib rendering previously crashed even for an empty chart, so chart rendering also needs verification on Databricks.

YAML-to-row synchronization and dashboard SQL logic are checked locally with synthetic data for model/window changes, origin selection, band alignment, timezone conversion and scored counts. SQL timestamp functions are adapted to SQLite for this check; Databricks SQL execution and edits to the live dashboard remain unverified because browser access was unavailable.

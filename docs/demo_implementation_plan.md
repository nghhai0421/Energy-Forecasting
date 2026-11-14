# Chronos-2 demo implementation

## Relevant files

The demo runs repository Python code on Databricks compute. Its notebook provides replay controls and charts.

```text
src/demo/
  config.yaml                          # Volume paths, tables, model and replay settings
  run.py                               # Load settings, initialize, advance, reset
  storage.py                           # Create Delta tables and evaluation views
  plot.py                              # Actuals, forecasts and error charts
src/models/chronos2/
  process.py                           # Existing preparation with batch selection
  inference.py                         # Existing model loading and prediction
preprocessing/add_seasonal_features.py  # Existing calendar features
configs/experiments/chronos2.yaml        # Shared target and covariate specification
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
| Main chart | Previous 24 hours of actuals and next three hours of forecasts |

The historical baseline uses the existing causal prior-mean imputation. Original targets are preserved in `observed_target` for scoring. Incoming batches use strict validation.

Weather uses fixed UTC+02:00 and consumption uses Europe/Berlin. Both weather files are read to cover the year boundary; only released observations enter preparation. Replay stops at the common coverage end. Missing holiday years are generated using the existing German/Baden-Württemberg calendar.

## Tables

Initialization fills only the input table. Forecasts and evaluation results start empty.

| Object in `energy_prediction.default` | Contents |
|---|---|
| `demo_model_inputs` | UTC timestamp, prepared target/covariates, original target |
| `demo_forecasts` | Model ID/path/revision, origin, target timestamp, horizon step, prediction, creation time, parameters |
| `demo_forecast_actuals` | View joining forecasts to released actuals and calculating errors |
| `demo_metrics` | View reporting MAE, RMSE and scored count by model and horizon |

Issued forecasts remain unchanged when actuals arrive. Metrics exclude unavailable actuals and group overlapping forecasts by lead time. Values retain MWh per quarter-hour units.

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

`environment.yml` already lists `holidays` for the local Conda environment. The notebook installs it with `%pip` and restarts Python before importing the demo, since Databricks does not automatically apply that Conda file.

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

Charts show the latest forecast, the previous origin's forecast against released actuals, and errors by horizon. One-hour-ahead score cards use `horizon_step=4`. All charts use notebook compute; an AI/BI dashboard and SQL warehouse are optional later.

## Verification

Use small manual checks for baseline counts, four-row advances, 96-row forecasts, correct time boundaries, and reset. Existing Chronos preparation/inference checks cover shared behavior; no dedicated demo test file is required.

Delta persistence and pretrained execution require a Databricks smoke run. Local Matplotlib rendering previously crashed even for an empty chart, so chart rendering also needs verification on Databricks.

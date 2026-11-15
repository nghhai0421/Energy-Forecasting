# Databricks notebook source
# MAGIC %md
# MAGIC # Energy forecasting replay
# MAGIC Open this notebook inside a Databricks Git folder containing the repository.
# MAGIC Attach Python compute with Unity Catalog volume/table access. Use one writer.
# MAGIC The first cells install requirements.txt and restart Python. The file mirrors
# MAGIC the packages in environment.yml; Databricks supplies Python, Spark, and Delta.
# MAGIC Run setup when starting a fresh Python session. Saved tables remain available.
# MAGIC The first model load requires access to the configured pretrained checkpoint.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Load Project
import sys
from pathlib import Path

project_root = Path.cwd().resolve().parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from src.demo.run import initialize_demo, load_demo_config, reset_demo, run_demo
from src.demo.plot import plot_demo
from src.demo.storage import sync_dashboard_config

settings = load_demo_config()
model = None
spark.conf.set("spark.sql.session.timeZone", "UTC")
sync_dashboard_config(spark, settings)

# COMMAND ----------

# MAGIC %md
# MAGIC Run initialization once for a new demo. Existing tables cause a clear error.
# MAGIC For a resumed session, skip initialization and run the next cell.
# MAGIC Each replay call releases one hour and then forecasts. If a call fails
# MAGIC after writing observations, use Reset Demo before starting again.
# MAGIC After pulling the uncertainty-band update, reset existing demo tables once
# MAGIC to create the lower/upper prediction columns before advancing again.

# COMMAND ----------

# Initialize Tables
initialize_demo(spark, settings)

# COMMAND ----------

# Advance One Hour
result = run_demo(spark, settings, model)
model = result["model"]
print({key: value for key, value in result.items() if key != "model"})
figure, status = plot_demo(spark, settings)
print(status)
plt.show()
plt.close(figure)

# COMMAND ----------

# MAGIC %md
# MAGIC Optional accelerated replay: set `replay_steps` above zero and run this cell.
# MAGIC Each iteration releases one simulated hour. Charts replace the prior display.
# MAGIC Each chart is saved as a numbered PNG in a new demo_frames folder in the
# MAGIC source volume. The files remain available for video creation after the session.

# COMMAND ----------

# Replay Hours
from datetime import datetime, timezone
from IPython.display import clear_output

replay_steps = 168  # Seven simulated days, advancing one hour per step.
recording_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
frame_folder = Path(settings["raw_path"]) / "demo_frames" / recording_id
if replay_steps > 0:
    frame_folder.mkdir(parents=True)

for step in range(1, replay_steps + 1):
    result = run_demo(spark, settings, model)
    model = result["model"]
    if result["status"] == "exhausted":
        print("Replay source exhausted.")
        break
    figure, status = plot_demo(spark, settings)
    frame_path = frame_folder / f"forecast_{step:05d}.png"
    figure.savefig(frame_path, dpi=150)
    clear_output(wait=True)
    print(status)
    print(f"Saved frame {step}: {frame_path}")
    plt.show()
    plt.close(figure)

# COMMAND ----------

# Inspect Scores
display(spark.table(settings["tables"]["metrics"]).orderBy("model_id", "horizon_step"))

# COMMAND ----------

# MAGIC %md
# MAGIC Reset restores historical inputs and clears demo forecasts. Set
# MAGIC `reset_requested = True` explicitly to run it. Source CSVs remain available.

# COMMAND ----------

# Reset Demo
reset_requested = False
if reset_requested:
    print(reset_demo(spark, settings))
    model = None

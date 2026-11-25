"""Upgrade a saved run's output layout without fitting or changing forecasts."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.outputs.output import save_results
from src.outputs.plot import create_forecast_plots
from src.config import portable_paths


def standardize_run(run_path):
    path = Path(run_path)
    manifest_path = path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Only completed snapshots can be standardized.")
    before, frames = {}, []
    for split in ("cv_validation", "final_test"):
        source = path / "predictions" / f"{split}.csv"
        if source.exists():
            before[split] = hashlib.sha256(source.read_bytes()).hexdigest()
            frames.append(pd.read_csv(source, float_precision="round_trip"))
    results = {"predictions": pd.concat(frames, ignore_index=True),
               "split_manifest": json.loads((path / "artifacts/split_manifest.json").read_text(encoding="utf-8"))}
    selection = path / "artifacts/selected_parameters.json"
    if selection.exists():
        results["selected_parameters"] = json.loads(selection.read_text(encoding="utf-8"))
    manifest.setdefault("summary", {}).update(save_results(path, results))
    for split in ("cv_validation", "final_test"):
        rows = results["predictions"].loc[results["predictions"].split == split]
        if rows.empty:
            continue
        table_path = path / "metrics" / f"{split}_evaluation.csv"
        key = "fold" if split == "cv_validation" else "forecast_origin"
        scores = []
        for value, group in rows.groupby(key):
            errors = group.actual - group.prediction
            scores.append({key: value, "test_MeanAbsoluteError": float(errors.abs().mean()),
                           "test_MeanSquaredError": float(np.sqrt((errors ** 2).mean()))})
        table = pd.DataFrame(scores)
        if table_path.exists():
            previous = pd.read_csv(table_path)
            table = previous.drop(columns=["test_MeanAbsoluteError", "test_MeanSquaredError"], errors="ignore").merge(table, on=key, validate="one_to_one")
        else:
            # Historical per-origin timings cannot be recovered from forecasts.
            table["fit_time"] = None
            table["pred_time"] = None
        table.to_csv(table_path, index=False)
    for rows in frames:
        saved = pd.read_csv(path / "predictions" / f"{rows.split.iloc[0]}.csv", float_precision="round_trip")
        for column in ("actual", "prediction"):
            if not np.array_equal(rows[column], saved[column], equal_nan=True):
                raise RuntimeError(f"Standardization changed {column} values.")
    audit_path = path / "artifacts/output_standardization.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {
        "original_prediction_sha256": before,
        "note": "Layout/schema/derived-metric upgrade only; actual and prediction values verified unchanged. Original verification artifacts describe the historical run.",
    }
    audit["standardized_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit["hash_scope"] = "Local file bytes at migration time; Git line-ending conversion changes these byte hashes. Forecast values were compared separately."
    audit["historical_timing"] = "Existing measurements retained; unavailable per-origin timings are blank, not estimated."
    audit["prediction_sha256"] = {split: hashlib.sha256((path / "predictions" / f"{split}.csv").read_bytes()).hexdigest() for split in before}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest["output_schema_version"] = 2
    manifest["output_standardization"] = "artifacts/output_standardization.json"
    if "checkpoint_created_before_demo" in manifest:
        manifest["checkpoint_created_before_demo"] = portable_paths((path / "checkpoints/model.joblib").resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Saved CV labels provide observed history for the first test forecast plot.
    series = results["predictions"][["target_timestamp", "actual"]].drop_duplicates("target_timestamp")
    series = series.rename(columns={"target_timestamp": "timestamp", "actual": "target"})
    series["timestamp"] = pd.to_datetime(series.timestamp, utc=True)
    create_forecast_plots(path, series.sort_values("timestamp").reset_index(drop=True))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(standardize_run(parser.parse_args().run))


if __name__ == "__main__":
    main()

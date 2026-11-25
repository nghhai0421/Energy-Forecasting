"""Compare final tests on identical targets and forecast origin/delivery pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.config import portable_paths
from src.evaluation.metrics import directional_errors

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEYS = ["forecast_origin", "target_timestamp", "horizon_step"]


def load_run(path):
    path = Path(path)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Only completed runs can be compared.")
    data = manifest.get("data", manifest.get("training_contract", {}).get("data", {}))
    forecast = manifest.get("forecast", manifest.get("training_contract", {}).get("forecast", {}))
    rows = pd.read_csv(path / "predictions/final_test.csv")
    for column in ("forecast_origin", "target_timestamp", "cutoff_timestamp"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="raise")
    if rows.empty or not rows.split.eq("final_test").all() or rows[KEYS].isna().any().any():
        raise ValueError("Expected nonempty final-test rows with complete keys.")
    if rows.duplicated(KEYS).any() or rows.duplicated(["forecast_origin", "target_timestamp"]).any():
        raise ValueError("Duplicate forecast origin/target pairs.")
    if not np.isfinite(rows[["actual", "prediction"]]).all().all():
        raise ValueError("Non-finite actuals or predictions.")
    frequency = pd.Timedelta(data["frequency"])
    expected = rows.forecast_origin + (rows.horizon_step - 1) * frequency
    if not rows.target_timestamp.eq(expected).all() or not rows.cutoff_timestamp.eq(rows.forecast_origin - frequency).all():
        raise ValueError("Forecast timestamps disagree with horizon steps or cutoffs.")
    if not rows.groupby("forecast_origin").horizon_step.apply(
            lambda steps: sorted(steps) == list(range(1, forecast["horizon"] + 1))).all():
        raise ValueError("Expected complete forecast horizons.")
    return manifest, data, forecast, rows


def score(rows, column):
    error = rows.actual.to_numpy() - rows[column].to_numpy()
    return {"n_predictions": len(rows), "unique_delivery_intervals": int(rows.target_timestamp.nunique()),
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            **directional_errors(rows.actual, rows[column])}


def compare_runs(gbt_run, chronos_run):
    """Fail on mismatched targets/keys; save comparison beside the GBT results."""
    output = Path(gbt_run)
    left_manifest, left_data, left_forecast, left = load_run(gbt_run)
    right_manifest, right_data, right_forecast, right = load_run(chronos_run)
    for key in ("target_column", "frequency"):
        if left_data[key] != right_data[key]:
            raise ValueError(f"Different {key}: rerun on a common target and frequency.")
    for key in ("horizon", "step_length"):
        if left_forecast[key] != right_forecast[key]:
            raise ValueError(f"Different {key}: rerun on a common forecast schedule.")
    if "[MWh]" not in left_data["target_column"]:
        raise ValueError("This energy comparison requires an explicit MWh target.")
    paired = left[KEYS + ["actual", "prediction"]].merge(
        right[KEYS + ["actual", "prediction"]], on=KEYS, how="outer", validate="one_to_one",
        suffixes=("_gbt", "_chronos2"), indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Forecast origin/target pairs differ; no silent intersection is allowed.")
    if not np.allclose(paired.actual_gbt, paired.actual_chronos2, rtol=0, atol=1e-8):
        raise ValueError("Actual labels differ; reconcile target sources before comparing.")
    paired = paired.drop(columns=["_merge", "actual_chronos2"]).rename(columns={"actual_gbt": "actual"})
    paired = paired.sort_values(KEYS).reset_index(drop=True)
    for model in ("gbt", "chronos2"):
        error = paired.actual - paired[f"prediction_{model}"]
        paired[f"residual_{model}"] = error
        paired[f"underprediction_{model}_mwh"] = error.clip(lower=0)
        paired[f"overprediction_{model}_mwh"] = (-error).clip(lower=0)
    duration = pd.Timedelta(left_data["frequency"]) * left_forecast["horizon"]
    origins = pd.date_range(paired.forecast_origin.min(), paired.forecast_origin.max(), freq=duration)
    delivery = paired.loc[paired.forecast_origin.isin(origins)].copy()
    if delivery.empty or delivery.target_timestamp.duplicated().any():
        raise ValueError("Cannot construct non-overlapping delivery forecasts.")
    if set(delivery.target_timestamp) != set(paired.target_timestamp):
        raise ValueError("Non-overlapping origins do not cover all delivery intervals.")
    summary = {
        "target": left_data["target_column"], "units": f"MWh per {left_data['frequency']} interval",
        "forecast": left_forecast, "matched_forecast_origins": int(paired.forecast_origin.nunique()),
        "start_utc": paired.target_timestamp.min().isoformat(),
        "last_interval_start_utc": paired.target_timestamp.max().isoformat(),
        "all_forecast_pairs": {model: score(paired, f"prediction_{model}") for model in ("gbt", "chronos2")},
        "non_overlapping_delivery": {model: score(delivery, f"prediction_{model}") for model in ("gbt", "chronos2")},
        "delivery_policy": "Select origins spaced by one full horizon, anchored at the first test origin; one forecast per delivery interval.",
        "pair_policy": "All hourly origin/horizon pairs; summed errors here count overlapping deliveries repeatedly and are not monthly energy volumes.",
        "limits": "Matched final-test target, labels, dates and origins. Training, validation, features, lookback and observation availability differ. Exploratory comparison; no cost claim or published grid-load benchmark for this residual-load target.",
        "input_hash_normalization": "CSV bytes with CRLF converted to LF, so hashes survive Git line-ending conversion.",
        "inputs": {label: {"run_id": manifest["run_id"], "path": portable_paths(Path(path).resolve()),
                    "prediction_sha256": hashlib.sha256((Path(path) / "predictions/final_test.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
                   for label, path, manifest in (("gbt", gbt_run, left_manifest), ("chronos2", chronos_run, right_manifest))},
    }
    paired.to_csv(output / "predictions/chronos2_matched_pairs.csv", index=False)
    delivery.to_csv(output / "predictions/chronos2_matched_delivery.csv", index=False)
    (output / "metrics/chronos2_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([{"horizon_step": int(step), "model": model, **score(rows, f"prediction_{model}")}
                  for step, rows in paired.groupby("horizon_step") for model in ("gbt", "chronos2")]
                 ).to_csv(output / "metrics/chronos2_comparison_by_horizon.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, title in zip(axes, ("underprediction_total", "overprediction_total"), ("Underprediction", "Overprediction")):
        axis.bar(["GBT", "Chronos-2"], [summary["non_overlapping_delivery"][model][key] for model in ("gbt", "chronos2")], color=["tab:orange", "tab:blue"])
        axis.set(title=title, ylabel="MWh across unique delivery intervals")
    figure.suptitle("January 2025 residual load — one forecast per interval")
    figure.tight_layout()
    figure.savefig(output / "plots/chronos2_directional_comparison.png", dpi=160)
    plt.close(figure)
    left_manifest.setdefault("comparisons", {})["chronos2"] = {
        "metrics": "metrics/chronos2_comparison.json",
        "matched_pairs": "predictions/chronos2_matched_pairs.csv",
        "unique_delivery": "predictions/chronos2_matched_delivery.csv",
        "plot": "plots/chronos2_directional_comparison.png",
    }
    (output / "run_manifest.json").write_text(json.dumps(left_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbt-run", type=Path, required=True)
    parser.add_argument("--chronos-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.gbt_run, args.chronos_run), indent=2))


if __name__ == "__main__":
    main()

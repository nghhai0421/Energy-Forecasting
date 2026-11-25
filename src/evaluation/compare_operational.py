"""Compare saved quarter-hour test predictions with SMARD load forecast exports.

This is a retrospective benchmark on identical delivery intervals. It does not
establish that forecast issue times, revisions or input availability were matched.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from src.config import portable_paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_smard(path, name):
    """Parse English SMARD CSVs and disambiguate repeated local autumn hours."""
    raw = pd.read_csv(path, sep=";", dtype=str)
    dates = pd.to_datetime(raw["Start date"], format="%b %d, %Y %I:%M %p")
    dates = dates.dt.tz_localize("Europe/Berlin", ambiguous="infer").dt.tz_convert("UTC")
    values = raw["grid load [MWh] Original resolutions"].str.replace(",", "", regex=False)
    values = pd.to_numeric(values.replace("-", np.nan), errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("SMARD timestamps must be valid, unique and ordered in UTC.")
    if not dates.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("SMARD export has gaps in its quarter-hour grid.")
    return pd.DataFrame({"target_timestamp": dates, name: values})


def score(rows, column):
    error = rows.actual_mwh.to_numpy() - rows[column].to_numpy()
    return {
        "intervals": len(rows), "mae_mwh": float(np.abs(error).mean()),
        "rmse_mwh": float(np.sqrt(np.mean(error ** 2))),
        "bias_actual_minus_forecast_mwh": float(error.mean()),
        "underprediction_mwh": float(np.maximum(error, 0).sum()),
        "overprediction_mwh": float(np.maximum(-error, 0).sum()),
        "absolute_error_mwh": float(np.abs(error).sum()),
        "overprediction_fraction": float((error < 0).mean()),
    }


def compare(predictions_path, actual_path, forecast_path, output=None, plot_path=None):
    output = Path(output) if output is not None else Path(predictions_path).parent.parent
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("artifacts", "metrics", "predictions", "plots"):
        (output / folder).mkdir(exist_ok=True)
    plot_path = Path(plot_path) if plot_path is not None else output / "plots/transnetbw_monthly_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(predictions_path)
    # Legacy run files may contain a baseline; it is outside this comparison.
    predictions = predictions.drop(columns=["weekly_baseline"], errors="ignore")
    manifest_path = Path(predictions_path).parent.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    delay = manifest.get("training_contract", {}).get("data", {}).get("observation_delay_minutes")
    if not predictions["split"].eq("final_test").all():
        raise ValueError("Only held-out final_test predictions belong in this comparison.")
    for column in ("target_timestamp", "forecast_origin"):
        predictions[column] = pd.to_datetime(predictions[column], utc=True)
    if predictions.target_timestamp.duplicated().any():
        raise ValueError("Use one forecast per target interval to avoid double counting.")
    if not predictions.groupby("forecast_origin").size().eq(96).all():
        raise ValueError("Expected complete non-overlapping 96-step forecasts.")
    actual = load_smard(actual_path, "actual_mwh")
    official = load_smard(forecast_path, "published_forecast_mwh")
    # Left joins preserve missing observations so they are counted explicitly.
    joined = predictions.merge(actual, on="target_timestamp", how="left", validate="one_to_one")
    joined = joined.merge(official, on="target_timestamp", how="left", validate="one_to_one")
    complete = joined[["actual_mwh", "published_forecast_mwh", "prediction"]].notna().all(axis=1)
    rows = joined.loc[complete].copy()
    if rows.empty:
        raise ValueError("No complete aligned prediction/actual/benchmark rows.")
    if not np.isfinite(rows[["actual_mwh", "published_forecast_mwh", "prediction"]]).all().all():
        raise ValueError("Non-finite values in aligned benchmark data.")
    if not np.allclose(rows.actual, rows.actual_mwh, rtol=0, atol=1e-8):
        raise ValueError("Downloaded actuals differ from the saved test labels; reconcile revisions first.")
    models = {"GBT": "prediction", "Published TransnetBW forecast": "published_forecast_mwh"}
    metrics = {name: score(rows, column) for name, column in models.items()}
    for name, column in models.items():
        prefix = {"GBT": "gbt", "Published TransnetBW forecast": "published"}[name]
        rows[f"{prefix}_error_mwh"] = rows.actual_mwh - rows[column]
    rows["local_month"] = rows.target_timestamp.dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m")
    monthly = pd.DataFrame([
        {"month": month, "model": name, **score(group, column)}
        for month, group in rows.groupby("local_month") for name, column in models.items()
    ])
    gbt, published = metrics["GBT"], metrics["Published TransnetBW forecast"]
    mae_delta = 100 * (gbt["mae_mwh"] / published["mae_mwh"] - 1)
    rmse_delta = 100 * (gbt["rmse_mwh"] / published["rmse_mwh"] - 1)
    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics, "source_prediction_rows": len(predictions),
        "scored_rows": len(rows), "excluded_missing_rows": int((~complete).sum()),
        "start_utc": rows.target_timestamp.min().isoformat(),
        "last_interval_start_utc": rows.target_timestamp.max().isoformat(),
        "gbt_mae_change_vs_published_percent": mae_delta,
        "gbt_rmse_change_vs_published_percent": rmse_delta,
        "gbt_lower_absolute_error_fraction": float((rows.gbt_error_mwh.abs() < rows.published_error_mwh.abs()).mean()),
        "downloaded_actuals_match_saved_labels": True,
        "input_hash_normalization": "CSV bytes with CRLF converted to LF, so hashes survive Git line-ending conversion.",
        "inputs": {label: {"path": portable_paths(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
                   for label, path in (("predictions", predictions_path), ("actual", actual_path), ("forecast", forecast_path))},
        "comparison_status": "retrospective delivery-time match; issue times and forecast revisions not matched",
        "observation_delay_minutes": delay,
        "source_value_column": "grid load [MWh] Original resolutions",
        "region_provenance": "TransnetBW selection supplied with these exports; region is not embedded in CSV headers",
    }
    for prefix, column in (("gbt", "prediction"), ("published", "published_forecast_mwh")):
        rows[f"{prefix}_underprediction_mwh"] = (rows.actual_mwh - rows[column]).clip(lower=0)
        rows[f"{prefix}_overprediction_mwh"] = (rows[column] - rows.actual_mwh).clip(lower=0)
    rows.to_csv(output / "predictions/transnetbw_aligned.csv", index=False)
    monthly.to_csv(output / "metrics/transnetbw_monthly.csv", index=False)
    (output / "metrics/transnetbw_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in models:
        subset = monthly.loc[monthly.model == name]
        for axis, metric in zip(axes, ("mae_mwh", "rmse_mwh")):
            axis.plot(subset.month, subset[metric], marker="o", markersize=3, label=name)
            axis.set(ylabel=metric.replace("_mwh", "").upper() + " (MWh per interval)", xlabel="2025 delivery month")
            axis.tick_params(axis="x", rotation=60)
    for axis in axes:
        axis.set_ylim(0, monthly["mae_mwh" if axis is axes[0] else "rmse_mwh"].max() * 1.1)
    axes[0].legend(fontsize=8)
    figure.suptitle("Same delivery intervals; forecast issuance times are not matched")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    output_manifest = output / "run_manifest.json"
    if output_manifest.exists():
        output_metadata = json.loads(output_manifest.read_text(encoding="utf-8"))
        output_metadata.setdefault("comparisons", {})["transnetbw"] = {
            "metrics": "metrics/transnetbw_summary.json",
            "predictions": "predictions/transnetbw_aligned.csv",
            "plot": portable_paths(plot_path.resolve()),
        }
        output_manifest.write_text(json.dumps(output_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "inputs"}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Run directory; defaults to the prediction file's run.")
    parser.add_argument("--plot", type=Path, help="Optional plot destination; defaults to the run's plots folder.")
    args = parser.parse_args()
    compare(args.predictions, args.actual, args.forecast, args.output, args.plot)


if __name__ == "__main__":
    main()

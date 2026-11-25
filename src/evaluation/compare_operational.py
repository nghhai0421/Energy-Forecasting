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
    availability_note = (
        f"GBT's lag features include a configured {delay}-minute availability buffer. "
        "This does not verify historical publication vintages or later revisions."
        if delay is not None else
        "Additional observation-publication delay is not verified by this comparison; consult the source run configuration."
    )
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

    metric_lines = "\n".join(
        f"| {name} | {value['mae_mwh']:.2f} | {value['rmse_mwh']:.2f} | {value['underprediction_mwh']:,.2f} | {value['overprediction_mwh']:,.2f} |"
        for name, value in metrics.items())
    monthly_lines = "\n".join(
        f"| {month} | {group.loc[group.model == 'GBT', 'mae_mwh'].iloc[0]:.2f} | {group.loc[group.model == 'Published TransnetBW forecast', 'mae_mwh'].iloc[0]:.2f} | {int(group.intervals.iloc[0]):,} |"
        for month, group in monthly.groupby("month"))
    first = rows.target_timestamp.min().tz_convert("Europe/Berlin")
    last = rows.target_timestamp.max().tz_convert("Europe/Berlin")
    report = f"""# GBT comparison with the published TransnetBW forecast

Evaluated {summary['evaluated_at_utc'][:10]} using saved predictions; no retraining or test-set tuning.

## Coverage and matching

- Source run: `{Path(predictions_path).parent.parent.name}`.
- {len(rows):,} identical quarter-hour intervals, {first.isoformat()} through {last.isoformat()} (interval starts).
- {rows.forecast_origin.nunique()} GBT forecast origins; each predicts 96 intervals.
- {int((~complete).sum())} saved prediction rows excluded for missing data.
- Downloaded actual grid load matches saved test labels exactly within 1e-8 MWh.
- SMARD files use MWh per quarter-hour already; no power-to-energy conversion applied.
- UTC joins disambiguate repeated autumn local times. No interpolation of benchmark forecasts.
- Both exports use `grid load [MWh] Original resolutions`. The TransnetBW region
  comes from the supplied download selection; the CSV headers do not encode it.

## Results

| Forecast | MAE (MWh/interval) | RMSE (MWh/interval) | Underprediction total (MWh) | Overprediction total (MWh) |
| --- | ---: | ---: | ---: | ---: |
{metric_lines}

GBT minus published forecast relative error: **{mae_delta:+.2f}% MAE** and **{rmse_delta:+.2f}% RMSE**
(positive means GBT has higher error). GBT has lower absolute error in
{summary['gbt_lower_absolute_error_fraction']:.2%} of scored intervals; this is descriptive, not a significance test.
Underprediction is max(actual minus forecast, 0); overprediction is the reverse.
These measure forecasting errors, not actual purchases or balancing-account positions.

The published series overpredicts in {published['overprediction_fraction']:.2%} of intervals,
with signed mean error (actual minus forecast) {published['bias_actual_minus_forecast_mwh']:.2f} MWh.
GBT's signed mean error is {gbt['bias_actual_minus_forecast_mwh']:.2f} MWh.
This systematic difference deserves investigation; its cause is not established here.

## Monthly MAE

| Month | GBT MAE | Published forecast MAE | Intervals |
| --- | ---: | ---: | ---: |
{monthly_lines}

Coverage is limited to the dates above; no forecasts are extrapolated into missing months.

## What this answers

The saved GBT does forecast held-out 2025 load: weights were fitted on 2023-2024,
and prior 2025 observations refresh lagged inputs at later daily origins without
retraining. Its accuracy can be scored against the published forecast on the same
actual load and delivery timestamps. This report provides that preliminary comparison.
It does not establish superiority under identical forecast-time information.

## Limits to the operational comparison

- GBT issues a rolling next-24-hour forecast at 23:00 UTC (midnight in winter,
  01:00 in summer); the benchmark is a day-ahead delivery-day forecast. The model
  may have more recent information. Lead times and issuance times are not matched.
- CSVs contain delivery timestamps but no forecast issue time or revision identifier.
  We have not established that the downloaded benchmark is the original forecast
  available at the intended cutoff.
- GBT uses lagged historical temperature, not archived weather forecasts. Different
  available feature sets are a limitation when interpreting an operational comparison.
- {availability_note}
- The comparison covers GBT and the published TransnetBW forecast.
  ARIMA and Chronos were not evaluated on this identical 2025 setup here.
- Forecast accuracy is not procurement savings; no reBAP or cost model is used.

Next: preserve this preliminary result, align issuance/horizons and data availability,
add archived weather forecasts, and rerun all models under the common protocol.
Once results from 2025 inform changes, further 2025 comparisons are exploratory;
reserve another untouched period for a confirmatory final test.

SMARD identifies forecasted consumption as a day-ahead total-load forecast.
ENTSO-E documents forecast revisions and subsequent updates of TransnetBW actual
load using settlement data. These support treating this as a retrospective
comparison rather than a reconstruction of the original information sets.
See the [SMARD user guide](https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf)
and [ENTSO-E load data definitions](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Actual-Total-Load-Day-ahead-Per-Bidding-Zone-6-1-A-6-1-B).

## Artifacts

Aligned predictions are in `predictions/transnetbw_aligned.csv`; metrics and input hashes
are in `metrics/transnetbw_summary.json` and `metrics/transnetbw_monthly.csv` under the run.
The comparison plot is saved to `{portable_paths(plot_path)}`.
"""
    (output / "artifacts/transnetbw_comparison.md").write_text(report, encoding="utf-8")
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

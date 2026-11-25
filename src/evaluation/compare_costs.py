"""Price the aligned step-96 forecasts using day-ahead procurement and reBAP."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.config import portable_paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {"Chronos-2": "chronos2_prediction", "GBT": "gbt_prediction",
          "TransnetBW": "transnetbw_prediction"}


def check_prices(frame):
    if frame.target_timestamp.isna().any() or frame.target_timestamp.duplicated().any():
        raise ValueError("Price timestamps must be valid and unique.")
    return frame


def load_day_ahead(path):
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["Start date"], format="%b %d, %Y %I:%M %p")
    dates = dates.dt.tz_localize("Europe/Berlin", ambiguous="infer").dt.tz_convert("UTC")
    values = pd.to_numeric(raw["Germany/Luxembourg [€/MWh]"].replace("-", np.nan), errors="raise")
    return check_prices(pd.DataFrame({"target_timestamp": dates, "day_ahead_eur_per_mwh": values}))


def load_settlement(path):
    raw = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    if not raw.Einheit.eq("EUR/MWh").all() or not raw.Zeitzone.isin(["CET", "CEST"]).all():
        raise ValueError("Expected EUR/MWh prices with CET/CEST timestamps.")
    dates = pd.to_datetime(raw.Datum + " " + raw.von, format="%d.%m.%Y %H:%M")
    dates = dates.dt.tz_localize("Europe/Berlin", ambiguous="infer")
    if not dates.dt.strftime("%Z").eq(raw.Zeitzone).all():
        raise ValueError("Settlement timezone labels disagree with timestamps.")
    if not (dates + pd.Timedelta(minutes=15)).dt.strftime("%H:%M").eq(raw["bis"]).all():
        raise ValueError("Settlement intervals must be 15 minutes.")
    return check_prices(pd.DataFrame({"target_timestamp": dates.dt.tz_convert("UTC"),
        "rebap_short_eur_per_mwh": pd.to_numeric(raw["reBAP unterdeckt"], errors="raise"),
        "rebap_long_eur_per_mwh": pd.to_numeric(raw["reBAP ueberdeckt"], errors="raise")}))


def cashflows(actual, forecast, day_ahead, short_price, long_price):
    """Positive cash flows are costs; negative are credits. Volumes already MWh."""
    actual, forecast, day_ahead, short_price, long_price = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (actual, forecast, day_ahead, short_price, long_price)])
    if not all(np.isfinite(x).all() for x in (actual, forecast, day_ahead, short_price, long_price)):
        raise ValueError("Cannot price missing or non-finite quantities/prices.")
    shortage = np.maximum(actual - forecast, 0)
    surplus = np.maximum(forecast - actual, 0)
    procurement = forecast * day_ahead
    short_settlement = shortage * short_price
    long_settlement = -surplus * long_price
    settlement = short_settlement + long_settlement
    total = procurement + settlement
    return dict(shortage_mwh=shortage, surplus_mwh=surplus,
        procurement_eur=procurement, shortage_settlement_eur=short_settlement,
        surplus_settlement_eur=long_settlement, settlement_eur=settlement,
        settlement_charges_eur=np.maximum(settlement, 0),
        settlement_credits_eur=np.maximum(-settlement, 0), total_cost_eur=total,
        actual_volume_at_day_ahead_eur=actual * day_ahead,
        cost_vs_actual_volume_at_day_ahead_eur=total - actual * day_ahead)


def compare_costs(run, day_ahead, settlement):
    run = Path(run)
    predictions = run / "predictions/evaluation_overview_step96.csv"
    rows = pd.read_csv(predictions)
    rows["target_timestamp"] = pd.to_datetime(rows.target_timestamp, utc=True)
    if rows.empty or not rows.horizon_step.eq(96).all():
        raise ValueError("Expected nonempty step-96 predictions only.")
    if rows.target_timestamp.isna().any() or rows.target_timestamp.duplicated().any():
        raise ValueError("Use exactly one forecast per delivery interval.")
    rows = rows.merge(load_day_ahead(day_ahead), on="target_timestamp", how="left", validate="one_to_one")
    rows = rows.merge(load_settlement(settlement), on="target_timestamp", how="left", validate="one_to_one")
    columns = ["actual", *MODELS.values(), "day_ahead_eur_per_mwh",
               "rebap_short_eur_per_mwh", "rebap_long_eur_per_mwh"]
    if not np.isfinite(rows[columns]).all().all():
        raise ValueError("All selected forecasts must have actuals and both price types; no rows are silently excluded.")
    results = []
    for model, column in MODELS.items():
        costs = cashflows(rows.actual, rows[column], rows.day_ahead_eur_per_mwh,
                          rows.rebap_short_eur_per_mwh, rows.rebap_long_eur_per_mwh)
        results.append(pd.DataFrame({"model": model, "target_timestamp": rows.target_timestamp,
            "actual_mwh": rows.actual, "forecast_mwh": rows[column],
            "day_ahead_eur_per_mwh": rows.day_ahead_eur_per_mwh,
            "rebap_short_eur_per_mwh": rows.rebap_short_eur_per_mwh,
            "rebap_long_eur_per_mwh": rows.rebap_long_eur_per_mwh, **costs}))
    intervals = pd.concat(results, ignore_index=True)
    monetary = [c for c in intervals if c.endswith("_eur")]
    totals = intervals.groupby("model", sort=False)[["actual_mwh", "forecast_mwh", "shortage_mwh", "surplus_mwh", *monetary]].sum()
    baseline = totals.loc["TransnetBW", "total_cost_eur"]
    totals["savings_vs_transnetbw_eur"] = baseline - totals.total_cost_eur
    totals["savings_vs_transnetbw_percent"] = (baseline - totals.total_cost_eur) / baseline * 100 if baseline != 0 else np.nan
    if (totals.actual_mwh <= 0).any():
        raise ValueError("A positive total actual volume is required for cost-per-MWh reporting.")
    totals["cost_per_actual_mwh"] = totals.total_cost_eur / totals.actual_mwh
    benchmark = intervals.loc[intervals.model.eq("TransnetBW"), ["target_timestamp", "total_cost_eur"]].rename(columns={"total_cost_eur": "transnetbw_cost_eur"})
    intervals = intervals.merge(benchmark, on="target_timestamp", validate="many_to_one")
    intervals["savings_vs_transnetbw_eur"] = intervals.transnetbw_cost_eur - intervals.total_cost_eur
    intervals["local_date"] = intervals.target_timestamp.dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m-%d")
    daily = intervals.groupby(["local_date", "model"], sort=True)[monetary + ["savings_vs_transnetbw_eur"]].sum().reset_index()
    for folder in ("artifacts", "metrics", "predictions", "plots"):
        (run / folder).mkdir(exist_ok=True)
    intervals.to_csv(run / "predictions/price_comparison_step96.csv", index=False)
    totals.to_csv(run / "metrics/price_comparison_step96.csv", index_label="model")
    daily.to_csv(run / "metrics/price_comparison_step96_daily.csv", index=False)
    summary = dict(scenario="forecast-volume day-ahead procurement, no intraday adjustment, remaining imbalance settled at directional reBAP",
        scope="same step-96 delivery intervals for all three models; not a full-month energy bill",
        intervals_per_model=len(rows), start_utc=rows.target_timestamp.min().isoformat(),
        last_interval_start_utc=rows.target_timestamp.max().isoformat(),
        sign_convention="positive cost = payment; negative cost = receipt; credits column is a positive receipt magnitude",
        formula="forecast * day_ahead + max(actual-forecast,0)*rebap_short - max(forecast-actual,0)*rebap_long",
        volume_units="MWh per interval; no additional 0.25 multiplier", price_units="EUR/MWh",
        settlement_prices_equal=bool(rows.rebap_short_eur_per_mwh.eq(rows.rebap_long_eur_per_mwh).all()),
        negative_settlement_intervals=int(rows.rebap_short_eur_per_mwh.lt(0).sum()),
        missing_price_rows=0, metrics=totals.astype(object).where(pd.notna(totals), None).to_dict(orient="index"),
        assumptions=["price-taking scenario; historical prices fixed for all models", "forecast quantity treated as signed net purchase quantity",
                     "day-ahead execution feasibility and benchmark issue times not reconstructed", "no intraday trades, transaction fees or taxes"],
        inputs={name: {"path": portable_paths(Path(path).resolve()),
                       "sha256_lf": hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
                for name, path in (("predictions", predictions), ("day_ahead", day_ahead), ("settlement", settlement))})
    (run / "metrics/price_comparison_step96.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(totals.index, totals.total_cost_eur / 1e6, color=["tab:blue", "tab:orange", "0.5"])
    axes[0].set(title="Procurement plus settlement", ylabel="Million EUR")
    axes[0].set_ylim(bottom=0)
    for model in MODELS:
        subset = daily.loc[daily.model.eq(model)]
        axes[1].plot(pd.to_datetime(subset.local_date), subset.savings_vs_transnetbw_eur.cumsum() / 1e3, label=model)
    axes[1].axhline(0, color="0.5", linewidth=0.7)
    axes[1].set(title="Cumulative savings versus published forecast", ylabel="Thousand EUR")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=45)
    figure.suptitle(f"January 2025: {len(rows)} step-96 delivery intervals; no intraday adjustment")
    figure.tight_layout()
    figure.savefig(run / "plots/price_comparison_step96.png", dpi=160)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--day-ahead", type=Path, default=Path("Dataset/raw/csv/Day-ahead_prices_2025_Quarterhour.csv"))
    parser.add_argument("--settlement", type=Path, default=Path("Dataset/benchmarks/settlement/rebap_2025_01.csv"))
    args = parser.parse_args()
    result = compare_costs(args.run, args.day_ahead, args.settlement)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

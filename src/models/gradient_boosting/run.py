"""Train on separate development files, freeze weights, then replay demo files."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sktime.forecasting.compose import YfromX

from src.models.gradient_boosting.inference import forecast, load_forecaster
from src.models.gradient_boosting.process import (
    FEATURE_VERSION, INTERVAL, boundary, build_features, load_holidays,
    load_sources, required_history,
)
from src.outputs.output import save_run_outputs
from src.config import load_run_settings, portable_paths


def write_json(path, payload):
    Path(path).write_text(json.dumps(portable_paths(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_model(features, labels, parameters):
    """Exclude missing original labels; imputation is only for historical inputs."""
    mask = labels.notna()
    if not mask.any():
        raise ValueError("No observed training labels.")
    model = YfromX(HistGradientBoostingRegressor(**parameters, early_stopping=False))
    model.set_config(remember_data=True)
    started = perf_counter()
    model.fit(labels.loc[mask].reset_index(drop=True), X=features.loc[mask].reset_index(drop=True))
    return model, int(mask.sum()), perf_counter() - started


def prediction_rows(dates, actual, values, horizon, split, fold=None):
    origins = dates[(np.arange(len(dates)) // horizon) * horizon]
    rows = pd.DataFrame({
        "split": split, "fold": fold, "forecast_origin": origins,
        "cutoff_timestamp": origins - INTERVAL, "target_timestamp": dates,
        "horizon_step": np.arange(len(dates)) % horizon + 1,
        "actual": actual.reindex(dates).to_numpy(), "prediction": values,
    })
    rows["residual"] = rows.actual - rows.prediction
    rows["deficit_mwh"] = rows.residual.clip(lower=0)
    rows["surplus_mwh"] = (-rows.residual).clip(lower=0)
    return rows


def validation_predictions(features, observed, parameters, folds, horizon):
    """Batch lag-only rows, equivalent to successive 96-step frozen-weight requests."""
    rows, timings = [], []
    for fold in folds:
        begin, end = boundary(fold["start"]), boundary(fold["end"])
        train = features.index < begin
        dates = features.index[(features.index >= begin) & (features.index < end)]
        dates = dates[:len(dates) // horizon * horizon]
        if dates.empty or not train.any():
            raise ValueError("Each validation fold needs training and a complete forecast horizon.")
        model, count, fit_seconds = fit_model(features.loc[train], observed.reindex(features.index[train]), parameters)
        X = features.loc[dates].reset_index(drop=True)
        X.index = pd.RangeIndex(count, count + len(X))
        started = perf_counter()
        values = model.predict(fh=np.arange(1, len(X) + 1), X=X).to_numpy().reshape(-1)
        timings.append(dict(fold=fold["fold"], fit_time=fit_seconds, pred_time=perf_counter() - started))
        rows.append(prediction_rows(dates, observed, values, horizon, "cv_validation", fold["fold"]))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(timings)


def error_summary(rows, column):
    scored = rows.dropna(subset=["actual", column])
    if scored.empty:
        raise ValueError("No observed targets for evaluation.")
    errors = scored.actual.to_numpy() - scored[column].to_numpy()
    return dict(mae_mwh=float(np.abs(errors).mean()), rmse_mwh=float(np.sqrt(np.mean(errors ** 2))),
                deficit_mwh=float(np.maximum(errors, 0).sum()), surplus_mwh=float(np.maximum(-errors, 0).sum()),
                net_deficit_mwh=float(errors.sum()), absolute_error_mwh=float(np.abs(errors).sum()), intervals=len(scored))


def run_split_experiment(experiment, settings, stage, hpo_run=None, run_name=None):
    dataset, split, forecast_settings = experiment["data"], experiment["split"], experiment["forecast"]
    horizon = forecast_settings["horizon"]
    warmup = required_history(dataset)
    if (forecast_settings["step_length"] != horizon or horizon > min(dataset["lag_hours"]) * 4
            or forecast_settings["lookback"] != warmup):
        raise ValueError("Use non-overlapping horizons no longer than minimum lag and the required lookback.")
    # Development loading and tuning never open the demo CSVs.
    development, observed, state, preparation = load_sources(dataset)
    start, test_start, test_end = (boundary(split[k]) for k in ("train_start", "test_start", "test_end"))
    if development.index[0] != start or development.index[-1] + INTERVAL != test_start or test_end <= test_start:
        raise ValueError("Development must cover exactly train_start up to test_start.")
    calendar_zone = dataset["sources"]["holidays"]["timezone"]
    calendar_years = range(start.tz_convert(calendar_zone).year,
                           (test_end - INTERVAL).tz_convert(calendar_zone).year + 1)
    holidays = load_holidays(dataset, years=calendar_years)
    features = build_features(development, development.index[warmup:], dataset, holidays)
    boundaries = [boundary(v) for v in split["validation_starts"]] + [test_start]
    if boundaries != sorted(set(boundaries)) or not features.index[0] < boundaries[0]:
        raise ValueError("Validation boundaries must increase within development data.")
    if any(v not in development.index for v in boundaries[:-1]):
        raise ValueError("Validation boundary missing from development data.")
    folds = [dict(fold=i, start=a.isoformat(), end=b.isoformat())
             for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1)]
    hashes = {name: hashlib.sha256(Path(source.get("development", source.get("path"))).read_bytes()).hexdigest()
              for name, source in dataset["sources"].items()}
    contract = dict(feature_version=FEATURE_VERSION, feature_columns=list(features), development_hashes=hashes,
                    holiday_dates=holidays, calendar_policy="team CSV; missing years generated with holidays.Germany(subdiv='BW')",
                    data={k: v for k, v in dataset.items() if k != "sources"},
                    source_settings={name: {k: v for k, v in source.items() if k not in {"path", "development", "demo"}}
                                     for name, source in dataset["sources"].items()},
                    split=split, forecast=forecast_settings)
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    defaults = dict(learning_rate=0.1, max_iter=150, max_depth=6, max_leaf_nodes=31,
                    min_samples_leaf=30, l2_regularization=1.0, random_state=42)
    parameters = {key: settings.get(key, value) for key, value in defaults.items()}
    selection = dict(parameters=parameters, source="model_config", experiment_signature=signature)
    split_manifest = dict(calendar=split, cv_folds=folds, development_observations=len(development),
                          warmup_observations=warmup, strategy="expanding validation; frozen demo weights")
    metadata = dict(model=experiment["model"], run_stage="hpo", hardware={"device": "cpu"},
                    preparation=preparation, training_contract=contract,
                    availability="past observations with 60-minute configured buffer; assumed, not verified publication latency",
                    research_holdout="2025 excluded from fit/tuning; earlier exploratory work already inspected 2025",
                    timing={})
    name, output_root = run_name or experiment["run_name"], experiment["output_root"]
    trials, hpo_path = None, None
    print(f"Development: {len(development):,} rows; usable inputs: {len(features):,}; features: {len(features.columns)}", flush=True)
    if stage in {"hpo", "all"}:
        hpo = settings.get("hpo", {})
        if not hpo.get("search_space") or set(hpo["search_space"]) - set(defaults):
            raise ValueError("Supply a supported hpo.search_space in the experiment YAML.")

        def objective(trial):
            candidate = parameters.copy()
            for key, specification in hpo["search_space"].items():
                options = specification.copy()
                kind = options.pop("type")
                if kind not in {"int", "float"}:
                    raise ValueError("HPO supports int and float parameters.")
                candidate[key] = getattr(trial, f"suggest_{kind}")(key, **options)
            rows, _ = validation_predictions(features, observed, candidate, folds, horizon)
            return float(np.mean([error_summary(group, "prediction")["rmse_mwh"] for _, group in rows.groupby("fold")]))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=hpo["seed"]))
        started = perf_counter()
        study.optimize(objective, n_trials=hpo["n_trials"])
        metadata["timing"]["hpo_seconds"] = perf_counter() - started
        trials = study.trials_dataframe().rename(columns={"number": "trial", "value": "score"})
        selection = dict(parameters={**parameters, **study.best_params}, source="hpo", trial=study.best_trial.number,
                         validation_score=study.best_value, metric="mean_fold_rmse", direction="minimize",
                         experiment_signature=signature)
        hpo_path = save_run_outputs(dict(trials=trials, selected_parameters=selection, split_manifest=split_manifest),
                                    output_root, f"{name}_hpo", metadata, stage="hpo")
        if stage == "hpo":
            return hpo_path
    if hpo_run is not None:
        hpo_path = Path(hpo_run)
        manifest = json.loads((hpo_path / "run_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status"), manifest.get("run_stage"), manifest.get("model")) != ("completed", "hpo", experiment["model"]):
            raise ValueError("Expected a completed HPO run for this model.")
        selection = json.loads((hpo_path / "artifacts/selected_parameters.json").read_text(encoding="utf-8"))
        if selection.get("experiment_signature") != signature:
            raise ValueError("HPO development files, features or split differ from this experiment.")
        trials = pd.read_csv(hpo_path / "artifacts/hpo_trials.csv")

    validation, fold_timing = validation_predictions(features, observed, selection["parameters"], folds, horizon)
    model, count, fit_seconds = fit_model(features, observed.reindex(features.index), selection["parameters"])
    bundle = dict(forecaster=model, feature_version=FEATURE_VERSION, feature_columns=list(features), dataset=portable_paths(dataset),
                  holiday_dates=holidays, horizon=horizon, training_rows=count,
                  training_cutoff=development.index[-1].isoformat(), imputation_state=state,
                  training_history=development.tail(warmup), experiment_signature=signature)
    # Persist final weights and their training contract BEFORE opening any demo CSV.
    checkpoint_dir = Path(output_root) / "trained_models" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_gbt_final")
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.joblib"
    joblib.dump(bundle, checkpoint)
    write_json(checkpoint_dir / "training_manifest.json", dict(contract=contract, selection=selection,
               training_rows=count, training_cutoff=bundle["training_cutoff"], fit_seconds=fit_seconds))
    # Export development-only features for inspection and reuse in training notebooks.
    processed = Path(dataset["sources"]["consumption"]["development"]).parents[2] / "processed" / "gbt_final_development.csv"
    processed.parent.mkdir(exist_ok=True, parents=True)
    features.assign(**{dataset["target_column"]: observed.reindex(features.index)}).to_csv(processed, index_label=dataset["timestamp_column"])
    bundle = load_forecaster(checkpoint)
    # sktime caches requested horizons during predict; compare learned estimator weights.
    before = joblib.hash(bundle["forecaster"].estimator_)
    demo, demo_actual, _, demo_preparation = load_sources(dataset, "demo", initial_state=state)
    if demo.index[0] != test_start or demo.index[-1] >= test_end:
        raise ValueError("Demo must begin at test_start and remain before test_end.")
    history = pd.concat([development.tail(warmup), demo])
    completed = len(demo) // horizon * horizon
    if completed == 0:
        raise ValueError("Demo needs at least one full forecast horizon.")
    values, predict_seconds = [], 0.0
    first_features = None
    for position in range(0, completed, horizon):
        result = forecast(bundle, history, demo.index[position])
        values.extend(result["prediction"].to_numpy())
        predict_seconds += result["predict_seconds"]
        if first_features is None:
            first_features = result["features"]
    if before != joblib.hash(bundle["forecaster"].estimator_):
        raise RuntimeError("Frozen model changed during demo replay.")
    final = prediction_rows(demo.index[:completed], demo_actual, values, horizon, "final_test")
    split_manifest.update(test_observations=len(demo), training_rows=count, final_test=dict(
        forecast_origins=completed // horizon, scored_observations=int(final.actual.notna().sum()),
        unscored_tail_observations=len(demo) - completed,
        evaluation_range=dict(start=demo.index[0].isoformat(), end=demo.index[completed - 1].isoformat())))
    metadata.update(run_stage="evaluation", demo_preparation=demo_preparation, hpo_source=str(hpo_path) if hpo_path else None,
                    frozen_model_hash=before, checkpoint_created_before_demo=str(checkpoint))
    metadata["timing"]["final_test"] = dict(fit_seconds=fit_seconds, predict_seconds=predict_seconds)
    results = dict(predictions=pd.concat([validation, final], ignore_index=True).dropna(subset=["actual"]),
                   selected_parameters=selection, split_manifest=split_manifest, trials=trials,
                   checkpoint_path=checkpoint, tables={"cv_validation_evaluation": fold_timing})
    path = save_run_outputs(results, output_root, f"{name}_evaluation", metadata)
    summary = error_summary(final, "prediction")
    write_json(path / "metrics/energy_errors.json", summary)
    write_json(path / "artifacts/experiment_config.json", experiment)
    final.to_csv(path / "predictions/demo_all_intervals.csv", index=False)
    first_features.to_csv(path / "artifacts/first_demo_features.csv", index_label=dataset["timestamp_column"])
    print(json.dumps(summary, indent=2), flush=True)
    return {"hpo": hpo_path, "evaluation": path} if stage == "all" else path


def run_experiment(exp_cfg, mdl_cfg=None, run_name=None,
                   stage="evaluate", hpo_run=None):
    """Run the final GBT setup from one experiment YAML.

    Like Chronos-2, configuration, preparation, and inference have one entry
    point each. GBT additionally supports development-only HPO because its
    weights are learned locally. Evaluation uses the selected parameters in
    the YAML, or a compatible saved search when hpo_run is supplied.
    """
    if stage not in {"evaluate", "hpo", "all"}:
        raise ValueError(f"Unknown run stage: {stage}")
    if stage != "evaluate" and hpo_run is not None:
        raise ValueError("--hpo-run is only used with stage='evaluate'.")
    if mdl_cfg is not None:
        raise ValueError("Gradient boosting uses the experiment YAML only; omit --mdl-cfg.")
    experiment, _ = load_run_settings(exp_cfg)
    settings = dict(experiment["parameters"])
    if "hpo" in experiment:
        settings["hpo"] = experiment["hpo"]
    return run_split_experiment(experiment, settings, stage, hpo_run, run_name)

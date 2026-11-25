"""SARIMA: load data, optionally tune order and seasonal_order, evaluate, and save results."""

import json
import locale
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
import sktime
from optuna.distributions import CategoricalDistribution
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ForecastingOptunaSearchCV
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.split import SlidingWindowSplitter, temporal_train_test_split

from src.config import load_run_settings
from src.outputs.output import save_run_outputs
from src.outputs.plot import create_forecast_plots, create_tuning_plot


class LowMemoryARIMA(ARIMA):
    """sktime ARIMA adapter that passes low_memory=True to pmdarima/statsmodels fit()."""

    def _fit(self, y, X, fh):
        self._cur_y = y
        self._cur_X = X
        if X is not None:
            X = X.loc[y.index]
        y_fit = y
        if pd.api.types.is_integer_dtype(y.index) and not isinstance(
            y.index, pd.RangeIndex
        ):
            y_fit = y.reset_index(drop=True)
            if X is not None:
                X = X.reset_index(drop=True)
        self._forecaster = self._instantiate_model()
        self._forecaster.fit(y_fit, X=X, low_memory=True)
        self._y_name = y.name
        return self

    def _predict_fixed_cutoff(self, fh, X=None, return_pred_int=False, alpha=0.05):
        n_periods = int(fh.to_relative(self.cutoff)[-1])
        result = self._forecaster.arima_res_.forecast(steps=n_periods, exog=X)
        fh_abs = fh.to_absolute(self.cutoff)
        fh_idx = fh.to_indexer(self.cutoff)
        result = pd.Series(result).iloc[fh_idx]
        result.index = fh_abs.to_pandas()
        result.name = self._y_name
        return result


class ParallelOptunaSearchCV(ForecastingOptunaSearchCV):
    """ForecastingOptunaSearchCV that forwards backend='loky' (n_jobs=-1) to evaluate()."""

    def _run_search(self, y, X, cv, scoring, scoring_name, sampler):
        from sktime.forecasting.model_selection._base import _fit_and_score

        all_results = []
        for param_grid_dict in self._param_grid:
            scoring_direction = (
                "minimize" if scoring.get_tag("lower_is_better") else "maximize"
            )
            study = optuna.create_study(direction=scoring_direction, sampler=sampler)
            meta = {
                "forecaster": self.forecaster,
                "y": y,
                "X": X,
                "cv": cv,
                "strategy": self.strategy,
                "scoring": scoring,
                "error_score": self.error_score,
                "scoring_name": scoring_name,
                "backend": self.backend,
                "backend_params": {"n_jobs": -1},
            }
            for _ in range(self.n_evals):
                trial = study.ask(param_grid_dict)
                params = {
                    name: trial.params[name] for name, v in param_grid_dict.items()
                }
                out = _fit_and_score(params, meta)
                study.tell(trial, self._get_score(out, scoring_name))

            params_list = [trial.params for trial in study.trials]
            results = study.trials_dataframe()
            results["params"] = params_list
            all_results.append(results)

        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results.rename(columns={"value": f"mean_{scoring_name}"})


def run_experiment(exp_cfg, mdl_cfg=None, run_name=None,
                   stage="evaluate", hpo_run=None):
    """Tune or evaluate SARIMA with chronological validation and held-out testing.

    Parameters
    ----------
    exp_cfg : str or pathlib.Path
        Experiment YAML path defining data, splits, forecasts, and output root.
    mdl_cfg : str or pathlib.Path or None, default None
        Optional model overrides; an hpo section is required for tuning.
    run_name : str or None, default None
        Run label; defaults to the experiment configuration's run name.
    stage : {'evaluate', 'hpo', 'all'}, default 'evaluate'
        Evaluate, tune on development folds, or tune and then evaluate.
    hpo_run : str or pathlib.Path or None, default None
        Completed HPO run to reuse during evaluation.

    Returns
    -------
    pathlib.Path or dict
        Saved run directory; 'all' returns hpo and evaluation directory paths.

    Raises
    ------
    ValueError
        If stage, HPO settings, or a supplied HPO run is invalid.
    NotImplementedError
        If the dataset requests covariates.

    Notes
    -----
    Parameter selection uses development validation scores only.
    """
    if stage not in {"all", "hpo", "evaluate"}:
        raise ValueError(f"Unknown run stage: {stage}")
    if stage != "evaluate" and hpo_run is not None:
        raise ValueError("hpo_run is only used with stage='evaluate'.")

    # Load Configs
    experiment, settings = load_run_settings(exp_cfg, mdl_cfg)
    if stage in {"hpo", "all"} and "hpo" not in settings:
        raise ValueError("HPO requires --mdl-cfg with an hpo section.")
    dataset = experiment["data"]
    horizon = experiment["forecast"]["horizon"]
    step = experiment["forecast"]["step_length"]
    output_root = experiment["output_root"]
    run_name = run_name if run_name is not None else experiment["run_name"]

    if dataset.get("covariates"):
        raise NotImplementedError("This SARIMA runner uses target history only.")

    # Set Model Defaults
    parameters = {
        "order": tuple(settings.get("order", (1, 1, 1))),
        "seasonal_order": tuple(settings.get("seasonal_order", (1, 0, 1, 4))),
        "with_intercept": settings.get("with_intercept", False),
        "enforce_stationarity": settings.get("enforce_stationarity", True),
        "enforce_invertibility": settings.get("enforce_invertibility", True),
        "suppress_warnings": settings.get("suppress_warnings", True),
    }
    selection = {
        "parameters": parameters,
        "source": "model_config" if parameters.keys() & settings.keys() else "defaults",
    }
    hpo_path = None
    trials = None
    timing = {}

    # Load Data
    locale.setlocale(locale.LC_TIME, "C")
    raw = pd.read_csv(dataset["path"])
    if dataset["timestamp_column"] not in raw.columns:
        raw = pd.read_csv(dataset["path"], sep=";", decimal=",", thousands=".")
    ts_col = raw[dataset["timestamp_column"]]
    dates = pd.to_datetime(ts_col, format="%b %d, %Y %I:%M %p", errors="coerce")
    if dates.isna().all():
        dates = pd.to_datetime(ts_col, format="%d.%m.%Y %H:%M")
    # SMARD uses Berlin time; infer repeated autumn hours in file order.
    dates = dates.dt.tz_localize("Europe/Berlin", ambiguous="infer").dt.tz_convert("UTC")
    target_series = pd.to_numeric(raw[dataset["target_column"]], errors="coerce")
    if target_series.isna().any():
        target_series = target_series.fillna(target_series.expanding(min_periods=1).mean().shift(1))
    values = target_series.to_numpy(dtype=float)
    series = pd.DataFrame({"timestamp": dates, "target": values})
    y = pd.Series(values, name=dataset["target_column"])

    # Split Data
    # Hold out the final test period before tuning.
    test_size = int(len(y) * experiment["split"]["test_fraction"])
    development, test = temporal_train_test_split(y, test_size=test_size)
    window_length = int(len(development) * experiment["split"]["train_window_fraction"])
    fh = np.arange(1, horizon + 1)

    # Build Model
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)
    forecaster = LowMemoryARIMA(**parameters)
    rmse = MeanSquaredError(square_root=True)

    # Set Run Metadata
    metadata = {
        "model": experiment["model"],
        "hardware": {"device": "cpu"},
        "sktime_version": sktime.__version__,
        "config_files": {
            "experiment": str(Path(exp_cfg).resolve()),
            "model": str(Path(mdl_cfg).resolve()) if mdl_cfg is not None else None,
        },
        "data": dataset,
        "forecast": experiment["forecast"],
        "timing": timing,
    }
    split_manifest = {
        "fractions": experiment["split"],
        "development_observations": len(development),
        "test_observations": len(test),
        "window_length": window_length,
        "splitter": "SlidingWindowSplitter",
        "strategy": "refit",
    }

    # Run HPO
    if stage in {"hpo", "all"}:
        hpo_settings = settings["hpo"]
        candidate_orders = [tuple(order) for order in hpo_settings["orders"]]
        candidate_seasonal_orders = [
            tuple(seasonal_order) for seasonal_order in hpo_settings["seasonal_orders"]
        ]
        search = ParallelOptunaSearchCV(
            forecaster=forecaster, cv=cv,
            param_grid={
                "order": CategoricalDistribution(candidate_orders),
                "seasonal_order": CategoricalDistribution(candidate_seasonal_orders),
            },
            scoring=rmse, strategy="refit", refit=False,
            n_evals=hpo_settings["n_trials"],
            sampler=optuna.samplers.TPESampler(seed=hpo_settings["seed"]),
            backend="loky", error_score=1e9,
        )
        started = perf_counter()
        search.fit(development, fh=fh)
        timing["hpo_seconds"] = perf_counter() - started
        trials = search.cv_results_.rename(columns={
            "number": "trial", f"mean_test_{rmse.name}": "score",
        })
        selection = {
            "parameters": {
                **parameters,
                "order": tuple(search.best_params_["order"]),
                "seasonal_order": tuple(search.best_params_["seasonal_order"]),
            },
            "source": "hpo",
            "trial": int(search.cv_results_.iloc[search.best_index_]["number"]),
            "validation_score": float(search.best_score_),
            "metric": "rmse",
            "direction": "minimize",
        }
        hpo_path = save_run_outputs(
            {"trials": trials, "selected_parameters": selection, "split_manifest": split_manifest},
            output_root=output_root, run_name=f"{run_name}_hpo",
            metadata={**metadata, "run_stage": "hpo"}, stage="hpo",
        )
        create_tuning_plot(hpo_path)
        if stage == "hpo":
            return hpo_path

    # Load HPO Parameters
    if hpo_run is not None:
        hpo_path = Path(hpo_run).expanduser().resolve()
        manifest = json.loads((hpo_path / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("run_stage") != "hpo":
            raise ValueError(f"{hpo_path} is not a completed HPO run.")
        if manifest["model"] != experiment["model"]:
            raise ValueError("The HPO run belongs to a different model.")
        selection = json.loads((hpo_path / "artifacts/selected_parameters.json").read_text(encoding="utf-8"))
        trials = pd.read_csv(hpo_path / "artifacts/hpo_trials.csv")

    # Fit Model
    selected_parameters = selection["parameters"].copy()
    selected_parameters["order"] = tuple(selected_parameters["order"])  # JSON loads orders as lists.
    if "seasonal_order" in selected_parameters:
        selected_parameters["seasonal_order"] = tuple(selected_parameters["seasonal_order"])
    model = forecaster.clone().set_params(**selected_parameters)
    started = perf_counter()
    model.fit(development, fh=fh)
    timing["fit_seconds"] = perf_counter() - started

    # Evaluate CV
    started = perf_counter()
    validation = evaluate(
        model.clone(), cv=cv, y=development, strategy="refit",
        scoring=[MeanAbsoluteError(), rmse], return_data=True, error_score="raise",
        backend="loky", backend_params={"n_jobs": -1},
    )
    timing["cv_validation"] = {
        "total_eval_seconds": perf_counter() - started,
        "mean_fit_seconds": float(validation["fit_time"].mean()),
        "mean_predict_seconds": float(validation["pred_time"].mean()),
    }

    # Evaluate Test
    # Start with the last development window and retain original row indices.
    test_history = y.iloc[len(development) - window_length:]
    test_cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step)
    started = perf_counter()
    final = evaluate(
        model.clone(), cv=test_cv, y=test_history, strategy="refit",
        scoring=[MeanAbsoluteError(), rmse], return_data=True, error_score="raise",
        backend="loky", backend_params={"n_jobs": -1},
    )
    timing["final_test"] = {
        "total_eval_seconds": perf_counter() - started,
        "mean_fit_seconds": float(final["fit_time"].mean()),
        "mean_predict_seconds": float(final["pred_time"].mean()),
    }

    # Format Results
    results = format_results(
        {"cv_validation": validation, "final_test": final}, series, split_manifest,
    )
    results["selected_parameters"] = selection
    results["trials"] = trials
    metadata["run_stage"] = "evaluation"
    if hpo_path is not None:
        metadata["hpo_source"] = str(hpo_path)

    # Save Outputs
    evaluation_path = save_run_outputs(
        results, output_root=output_root, run_name=f"{run_name}_evaluation",
        metadata=metadata,
    )
    joblib.dump(model, evaluation_path / "checkpoints/model.joblib")

    # Plot Results
    if trials is not None:
        create_tuning_plot(evaluation_path)
    create_forecast_plots(evaluation_path, series)
    if stage == "all":
        return {"hpo": hpo_path, "evaluation": evaluation_path}
    return evaluation_path


def format_results(evaluations, series, split_manifest):
    """Convert rolling forecast evaluations into standardized results.

    Parameters
    ----------
    evaluations : dict of pandas.DataFrame
        Split names mapped to fold rows with cutoff, y_train, y_test, and y_pred.
    series : pandas.DataFrame
        Timestamp and target columns aligned with the evaluation row indices.
    split_manifest : dict
        Development/test observation counts and chronological split settings.

    Returns
    -------
    dict
        Prediction rows, evaluation tables, and split boundaries in a copied manifest.
    """
    # Format Predictions
    predictions = []
    tables = {}
    split_manifest = split_manifest.copy()
    dates = series["timestamp"]
    for split, evaluation in evaluations.items():
        split_rows = []
        folds = []
        for number, (_, result) in enumerate(evaluation.iterrows(), start=1):
            cutoff = int(result["cutoff"])
            forecast = result["y_pred"]
            training = result["y_train"]
            rows = pd.DataFrame({
                "split": split,
                "fold": pd.array([number if split == "cv_validation" else None] * len(forecast), dtype="Int64"),
                "forecast_origin": dates.iloc[cutoff + 1],
                "cutoff_timestamp": dates.iloc[cutoff],
                "target_timestamp": dates.iloc[forecast.index].to_numpy(),
                "horizon_step": forecast.index.to_numpy() - cutoff,
                "actual": result["y_test"].reindex(forecast.index).to_numpy(),
                "prediction": forecast.to_numpy(),
            })
            rows["residual"] = rows["actual"] - rows["prediction"]
            split_rows.append(rows)

            # Record Ranges
            fold = {"fold": number}
            for name, positions in (("fit_range", training.index),
                                    ("evaluation_range", forecast.index)):
                first, last = int(positions[0]), int(positions[-1])
                fold[name] = {
                    "first_row": first, "last_row": last, "n_observations": len(positions),
                    "start": dates.iloc[first].isoformat(),
                    "end": dates.iloc[last].isoformat(),
                }
            fold["forecast_origin"] = dates.iloc[cutoff + 1].isoformat()
            folds.append(fold)

        # Record Split
        rows = pd.concat(split_rows, ignore_index=True)
        predictions.append(rows)
        tables[f"{split}_evaluation"] = evaluation.drop(columns=["y_train", "y_test", "y_pred"])
        if split == "cv_validation":
            split_manifest["cv_folds"] = folds
        else:
            start = split_manifest["development_observations"]
            scored = int(rows["target_timestamp"].nunique())
            split_manifest["final_test"] = {
                "initial_fit_range": folds[0]["fit_range"],
                "evaluation_range": {
                    "first_row": start, "last_row": len(series) - 1,
                    "n_observations": len(series) - start,
                    "start": dates.iloc[start].isoformat(), "end": dates.iloc[-1].isoformat(),
                },
                "folds": folds,
                "forecast_origins": len(folds),
                "scored_observations": scored,
                "unscored_observations": split_manifest["test_observations"] - scored,
            }
    return {
        "predictions": pd.concat(predictions, ignore_index=True),
        "tables": tables,
        "split_manifest": split_manifest,
    }

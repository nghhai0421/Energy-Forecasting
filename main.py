"""Choose config files and start the model named in the experiment."""

import argparse
from pathlib import Path

from src.config import load_run_settings


def main():
    """Read CLI options and start the selected experiment; print saved run paths."""
    # Read Args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-cfg", type=Path, required=True,
                        help="Experiment YAML file.")
    parser.add_argument("--mdl-cfg", type=Path, help="Optional model/HPO overrides.")
    parser.add_argument("--stage", choices=("all", "hpo", "evaluate"), default="evaluate",
                        help="Evaluate defaults or a saved HPO result (default), run HPO, or both.")
    parser.add_argument("--hpo-run", type=Path,
                        help="Optional HPO run directory to use with --stage evaluate.")
    parser.add_argument("--output-root", type=Path,
                        help="Override the parent folder for hpo_runs/ and runs/.")
    parser.add_argument("--run-name", help="Override the run name.")
    args = parser.parse_args()
    if args.stage != "evaluate" and args.hpo_run is not None:
        parser.error("--hpo-run is only used with --stage evaluate.")

    # Select Model
    experiment, _ = load_run_settings(args.exp_cfg, args.mdl_cfg)
    model_name = experiment["model"]
    if model_name == "ridge_regression":
        from src.models.ridge_regression.run import run_experiment
    elif model_name == "ARIMA":
        from src.models.ARIMA.run import run_experiment
    elif model_name in {"itransformer", "tabpfn_ts"}:
        raise NotImplementedError(f"{model_name} is still a placeholder.")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Run Experiment
    run_experiment(
        args.exp_cfg, args.mdl_cfg, args.output_root, args.run_name,
        stage=args.stage, hpo_run=args.hpo_run,
    )


if __name__ == "__main__":
    main()

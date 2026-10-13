"""Choose config files and start the model named in the experiment."""

import argparse
from pathlib import Path

from src.config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-cfg", type=Path, required=True,
                        help="Experiment YAML file.")
    parser.add_argument("--mdl-cfg", type=Path, required=True, help="Model/HPO YAML file.")
    parser.add_argument("--stage", choices=("all", "hpo", "evaluate"), default="all",
                        help="Run HPO, evaluate a saved HPO result, or run both (default).")
    parser.add_argument("--hpo-run", type=Path,
                        help="HPO run directory to use with --stage evaluate.")
    parser.add_argument("--output-root", type=Path,
                        help="Override the parent folder for hpo_runs/ and runs/.")
    parser.add_argument("--run-name", help="Override the run name.")
    args = parser.parse_args()
    if args.stage == "evaluate" and args.hpo_run is None:
        parser.error("--stage evaluate requires --hpo-run.")
    if args.stage != "evaluate" and args.hpo_run is not None:
        parser.error("--hpo-run is only used with --stage evaluate.")
    model_name = read_config(args.exp_cfg)["model"]
    if model_name == "ridge_regression":
        from src.models.ridge_regression.run import run_experiment
    elif model_name in {"itransformer", "tabpfn_ts"}:
        raise NotImplementedError(f"{model_name} is still a placeholder.")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    run_experiment(
        args.exp_cfg, args.mdl_cfg, args.output_root, args.run_name,
        stage=args.stage, hpo_run=args.hpo_run,
    )


if __name__ == "__main__":
    main()

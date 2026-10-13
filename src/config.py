"""Read YAML settings shared by the model runners."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    with open(path, encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    if not isinstance(settings, dict):
        raise ValueError(f"Expected settings in {path}.")
    return settings


def load_run_settings(exp_cfg, mdl_cfg):
    experiment = read_config(exp_cfg)
    settings = read_config(mdl_cfg)
    experiment["data"]["path"] = str((PROJECT_ROOT / experiment["data"]["path"]).resolve())
    experiment["output_root"] = str((PROJECT_ROOT / experiment["output_root"]).resolve())

    split = experiment["split"]
    if not 0 < split["test_fraction"] < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not 0 < split["train_window_fraction"] < 1:
        raise ValueError("train_window_fraction must be between 0 and 1.")
    if min(experiment["forecast"].values()) < 1:
        raise ValueError("Lookback, horizon and step_length must be positive.")
    return experiment, settings

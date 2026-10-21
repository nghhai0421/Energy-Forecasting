"""Load experiment and model settings."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_run_settings(exp_cfg, mdl_cfg=None):
    """Read experiment YAML and optional model YAML; return settings and overrides."""
    # Load Configs
    configs = []
    for path in (exp_cfg, mdl_cfg):
        if path is None:
            configs.append({})
            continue
        with open(path, encoding="utf-8") as file:
            settings = yaml.safe_load(file)
        if not isinstance(settings, dict):
            raise ValueError(f"Expected settings in {path}.")
        configs.append(settings)
    experiment, settings = configs

    # Resolve Paths
    experiment["data"]["path"] = str((PROJECT_ROOT / experiment["data"]["path"]).resolve())
    experiment["output_root"] = str((PROJECT_ROOT / experiment["output_root"]).resolve())

    # Check Settings
    split = experiment["split"]
    if not 0 < split["test_fraction"] < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not 0 < split["train_window_fraction"] < 1:
        raise ValueError("train_window_fraction must be between 0 and 1.")
    if min(experiment["forecast"].values()) < 1:
        raise ValueError("Lookback, horizon and step_length must be positive.")
    return experiment, settings

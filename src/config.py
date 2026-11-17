"""Load experiment and model settings."""

from pathlib import Path
from datetime import date
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_run_settings(exp_cfg, mdl_cfg=None):
    """Load experiment settings and optional model overrides.

    Parameters
    ----------
    exp_cfg : str or pathlib.Path
        Experiment YAML path defining data, splits, forecasts, and output root.
    mdl_cfg : str or pathlib.Path or None, default None
        Optional model/HPO YAML path.

    Returns
    -------
    tuple of dict
        Experiment with resolved dataset/output paths and model overrides.

    Raises
    ------
    ValueError
        If YAML content, split fractions, or forecast settings are invalid.
    """
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
    dataset = experiment["data"]
    if "sources" in dataset:
        for source in dataset["sources"].values():
            for key in ("path", "development", "test", "demo"):
                if key in source:
                    paths = source[key]
                    if isinstance(paths, list):
                        source[key] = [str((PROJECT_ROOT / path).resolve()) for path in paths]
                    else:
                        source[key] = str((PROJECT_ROOT / paths).resolve())
    else:
        dataset["path"] = str((PROJECT_ROOT / dataset["path"]).resolve())
    experiment["output_root"] = str((PROJECT_ROOT / experiment["output_root"]).resolve())

    # Check Settings
    split = experiment["split"]
    if experiment["model"] == "chronos2":
        boundaries = [date.fromisoformat(split[key]) for key in
                      ("development_start", "test_start", "test_end")]
        if not boundaries[0] < boundaries[1] < boundaries[2]:
            raise ValueError("Require development_start < test_start < test_end (exclusive).")
        ZoneInfo(split["timezone"])
        for name in ("consumption", "weather"):
            if not all(dataset["sources"][name].get(key) for key in ("development", "test")):
                raise ValueError(f"{name} requires development and test source paths.")
    elif not 0 < split["test_fraction"] < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not 0 < split["train_window_fraction"] < 1:
        raise ValueError("train_window_fraction must be between 0 and 1.")
    if min(experiment["forecast"].values()) < 1:
        raise ValueError("Lookback, horizon and step_length must be positive.")
    return experiment, settings

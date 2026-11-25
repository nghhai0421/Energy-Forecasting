"""Exported configuration must work after moving the repository."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.config import PROJECT_ROOT, load_run_settings, portable_paths


class PortableConfigurationTests(unittest.TestCase):
    def test_chronos_boundaries_and_gbt_config_coexist(self):
        config = PROJECT_ROOT / "configs/experiments/chronos2.yaml"
        experiment, _ = load_run_settings(config)
        self.assertEqual(experiment["split"]["test_end"], "2025-02-01")
        with tempfile.TemporaryDirectory() as temporary:
            experiment["split"]["test_end"] = "2024-01-01"
            invalid = Path(temporary) / "invalid.yaml"
            invalid.write_text(yaml.safe_dump(experiment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "development_start < test_start"):
                load_run_settings(invalid)

    def test_exported_config_loads_in_another_checkout(self):
        source = PROJECT_ROOT / "configs/experiments/gradient_boosting.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        loaded, _ = load_run_settings(source)
        exported = portable_paths(loaded)
        self.assertEqual(exported, raw)
        self.assertTrue(Path(loaded["output_root"]).is_absolute())
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            config = checkout / "experiment.yaml"
            config.write_text(yaml.safe_dump(exported, sort_keys=False), encoding="utf-8")
            with patch("src.config.PROJECT_ROOT", checkout):
                relocated, _ = load_run_settings(config)
                self.assertEqual(Path(relocated["output_root"]), checkout / "outputs")
                for source in relocated["data"]["sources"].values():
                    for key in ("path", "development", "demo"):
                        if key in source:
                            self.assertTrue(Path(source[key]).is_relative_to(checkout))
                self.assertEqual(portable_paths(relocated), raw)

    def test_non_paths_and_external_inputs_are_preserved(self):
        external = str(PROJECT_ROOT.parent / "external.csv")
        original = {"score": 72.7, "enabled": True, "values": [None, "grid load [MWh]", external]}
        self.assertEqual(portable_paths(original), original)


if __name__ == "__main__":
    unittest.main()

"""Save run metadata, predictions, metrics, and checkpoints."""

import json
import os
import platform
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.evaluation.metrics import summarize_predictions


def save_run_outputs(results, output_root="outputs", run_name=None,
                     metadata=None, stage="evaluation"):
    """Save finished results; return the new run Path.

    Input: standardized result dict; predictions for evaluation or trials for HPO.
    Optional checkpoint_path is an exported file/folder. Save failures are logged.
    """
    # Set Run ID
    stage_folders = {"hpo": "hpo_runs", "evaluation": "runs"}
    if stage not in stage_folders:
        raise ValueError(f"Unsupported run output stage: {stage}")
    root = Path(output_root).expanduser().resolve() / stage_folders[stage]
    root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    label = re.sub(r"[^a-z0-9]+", "-", (run_name or "run").lower()).strip("-")
    label = label[:64].rstrip("-") or "run"
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    json_options = {"indent": 2, "ensure_ascii": False, "allow_nan": False}

    # Create Run
    while True:
        run_id = f"{timestamp}_{label}_{uuid4().hex[:12]}"
        run_path = root / run_id
        try:
            run_path.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        break
    for name in ("predictions", "metrics", "plots", "checkpoints", "artifacts", "logs"):
        (run_path / name).mkdir()
    manifest = {
        "run_id": run_id,
        "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
        "status": "initialized",
        "run_name": run_name,
    }
    manifest_path = run_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, **json_options) + "\n", encoding="utf-8")
    temporary = manifest_path.with_suffix(".tmp")
    print(f"Run directory: {run_path}", flush=True)

    # Save Results
    try:
        manifest.update(metadata or {})

        # Record Hardware
        cpu_model = platform.processor() or "unknown"
        if platform.system() == "Windows":
            import winreg

            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                    cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            except OSError:
                pass  # Keep the platform fallback.
        manifest["hardware"] = {
            "device": "unspecified", **manifest.get("hardware", {}),
            "cpu_model": cpu_model, "logical_core_count": os.cpu_count(),
        }

        # Save Manifest
        manifest["status"] = "saving"
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        temporary.write_text(json.dumps(manifest, **json_options) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        if stage == "hpo":
            summary = save_hpo_results(run_path, results)
        else:
            summary = save_results(run_path, results)
        manifest["status"] = "completed"
        manifest["summary"] = summary
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        temporary.write_text(json.dumps(manifest, **json_options) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    except (Exception, KeyboardInterrupt) as error:
        (run_path / "logs/error.log").write_text(traceback.format_exc(), encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        temporary.write_text(json.dumps(manifest, **json_options) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        raise
    report = {**summary, "timing": manifest.get("timing", {}), "hardware": manifest["hardware"]}
    print(json.dumps(report, indent=2, allow_nan=False))
    return run_path


def save_hpo_results(run_path, results):
    """Save trials, selected_parameters, and split_manifest; return an HPO summary."""
    # Save Trials
    trials = pd.DataFrame(results["trials"])
    required = {"trial", "score"}
    if not required.issubset(trials.columns):
        raise ValueError("HPO trials need trial and score columns.")
    trials.to_csv(run_path / "artifacts/hpo_trials.csv", index=False)

    # Save Parameters
    selection = results["selected_parameters"]
    json_options = {"indent": 2, "ensure_ascii": False, "allow_nan": False}
    for name, payload in (("split_manifest.json", results["split_manifest"]),
                          ("selected_parameters.json", selection)):
        path = run_path / "artifacts" / name
        path.write_text(json.dumps(payload, **json_options) + "\n", encoding="utf-8")
    return {
        "selected_parameters": selection["parameters"],
        "validation_score": selection["validation_score"],
        "metric": selection["metric"],
        "trial_count": len(trials),
    }


def save_results(run_path, results):
    """Save prediction rows and split details; return metrics and artifact paths.

    Optional fields: selected_parameters, trials, tables, checkpoint_path.
    Checkpoints are exported by the model runner before this call.
    """
    # Check Predictions
    predictions = results["predictions"]
    required = {"split", "forecast_origin", "target_timestamp", "horizon_step",
                "actual", "prediction"}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError(f"Predictions must be nonempty and contain {sorted(required)}.")
    if not set(predictions["split"]).issubset({"cv_validation", "final_test"}):
        raise ValueError("Prediction splits must be cv_validation or final_test.")
    if predictions["split"].isna().any():
        raise ValueError("Prediction splits cannot be missing.")
    if (predictions["split"] == "cv_validation").any():
        cv_rows = predictions["split"] == "cv_validation"
        if "fold" not in predictions or predictions.loc[cv_rows, "fold"].isna().any():
            raise ValueError("CV predictions need a fold for every row.")

    # Save Predictions
    scores = {}
    json_options = {"indent": 2, "ensure_ascii": False, "allow_nan": False}
    for split, rows in predictions.groupby("split", sort=False):
        rows = rows.copy()
        rows["residual"] = rows["actual"] - rows["prediction"]
        scores[split] = summarize_predictions(rows)
        rows.to_csv(run_path / "predictions" / f"{split}.csv", index=False)
        path = run_path / "metrics" / f"{split}.json"
        path.write_text(json.dumps(scores[split], **json_options) + "\n", encoding="utf-8")

    # Save Artifacts
    artifacts = {"split_manifest.json": results["split_manifest"]}
    selection = results.get("selected_parameters")
    if selection is not None:
        artifacts["selected_parameters.json"] = selection
    for name, payload in artifacts.items():
        path = run_path / "artifacts" / name
        path.write_text(json.dumps(payload, **json_options) + "\n", encoding="utf-8")
    if results.get("trials") is not None:
        trials = pd.DataFrame(results["trials"])
        if not {"trial", "score"}.issubset(trials.columns):
            raise ValueError("HPO trials need trial and score columns.")
        trials.to_csv(run_path / "artifacts/hpo_trials.csv", index=False)
    for name, table in results.get("tables", {}).items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Table names must be simple filenames.")
        table.to_csv(run_path / "metrics" / f"{name}.csv", index=False)

    # Copy Checkpoint
    checkpoint_path = results.get("checkpoint_path")
    checkpoint = None
    if checkpoint_path is not None:
        source = Path(checkpoint_path)
        destination = run_path / "checkpoints" / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        checkpoint = str(destination.relative_to(run_path)).replace("\\", "/")

    # Summarize Results
    summary = {}
    if "cv_validation" in scores:
        summary["cv_mean_fold"] = scores["cv_validation"]["mean_fold"]
    if "final_test" in scores:
        summary["final_test"] = scores["final_test"]["overall"]
    if selection is not None:
        summary["selected_parameters"] = selection["parameters"]
        summary["parameter_source"] = selection.get("source")
        if selection.get("validation_score") is not None:
            summary["validation_score"] = selection["validation_score"]
    if checkpoint is not None:
        summary["checkpoint"] = checkpoint
    return summary

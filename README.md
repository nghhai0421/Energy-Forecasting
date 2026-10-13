# Energy Forecasting

Compare models for hourly energy forecasting. Ridge regression is the current
working example; iTransformer and TabPFN-TS are placeholders.

## Setup

Create and activate the Conda environment:

```powershell
conda env create --file environment.yml
conda activate energy_forecasting
```

Without Conda, use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn matplotlib optuna pyyaml tzdata sktime==1.2.0
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

On Linux/macOS, use `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## Run a model

Run commands from the repository root with the environment active.
Set the data, target, forecast and split settings in an experiment YAML under
`configs/experiments/`. Set model parameters and HPO settings in a model YAML
under `configs/models/`.

Run Ridge HPO:

```powershell
python main.py --stage hpo --exp-cfg configs/experiments/ridge_prototype.yaml --mdl-cfg configs/models/ridge_regression.yaml
```

Then evaluate using the full HPO run folder printed by that command:

```powershell
python main.py --stage evaluate --exp-cfg configs/experiments/ridge_prototype.yaml --mdl-cfg configs/models/ridge_regression.yaml --hpo-run outputs/hpo_runs/<hpo-run-id>
```

Use `--stage all` to run both stages together. HPO outputs go to
`outputs/hpo_runs/`; model evaluation outputs go to `outputs/runs/`.
Use `--output-root` and `--run-name` to override the output parent and run name.

## Add a model

1. Create `src/models/<model_name>/run.py`, following the Ridge example.
   Expose `run_experiment(exp_cfg, mdl_cfg, output_root=None, run_name=None, stage="all", hpo_run=None)`.
2. Add experiment and model YAML files directly in their config folders.
   Set the experiment's `model` field to your model name.
3. Add the model's dispatch entry in `main.py` and its dependencies to
   `environment.yml`.
4. Use `save_run_outputs()` from `src/outputs/output.py` to save results.

Fit preprocessing on training data only, tune on development data, and keep
final-test scores out of hyperparameter selection.

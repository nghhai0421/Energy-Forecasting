# Optional model YAML

Model defaults live inside each model's runner function. Omit `--mdl-cfg`
for evaluation with those defaults. Starting HPO requires a model YAML with
an `hpo` section; tuning settings have no fallback values in the runner.

For Ridge, the function defaults are `alpha=1.0` and `solver="cholesky"`.
For example, an override file may contain:

```yaml
alpha: 10.0
solver: cholesky
hpo:
  n_trials: 10
  seed: 42
  alpha:
    low: 0.0001
    high: 10000
    log: true
```

Normal evaluation uses the fixed parameters and does not run HPO. Use
`--stage hpo` or `--stage all` with `--mdl-cfg` to tune explicitly. Ridge reads
`hpo.alpha`, `hpo.n_trials`, and `hpo.seed` directly from the YAML. The Ridge
parameter mapping, TPE sampler, and RMSE objective remain in the model code.

ARIMA keeps its default order `(1, 1, 1)` and fitting options in its runner.
An optional YAML `order: [2, 1, 0]` overrides the default for evaluation.
`configs/models/ARIMA.yaml` contains only its HPO settings: `n_trials`, `seed`,
and a list of candidate `(p, d, q)` orders. HPO results and JSON selections
record the effective order and fitting options; the runner restores the order
tuple when loading a saved selection.

With `--hpo-run`, evaluation uses the saved selected parameters. An HPO
selection is an output artifact, not another config file. Outputs record
the effective parameters and whether they came from defaults, YAML, or HPO.

Pretrained foundation models can use model identifiers and inference defaults
defined inside their runner without a model YAML. Record the actual model ID,
revision, and inference settings in the run metadata.

The shared writer receives standard dictionaries and DataFrames, not sktime
search objects. Each runner converts its library's results to that contract;
see the root README.

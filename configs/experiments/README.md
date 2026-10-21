# Experiment YAML

Keep YAML files directly here, such as ridge_prototype.yaml.

model selects the runner. data selects the source CSV, timestamp column,
target column and covariates (currently empty for Ridge). forecast specifies
lookback, horizon and step_length in observations. For hourly data, a
step_length of 24 evaluates a forecast once a day; 1 evaluates it every hour.

Split settings are fractions:

- test_fraction: 0.2 holds out the last 20% of the full series.
- train_window_fraction: 0.7 sets each fixed CV/test training window to 70% of
  development; CV origins then move through the remaining 30%.

Development data is everything before final test. Counts are rounded down.
A SlidingWindowSplitter uses a fixed training window of that length and creates
one CV fold per forecast origin. With hourly data and step_length=24, the origin
moves forward once per day. There is no separate cv_folds setting. Both CV
and test refit at each origin. Test evaluation receives the latest development
window plus test data, so its first fit uses development only and its first
forecast begins at the test boundary. Later windows include newly observed
test actuals. The chosen parameters stay fixed, whether they came from function defaults,
optional model YAML, or HPO. The checkpoint is a separate
model fitted on all development data before test evaluation.

output_root is the parent folder for stage-specific outputs: HPO runs go in
hpo_runs/ and normal evaluation runs go in runs/. run_name controls the run
folder name. Paths in YAML resolve from the repository root. Configs are
recorded by path in the run manifest.
Checkpoint export is optional and model-specific; pretrained inference can
omit it. The runner calls plotting separately. Configs and full datasets are
not copied.

The experiment YAML remains necessary when no model YAML or HPO is used.
The Ridge reader handles the supplied SMARD CSV and daylight-saving timestamps.
sktime uses observation positions internally; outputs restore the timestamps.
Nonempty covariates currently raise an error.

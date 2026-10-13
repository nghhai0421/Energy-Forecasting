# Dataset preparation (planned)

Prepare reproducible variants here: univariate, target-derived features, and
exogenous features. Record raw input, transformations, feature availability and
variant version; do not transform the supplied raw files in place.

Processed rows will expose timestamp, target, and optional feature columns.
Start with one target series; choose the source target explicitly after reviewing
its meaning. Add series_id if multiple series are supported later. See
../docs/data_inspection.md for verified raw columns and timestamp caveats.

Every input feature must be available at its forecast origin. Lags and rolling
statistics must not use future target observations; realized future exogenous
values are not automatically available inputs. Frequency regularization, timezone
and daylight-saving ambiguity, missing values/hours and duplicates require an
explicit policy before folds are built. Do not silently discard repeated wall
clock hours or interpolate without documenting the decision.

Model-specific scaling, encoding and window construction belong in each model's
process.py. Fit preprocessing on that fold's training data only and reuse it for
validation/prediction. Future shared folds use the same processed row indices
across models and candidates.

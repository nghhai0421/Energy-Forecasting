"""Common forecasting adapter for TabPFN-TS."""


def fit_predict_fold(
    data, fold, params
):
    """Later prepare a predictor for one fold; return model and prediction rows.

    Legacy scaffold: adapt this entry point to sktime when implementing the model.
    """
    # TODO: Wire the local modules and emit forecast_origin, target_timestamp,
    # horizon_step, actual, prediction, and optional series_id (see docs/outputs.md).
    raise NotImplementedError("TabPFN-TS fold adapter is not implemented.")

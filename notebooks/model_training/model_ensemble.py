# Ensemble model creation and prediction testing for SPP WEIS.
#
# Retrains models with best hyperparameters, creates a NaiveEnsembleModel,
# and plots test predictions across multiple nodes and time periods.
#
# Usage:
#   Interactive: marimo edit notebooks/model_training/model_ensemble.py

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    import pathlib
    import pandas as pd
    import polars as pl
    import boto3
    import torch
    from darts.models import NaiveEnsembleModel

    import warnings

    warnings.filterwarnings("ignore")

    from dotenv import load_dotenv

    load_dotenv(override=True)

    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    _project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    return NaiveEnsembleModel, log, logging, os, pd, torch


@app.cell
def _():
    import plotly.io as pio

    pio.renderers.default = "notebook"
    return


@app.cell
def _(logging):
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
        logging.WARNING
    )
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(
        logging.WARNING
    )
    return


@app.cell
def _():
    import src.data_engineering as de
    from src import parameters
    from src import plotting
    from src.modeling import build_fit_tsmixerx, build_fit_tide, build_fit_tft

    return (
        build_fit_tft,
        build_fit_tide,
        build_fit_tsmixerx,
        de,
        parameters,
        plotting,
    )


@app.cell
def _(log, parameters):
    log.info(f"FORECAST_HORIZON: {parameters.FORECAST_HORIZON}")
    log.info(f"INPUT_CHUNK_LENGTH: {parameters.INPUT_CHUNK_LENGTH}")
    return


@app.cell
def _(torch):
    torch.set_float32_matmul_precision("medium")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data prep""")
    return


@app.cell
def _(de):
    con = de.create_database()
    con.execute("SHOW TABLES").fetchall()
    return (con,)


@app.cell
def _(con, de):
    lmp = de.prep_lmp(con)
    lmp
    return (lmp,)


@app.cell
def _(lmp):
    lmp_df = lmp.to_pandas().rename(
        columns={"LMP": "LMP_HOURLY", "unique_id": "node", "timestamp_mst": "time"}
    )
    return (lmp_df,)


@app.cell
def _(con, de):
    all_df_pd = de.all_df_to_pandas(de.prep_all_df(con))
    all_df_pd
    return (all_df_pd,)


@app.cell
def _(all_df_pd):
    all_df_pd.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prep model training data""")
    return


@app.cell
def _(con, de):
    lmp_all, train_all, test_all, train_test_all = de.get_train_test_all(con)
    return lmp_all, test_all, train_all, train_test_all


@app.cell
def _(de, lmp_all):
    all_series = de.get_series(lmp_all)
    all_series[0].plot()
    return (all_series,)


@app.cell
def _(de, train_test_all):
    train_test_all_series = de.get_series(train_test_all)
    train_test_all_series[0].plot()
    return (train_test_all_series,)


@app.cell
def _(de, test_all):
    test_series = de.get_series(test_all)
    test_series[0].plot()
    return (test_series,)


@app.cell
def _(all_df_pd, de):
    futr_cov = de.get_futr_cov(all_df_pd)
    futr_cov[0].plot()
    return (futr_cov,)


@app.cell
def _(all_df_pd, de):
    past_cov = de.get_past_cov(all_df_pd)
    past_cov[0].plot()
    return (past_cov,)


@app.cell
def _(con):
    con.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Retrain models with the best params""")
    return


@app.cell
def _(
    build_fit_tsmixerx,
    futr_cov,
    parameters,
    past_cov,
    test_series,
    train_test_all_series,
):
    models_tsmixer = []
    if parameters.USE_TSMIXER:
        for _i, _param in enumerate(parameters.TSMIXER_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {_i} \t" + "*" * 25, flush=True)
            _m = build_fit_tsmixerx(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **_param,
            )
            models_tsmixer += [_m]
    return (models_tsmixer,)


@app.cell
def _(
    build_fit_tide,
    futr_cov,
    parameters,
    past_cov,
    test_series,
    train_test_all_series,
):
    models_tide = []
    if parameters.USE_TIDE:
        for _i, _param in enumerate(parameters.TIDE_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {_i} \t" + "*" * 25, flush=True)
            _m = build_fit_tide(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **_param,
            )
            models_tide += [_m]
    return (models_tide,)


@app.cell
def _(
    build_fit_tft,
    futr_cov,
    parameters,
    past_cov,
    test_series,
    train_test_all_series,
):
    models_tft = []
    if parameters.USE_TFT:
        for _i, _param in enumerate(parameters.TFT_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {_i} \t" + "*" * 25, flush=True)
            _m = build_fit_tft(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **_param,
            )
            models_tft += [_m]
    return (models_tft,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Create ensemble model""")
    return


@app.cell
def _(models_tft, models_tide, models_tsmixer):
    forecasting_models = models_tsmixer + models_tide + models_tft
    return (forecasting_models,)


@app.cell
def _(NaiveEnsembleModel, forecasting_models):
    loaded_model = NaiveEnsembleModel(
        forecasting_models=forecasting_models, train_forecasting_models=False
    )
    return (loaded_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plot test predictions""")
    return


@app.cell
def _(all_series):
    plot_ind = 3
    plot_series = all_series[plot_ind]
    return plot_ind, plot_series


@app.cell
def _(plot_series):
    plot_series.plot()
    return


@app.cell
def _(pd, plot_ind, test_series):
    plot_end_times = pd.date_range(
        end=test_series[plot_ind].end_time(), periods=10, freq="d"
    )
    plot_end_times
    return (plot_end_times,)


@app.cell
def _(
    futr_cov,
    lmp_df,
    loaded_model,
    log,
    parameters,
    past_cov,
    plot_end_times,
    plot_series,
    plotting,
):
    for _pet in plot_end_times:
        log.info(f"plot_end_time: {_pet}")
        _node_name = plot_series.static_covariates.unique_id.LMP
        _ns = plot_series.drop_after(_pet)
        log.info(f"node_series.end_time(): {_ns.end_time()}")
        _fc = futr_cov[0]
        _pc = past_cov[0]

        _preds = loaded_model.predict(
            series=_ns,
            past_covariates=_pc,
            future_covariates=_fc,
            n=parameters.FORECAST_HORIZON,
            num_samples=500,
        )

        _cov_df = (
            _fc.pd_dataframe()
            .reset_index()
            .rename(columns={"timestamp_mst": "time", "re_ratio": "Ratio"})
        )
        _plot_df = plotting.get_plot_df(_preds, _cov_df, lmp_df, _node_name)
        _plot_df.rename(columns={"mean": "mean_fcast"}, inplace=True)
        plotting.plotly_forecast(_plot_df, _node_name, show_fig=True)
    return


if __name__ == "__main__":
    app.run()

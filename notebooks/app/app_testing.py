# Shiny app testing notebook for SPP WEIS price forecast.
#
# Loads the champion model via MLflow and tests prediction + plotting
# for a selected node (PSCO_PRPM_PR).
#
# Usage:
#   Interactive: marimo edit notebooks/app/app_testing.py

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
    import pickle
    import pandas as pd
    import polars as pl
    from darts import TimeSeries
    import mlflow

    import warnings

    warnings.filterwarnings("ignore")

    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    _project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    return TimeSeries, log, mlflow, os, pd, pickle, pl


@app.cell
def _():
    import src.data_engineering as de
    from src import parameters as params
    from src import plotting

    return de, params, plotting


@app.cell
def _(log, params):
    log.info(f"FORECAST_HORIZON: {params.FORECAST_HORIZON}")
    log.info(f"INPUT_CHUNK_LENGTH: {params.INPUT_CHUNK_LENGTH}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load model""")
    return


@app.cell
def _(mlflow, os):
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"
    log_uri = mlflow.get_tracking_uri()
    return


@app.cell
def _(mlflow):
    model_uri = "models:/spp_weis@champion"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return (loaded_model,)


@app.cell
def _(loaded_model):
    load_model_dict = loaded_model.metadata.to_dict()
    load_model_dict
    return (load_model_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prep data""")
    return


@app.cell
def _(de):
    con = de.create_database()
    all_df_pd = de.all_df_to_pandas(de.prep_all_df(con))
    lmp = de.prep_lmp(con)
    lmp_pd_df = lmp.to_pandas().set_index("timestamp_mst")
    con.close()
    return all_df_pd, lmp, lmp_pd_df


@app.cell
def _(lmp_pd_df):
    lmp_pd_df.index.max()
    return


@app.cell
def _(lmp_pd_df):
    lmp_pd_df.unique_id.unique()
    return


@app.cell
def _(lmp_pd_df):
    plot_node_name = "PSCO_PRPM_PR"
    _idx = lmp_pd_df.unique_id == plot_node_name
    price_df = lmp_pd_df[_idx]
    price_df
    return plot_node_name, price_df


@app.cell
def _(all_df_pd):
    _idx = all_df_pd.unique_id == "PSCO_PRPM_PR"
    node_all_df_pd = all_df_pd[_idx]
    node_all_df_pd
    return (node_all_df_pd,)


@app.cell
def _(de, price_df):
    plot_series = de.get_all_series(price_df)[0]
    return (plot_series,)


@app.cell
def _(plot_series):
    plot_series.plot()
    return


@app.cell
def _(de, node_all_df_pd):
    future_cov_series = de.get_futr_cov(node_all_df_pd)[0]
    future_cov_series.plot()
    return (future_cov_series,)


@app.cell
def _(de, node_all_df_pd):
    past_cov_series = de.get_past_cov(node_all_df_pd)[0]
    past_cov_series.plot()
    return (past_cov_series,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Test plotting""")
    return


@app.cell
def _(pd, plot_series):
    forecast_start = plot_series.end_time() - pd.Timedelta("72h")
    return (forecast_start,)


@app.cell
def _(forecast_start, log, plot_series):
    node_series = plot_series.drop_after(forecast_start)
    log.info(f"node_series.end_time(): {node_series.end_time()}")
    return (node_series,)


@app.cell
def _(
    TimeSeries,
    future_cov_series,
    lmp,
    loaded_model,
    node_series,
    params,
    past_cov_series,
    pd,
    plot_node_name,
    plotting,
):
    _data = {
        "series": [node_series.to_json()],
        "past_covariates": [past_cov_series.to_json()],
        "future_covariates": [future_cov_series.to_json()],
        "n": params.FORECAST_HORIZON,
        "num_samples": 500,
    }
    _df = pd.DataFrame(_data)

    _plot_cov_df = (
        future_cov_series.to_dataframe()
        .reset_index()
        .rename(columns={"timestamp_mst": "time", "re_ratio": "Ratio"})
    )

    _pred = loaded_model.predict(_df)
    _preds = TimeSeries.from_json(_pred)

    _lmp_df = lmp.to_pandas().rename(
        columns={"LMP": "LMP_HOURLY", "unique_id": "node", "timestamp_mst": "time"}
    )

    _plot_df = plotting.get_plot_df(
        TimeSeries.from_json(_pred), _plot_cov_df, _lmp_df, plot_node_name
    )
    _plot_df.rename(columns={"mean": "mean_fcast"}, inplace=True)
    plotting.plotly_forecast(_plot_df, plot_node_name, show_fig=False)
    return


if __name__ == "__main__":
    app.run()

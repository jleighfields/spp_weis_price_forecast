# Backfill data collection for SPP WEIS price forecast.
#
# Full historical backfill of MTLF, MTRF, daily LMP, and 5-min LMP data.
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_backfill.py
#   Script:      python notebooks/data_collection/data_collection_backfill.py

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Data collection backfill
    Gather public SPP Weis data from https://marketplace.spp.org/groups/operational-data-weis
    """
    )
    return


@app.cell
def _():
    import os
    import sys
    import pathlib
    import pandas as pd
    import polars as pl
    import logging

    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("py4j").setLevel(logging.ERROR)
    log = logging.getLogger(__name__)

    _project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
    AWS_S3_FOLDER = os.environ.get("AWS_S3_FOLDER")
    assert AWS_S3_BUCKET
    assert AWS_S3_FOLDER

    return log, pd, pl


@app.cell
def _():
    import src.data_collection as dc

    return (dc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Load Forecast""")
    return


@app.cell
def _(dc, pd):
    end_ts = (
        pd.Timestamp.now("UTC").tz_convert("America/Chicago").tz_localize(None)
        - pd.Timedelta("2D")
    )

    _range_df = dc.get_range_data_mtlf(end_ts=end_ts, n_periods=8760 * 2)
    _range_df[:10]
    return (end_ts,)


@app.cell
def _(dc, end_ts):
    mtlf_range = dc.get_range_data_mtlf(end_ts=end_ts, n_periods=8760 * 2)
    mtlf_parquet = [pf for pf in mtlf_range if pf.endswith(".parquet")]
    mtlf_parquet[:10]
    return (mtlf_parquet,)


@app.cell
def _(dc, mtlf_parquet):
    dc.upsert_mtlf_mtrf_lmp(mtlf_parquet, target="mtlf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Resource Forecast""")
    return


@app.cell
def _(dc, end_ts):
    mtrf_range = dc.get_range_data_mtrf(end_ts=end_ts, n_periods=8760 * 2)
    mtrf_parquet = [pf for pf in mtrf_range if pf.endswith(".parquet")]
    mtrf_parquet[:10]
    return (mtrf_parquet,)


@app.cell
def _(dc, mtrf_parquet):
    dc.upsert_mtlf_mtrf_lmp(mtrf_parquet, target="mtrf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP daily file""")
    return


@app.cell
def _(dc, end_ts):
    lmp_daily_range = dc.get_range_data_interval_daily_lmps(
        end_ts=end_ts, n_periods=365 * 2
    )
    lmp_daily_parquet = [pf for pf in lmp_daily_range if pf.endswith(".parquet")]
    lmp_daily_parquet[:10]
    return (lmp_daily_parquet,)


@app.cell
def _(dc, lmp_daily_parquet):
    dc.upsert_mtlf_mtrf_lmp(lmp_daily_parquet, target="lmp")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP 5-minute file""")
    return


@app.cell
def _(dc, end_ts):
    lmp_5min_range = dc.get_range_data_interval_5min_lmps(
        end_ts=end_ts, n_periods=24 * 12
    )
    lmp_5min_parquet = [pf for pf in lmp_5min_range if pf.endswith(".parquet")]
    lmp_5min_parquet[:10]
    return (lmp_5min_parquet,)


@app.cell
def _(dc, lmp_5min_parquet):
    dc.upsert_mtlf_mtrf_lmp(lmp_5min_parquet, target="lmp")
    return


if __name__ == "__main__":
    app.run()

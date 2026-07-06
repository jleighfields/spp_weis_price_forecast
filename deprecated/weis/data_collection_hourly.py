# Hourly data collection for SPP WEIS price forecast.
#
# Collects MTLF, MTRF, and 5-min LMP data.
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_hourly.py
#   Script:      python notebooks/data_collection/data_collection_hourly.py
#   Modal:       modal run modal_jobs/data_collection.py::collect_hourly

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
    # Data collection hourly
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
    import logging

    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("py4j").setLevel(logging.ERROR)
    log = logging.getLogger(__name__)

    # Add project root to sys.path for src/ imports
    _project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    return log, pd


@app.cell
def _():
    import src.data_collection as dc

    return (dc,)


@app.cell
def _(pd):
    end_ts = pd.Timestamp.now("UTC").tz_convert("America/Chicago").tz_localize(None)
    end_ts
    return (end_ts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Load Forecast""")
    return


@app.cell
def _(dc, end_ts):
    mtlf_range = dc.get_range_data_mtlf(end_ts=end_ts, n_periods=24)
    mtlf_parquet = [pf for pf in mtlf_range if pf.endswith(".parquet")]
    mtlf_parquet[:10]
    return (mtlf_parquet,)


@app.cell
def _(dc, mtlf_parquet):
    if mtlf_parquet:
        dc.upsert_mtlf_mtrf_lmp(mtlf_parquet, target="mtlf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Resource Forecast""")
    return


@app.cell
def _(dc, end_ts):
    mtrf_range = dc.get_range_data_mtrf(end_ts=end_ts, n_periods=24)
    mtrf_parquet = [pf for pf in mtrf_range if pf.endswith(".parquet")]
    mtrf_parquet[:10]
    return (mtrf_parquet,)


@app.cell
def _(dc, mtrf_parquet):
    if mtrf_parquet:
        dc.upsert_mtlf_mtrf_lmp(mtrf_parquet, target="mtrf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP settlement location prices (5-min intervals)""")
    return


@app.cell
def _(dc, end_ts):
    lmp_range = dc.get_range_data_interval_5min_lmps(end_ts=end_ts, n_periods=24 * 12)
    lmp_parquet = [pf for pf in lmp_range if pf.endswith(".parquet")]
    lmp_parquet[:10]
    return (lmp_parquet,)


@app.cell
def _(dc, lmp_parquet):
    if lmp_parquet:
        dc.upsert_mtlf_mtrf_lmp(lmp_parquet, target="lmp")
    return


if __name__ == "__main__":
    app.run()

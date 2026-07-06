# Daily data collection for SPP WEIS price forecast.
#
# Collects daily LMP settlement data.
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_daily.py
#   Script:      python notebooks/data_collection/data_collection_daily.py
#   Modal:       modal run modal_jobs/data_collection.py::collect_daily

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
    # Data collection daily files
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP daily settlement data""")
    return


@app.cell
def _(pd):
    # Offset by 2 days to allow for settlement data availability
    end_ts = (
        pd.Timestamp.now("UTC").tz_convert("America/Chicago").tz_localize(None)
        - pd.Timedelta("2D")
    )
    end_ts
    return (end_ts,)


@app.cell
def _(dc, end_ts):
    # Lookback 7 days to cover holidays and long weekends
    range_df = dc.get_range_data_interval_daily_lmps(end_ts=end_ts, n_periods=7)
    parquet_files = [pf for pf in range_df if pf.endswith(".parquet")]
    parquet_files[:10]
    return (parquet_files,)


@app.cell
def _(dc, parquet_files):
    if parquet_files:
        dc.upsert_mtlf_mtrf_lmp(parquet_files, target="lmp")
    return


if __name__ == "__main__":
    app.run()

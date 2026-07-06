# 5-minute LMP data collection for SPP WEIS price forecast.
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_5_min.py
#   Script:      python notebooks/data_collection/data_collection_5_min.py

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
    # Data collection 5 minute intervals
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
    mo.md(r"""## LMP 5-minute interval files""")
    return


@app.cell
def _(pd):
    end_ts = pd.Timestamp.now("UTC").tz_convert("America/Chicago").tz_localize(None)
    end_ts
    return (end_ts,)


@app.cell
def _(dc, end_ts):
    range_df = dc.get_range_data_interval_5min_lmps(end_ts=end_ts, n_periods=24 * 12)
    parquet_files = [pf for pf in range_df if pf.endswith(".parquet")]
    parquet_files[:10]
    return (parquet_files,)


@app.cell
def _(dc, parquet_files):
    dc.upsert_mtlf_mtrf_lmp(parquet_files, target="lmp")
    return


if __name__ == "__main__":
    app.run()

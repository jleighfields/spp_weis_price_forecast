# Hourly data collection for SPP RTO West / Integrated Marketplace (IM).
# (Detail lives below the app definition on purpose: marimo's file browser
# only scans the first 512 bytes of a notebook for its app declaration, so a
# long header here would hide this file from the editor's workspace list.)

import marimo

__generated_with = "0.20.2"
app = marimo.App()

# Collects MTLF, MTRF, RF_RESERVE_ZONE, 5-min LMP, and Day-Ahead LMP into
# im/. DA (the primary forecast target) rides the hourly job so it stays as
# up to date as the real-time feed.
# Parallel to data_collection_hourly.py (the WEIS pipeline).
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_im_hourly.py
#   Script:      python notebooks/data_collection/data_collection_im_hourly.py
#   Modal:       modal run modal_jobs/data_collection_im.py::collect_im_hourly



@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # IM data collection — hourly
    Gather public SPP Integrated Marketplace data from https://portal.spp.org
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
    import src.data_collection_im as dcim

    return (dcim,)


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
def _(dcim, end_ts):
    mtlf_range = dcim.get_range_data_mtlf(end_ts=end_ts, n_periods=24)
    mtlf_parquet = [pf for pf in mtlf_range if pf.endswith(".parquet")]
    mtlf_parquet[:10]
    return (mtlf_parquet,)


@app.cell
def _(dcim, mtlf_parquet):
    if mtlf_parquet:
        dcim.upsert_im(mtlf_parquet, target="mtlf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Resource Forecast""")
    return


@app.cell
def _(dcim, end_ts):
    mtrf_range = dcim.get_range_data_mtrf(end_ts=end_ts, n_periods=24)
    mtrf_parquet = [pf for pf in mtrf_range if pf.endswith(".parquet")]
    mtrf_parquet[:10]
    return (mtrf_parquet,)


@app.cell
def _(dcim, mtrf_parquet):
    if mtrf_parquet:
        dcim.upsert_im(mtrf_parquet, target="mtrf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Resource forecast by reserve zone (wind/solar)""")
    return


@app.cell
def _(dcim, end_ts):
    rf_range = dcim.get_range_data_rf_reserve_zone(end_ts=end_ts, n_periods=24)
    rf_parquet = [pf for pf in rf_range if pf.endswith(".parquet")]
    rf_parquet[:10]
    return (rf_parquet,)


@app.cell
def _(dcim, rf_parquet):
    if rf_parquet:
        dcim.upsert_im(rf_parquet, target="rf_reserve_zone")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Day-Ahead LMP""")
    return


@app.cell
def _(dcim, end_ts, pd):
    # DA is the primary forecast target, so collect it here (every 4 hours)
    # rather than on the daily job — that picks up each day's file soon after
    # it publishes and keeps DA as current as the real-time feed. The file for
    # operating day D publishes the prior afternoon, so look ahead one day to
    # grab tomorrow's file; a 3-day window (yesterday..tomorrow) repairs any
    # late-published day without re-fetching a wide range every run.
    #
    # Ordered before the 288-file 5-min LMP fetch below: DA is only ~3 files,
    # so collecting it first guarantees the primary target lands even when the
    # portal is slow and the RT fetch runs long (which would otherwise burn the
    # whole Modal timeout before DA ever ran).
    da_range = dcim.get_range_data_da_lmp(
        end_ts=end_ts + pd.Timedelta(days=1), n_periods=3
    )
    da_parquet = [pf for pf in da_range if pf.endswith(".parquet")]
    da_parquet[:10]
    return (da_parquet,)


@app.cell
def _(dcim, da_parquet):
    if da_parquet:
        dcim.upsert_im(da_parquet, target="da_lmp")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP settlement location prices (5-min intervals)""")
    return


@app.cell
def _(dcim, end_ts):
    # 24 hours x 12 five-minute intervals
    lmp_range = dcim.get_range_data_5min_lmp(end_ts=end_ts, n_periods=24 * 12)
    lmp_parquet = [pf for pf in lmp_range if pf.endswith(".parquet")]
    lmp_parquet[:10]
    return (lmp_parquet,)


@app.cell
def _(dcim, lmp_parquet):
    if lmp_parquet:
        dcim.upsert_im(lmp_parquet, target="lmp")
    return


if __name__ == "__main__":
    app.run()

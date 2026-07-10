# Daily data collection for SPP RTO West / Integrated Marketplace (IM).
#
# Runs the daily-LMP repair sweep: fills the 5-min LMP history from the
# lag-published daily rollups (~D+5), which the hourly job's live 5-min
# feed can't reach. Also re-collects Day-Ahead LMP over a wider window as a
# gap-catch backstop — DA rides the hourly job for freshness, and this daily
# pass repairs any run the hourly job missed.
# Parallel to data_collection_daily.py (the WEIS pipeline).
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_im_daily.py
#   Script:      python notebooks/data_collection/data_collection_im_daily.py
#   Modal:       modal run modal_jobs/data_collection_im.py::collect_im_daily

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
    # IM data collection — daily
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
    now = pd.Timestamp.now("UTC").tz_convert("America/Chicago").tz_localize(None)
    now
    return (now,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Daily LMP rollup (repair sweep)""")
    return


@app.cell
def _(dcim, now):
    # get_range_data_daily_lmp applies the ~D+5 publication lag internally,
    # so pass the current time. A 7-day window overlaps this job's 3-day
    # cadence, so one missed run (or a late-publishing day) leaves no gap.
    daily_range = dcim.get_range_data_daily_lmp(end_ts=now, n_periods=7)
    daily_parquet = [pf for pf in daily_range if pf.endswith(".parquet")]
    daily_parquet[:10]
    return (daily_parquet,)


@app.cell
def _(dcim, daily_parquet):
    if daily_parquet:
        dcim.upsert_im(daily_parquet, target="lmp")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Day-Ahead LMP (gap-catch)""")
    return


@app.cell
def _(dcim, now, pd):
    # DA is collected primarily on the hourly job (fresh, tight window). This
    # daily pass is a wider-window backstop: it re-scans ~6 days so a run the
    # hourly job missed (portal outage, a late-published file) still gets
    # filled. The upsert is idempotent, so the overlap is harmless. Look ahead
    # one day to include tomorrow's file (publishes the prior afternoon).
    da_range = dcim.get_range_data_da_lmp(
        end_ts=now + pd.Timedelta(days=1), n_periods=6
    )
    da_parquet = [pf for pf in da_range if pf.endswith(".parquet")]
    da_parquet[:10]
    return (da_parquet,)


@app.cell
def _(dcim, da_parquet):
    if da_parquet:
        dcim.upsert_im(da_parquet, target="da_lmp")
    return


if __name__ == "__main__":
    app.run()

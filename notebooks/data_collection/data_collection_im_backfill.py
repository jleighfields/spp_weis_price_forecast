# Historical backfill for SPP RTO West / Integrated Marketplace (IM).
#
# One-time backfill of MTLF, MTRF, RF_RESERVE_ZONE, and LMP (via the daily
# rollup) from BACKFILL_START to now into data_im/. Covers both Phase 2
# history segments in one pass — the collectors filter LMP to STORED_NODES and
# fill BAA='SPP' for pre-launch (East-only) files, so:
#   * 2026-04-01 -> now: both BAAs, all hub/BA nodes (IM era)
#   * 2025-04-01 -> 2026-03-31: East hubs only, BAA='SPP' (pre-launch East era)
# The WEIS West stitch (<= 2026-03-31, BAA='SWPW') is a separate one-time
# script: scripts/weis_stitch_fill.py.
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_im_backfill.py
#   Script:      python notebooks/data_collection/data_collection_im_backfill.py

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
    # IM data collection — historical backfill
    One-time backfill of MTLF, MTRF, RF reserve zone, and LMP into data_im/.
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

    assert os.environ.get("AWS_S3_BUCKET")

    return log, pd


@app.cell
def _():
    import src.data_collection_im as dcim

    return (dcim,)


@app.cell
def _(dcim, pd):
    # One year before the RTO West launch, so the East BAA also has
    # >= 365 days of history once the seam is crossed.
    BACKFILL_START = dcim.RTO_WEST_LAUNCH - pd.DateOffset(years=1)
    now = pd.Timestamp.now("UTC").tz_convert("America/Chicago").tz_localize(None)

    # Hourly feeds (MTLF/MTRF/RF) are one file per hour; LMP uses the daily
    # rollup, one file per day (get_range_data_daily_lmp applies its own lag).
    # The small buffers (+24h, +10d, +2d) over-cover the start of the window
    # so the range reaches slightly past BACKFILL_START and never leaves a
    # boundary hole from rounding/lag; duplicate fetches are idempotent upserts.
    n_hours = int((now - BACKFILL_START).total_seconds() // 3600) + 24
    n_days = (now - BACKFILL_START).days + 10

    # Day-Ahead market only exists from the RTO West launch onward.
    DA_START = dcim.RTO_WEST_LAUNCH
    n_da_days = (now - DA_START).days + 2
    print(
        f"backfill {BACKFILL_START.date()} -> {now.date()}: {n_hours} hours, {n_days} days"
    )
    return n_da_days, n_days, n_hours, now


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Load Forecast""")
    return


@app.cell
def _(dcim, n_hours, now):
    mtlf_range = dcim.get_range_data_mtlf(end_ts=now, n_periods=n_hours)
    mtlf_parquet = [pf for pf in mtlf_range if pf.endswith(".parquet")]
    len(mtlf_parquet)
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
def _(dcim, n_hours, now):
    mtrf_range = dcim.get_range_data_mtrf(end_ts=now, n_periods=n_hours)
    mtrf_parquet = [pf for pf in mtrf_range if pf.endswith(".parquet")]
    len(mtrf_parquet)
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
def _(dcim, n_hours, now):
    rf_range = dcim.get_range_data_rf_reserve_zone(end_ts=now, n_periods=n_hours)
    rf_parquet = [pf for pf in rf_range if pf.endswith(".parquet")]
    len(rf_parquet)
    return (rf_parquet,)


@app.cell
def _(dcim, rf_parquet):
    if rf_parquet:
        dcim.upsert_im(rf_parquet, target="rf_reserve_zone")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP via daily rollup (both eras)""")
    return


@app.cell
def _(dcim, n_days, now):
    lmp_range = dcim.get_range_data_daily_lmp(end_ts=now, n_periods=n_days)
    lmp_parquet = [pf for pf in lmp_range if pf.endswith(".parquet")]
    len(lmp_parquet)
    return (lmp_parquet,)


@app.cell
def _(dcim, lmp_parquet):
    if lmp_parquet:
        dcim.upsert_im(lmp_parquet, target="lmp")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Day-Ahead LMP (post-launch only)""")
    return


@app.cell
def _(dcim, n_da_days, now, pd):
    da_range = dcim.get_range_data_da_lmp(
        end_ts=now + pd.Timedelta(days=1), n_periods=n_da_days
    )
    da_parquet = [pf for pf in da_range if pf.endswith(".parquet")]
    len(da_parquet)
    return (da_parquet,)


@app.cell
def _(dcim, da_parquet):
    if da_parquet:
        dcim.upsert_im(da_parquet, target="da_lmp")
    return


if __name__ == "__main__":
    app.run()

# Testing notebook for SPP WEIS data collection functions.
#
# Exercises URL generation, CSV fetching, range queries, and upsert
# for all data types (MTLF, MTRF, LMP daily, LMP 5-min).
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_testing.py

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
    # Data collection testing
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
    mo.md(r"""## URL generation tests""")
    return


@app.cell
def _(dc):
    tc = dc.get_time_components("6/7/2025 08:00:00")
    return (tc,)


@app.cell
def _(dc, tc):
    mtlf_url = dc.get_hourly_mtlf_url(tc)
    print(mtlf_url)
    mtlf_url.split("WEIS-")[-1].replace(".csv", ".parquet")
    return (mtlf_url,)


@app.cell
def _(dc, tc):
    mtrf_url = dc.get_hourly_mtrf_url(tc)
    print(mtrf_url)
    mtrf_url.split("WEIS-")[-1].replace(".csv", ".parquet")
    return


@app.cell
def _(dc, tc):
    lmp_5min = dc.get_5min_lmp_url(tc)
    print(lmp_5min)
    lmp_5min.split("WEIS-")[-1].replace(".csv", ".parquet")
    return


@app.cell
def _(dc, tc):
    lmp_daily = dc.get_daily_lmp_url(tc)
    print(lmp_daily)
    lmp_daily.split("WEIS-")[-1].replace(".csv", ".parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## CSV fetch tests""")
    return


@app.cell
def _(dc, mtlf_url):
    # test error handling - bad suffix
    _df = dc.get_csv_from_url(mtlf_url + "bad_url")
    _df
    return


@app.cell
def _(dc, mtlf_url):
    # test error handling - bad prefix
    _df = dc.get_csv_from_url("a" + mtlf_url)
    _df
    return


@app.cell
def _(dc, mtlf_url):
    # test success
    _df = dc.get_csv_from_url(mtlf_url)
    _df
    return


@app.cell
def _(dc, tc):
    dc.get_process_mtlf(tc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## MTLF range test""")
    return


@app.cell
def _(dc, pd):
    end_ts = (
        pd.Timestamp.now("UTC").tz_convert("America/Chicago").tz_localize(None)
        - pd.Timedelta("2D")
    )

    range_df = dc.get_range_data_mtlf(end_ts=end_ts, n_periods=24)
    range_df[:10]
    return end_ts, range_df


@app.cell
def _(pl, range_df):
    pl.read_parquet(range_df[0])
    return


@app.cell
def _(range_df):
    parquet_files = [pf for pf in range_df if pf.endswith(".parquet")]
    parquet_files[:10]
    return (parquet_files,)


@app.cell
def _(dc, parquet_files):
    dc.upsert_mtlf_mtrf_lmp(parquet_files, target="mtlf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## MTRF range test""")
    return


@app.cell
def _(dc, end_ts):
    mtrf_range = dc.get_range_data_mtrf(end_ts=end_ts, n_periods=24)
    mtrf_parquet = [pf for pf in mtrf_range if pf.endswith(".parquet")]
    mtrf_parquet[:10]
    return (mtrf_parquet,)


@app.cell
def _(dc, mtrf_parquet):
    dc.upsert_mtlf_mtrf_lmp(mtrf_parquet, target="mtrf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP daily test""")
    return


@app.cell
def _(dc, end_ts):
    lmp_daily_range = dc.get_range_data_interval_daily_lmps(end_ts=end_ts, n_periods=7)
    lmp_daily_parquet = [pf for pf in lmp_daily_range if pf.endswith(".parquet")]
    lmp_daily_parquet[:10]
    return (lmp_daily_parquet,)


@app.cell
def _(dc, lmp_daily_parquet):
    dc.upsert_mtlf_mtrf_lmp(lmp_daily_parquet, target="lmp")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP 5-minute test""")
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

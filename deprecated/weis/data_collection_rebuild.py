# Rebuild data from S3 raw files for SPP WEIS price forecast.
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_rebuild.py
#   Script:      python notebooks/data_collection/data_collection_rebuild.py

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
    # Data collection rebuild
    Gather public SPP Weis data from https://marketplace.spp.org/groups/operational-data-weis
    """
    )
    return


@app.cell
def _():
    import os
    import sys
    import pathlib
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

    return (log,)


@app.cell
def _():
    import src.data_collection as dc

    return (dc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Load Forecast""")
    return


@app.cell
def _(dc):
    dc.rebuild_mtlf_mtrf_lmp_from_s3(src_dir="mtlf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mid Term Resource Forecast""")
    return


@app.cell
def _(dc):
    dc.rebuild_mtlf_mtrf_lmp_from_s3(src_dir="mtrf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## LMP daily file""")
    return


@app.cell
def _(dc):
    dc.rebuild_mtlf_mtrf_lmp_from_s3(src_dir="lmp_daily")
    return


if __name__ == "__main__":
    app.run()

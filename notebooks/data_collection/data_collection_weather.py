# Weather data collection for SPP WEIS price forecast.
#
# Usage:
#   Interactive: marimo edit notebooks/data_collection/data_collection_weather.py
#   Script:      python notebooks/data_collection/data_collection_weather.py

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
    # Data collection weather
    Gather public SPP Weis data from https://marketplace.spp.org/groups/operational-data-weis
    """
    )
    return


@app.cell
def _():
    import os
    import sys
    import pathlib
    import duckdb
    import logging

    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    _project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    return duckdb, log, os


@app.cell
def _():
    import src.data_collection as dc

    return (dc,)


@app.cell
def _():
    # dc.upsert_weather()
    return


@app.cell
def _(duckdb):
    con = duckdb.connect("data/spp.ddb")
    con.execute("SHOW TABLES").fetchall()
    return (con,)


@app.cell
def _(con):
    con.execute("SELECT * FROM weather").pl()
    return


@app.cell
def _(con):
    con.close()
    return


if __name__ == "__main__":
    app.run()

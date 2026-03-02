"""Modal data collection jobs for SPP WEIS price forecast.

Based on the Databricks data collection notebooks:
- collect_hourly: notebooks/data_collection/data_collection_hourly.ipynb
- collect_daily:  notebooks/data_collection/data_collection_daily.ipynb

Test:  modal run modal_jobs/data_collection.py::collect_hourly
Deploy: modal deploy modal_jobs/data_collection.py

TODO: Migrate data collection notebooks from Jupyter to marimo and update
      these Modal jobs to use the marimo notebooks directly.
"""

import modal

app = modal.App("spp-weis-data-collection")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "polars==1.37.1",
        "pyarrow==19.0.1",
        "boto3==1.35.92",
        "duckdb==1.4.3",
        "requests",
        "tqdm==4.67.1",
        "polars-xdt==0.17.1",
        "pandas",
        "joblib",
    )
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    schedule=modal.Period(hours=6),
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1800,
)
def collect_hourly():
    """Collect MTLF, MTRF, and 5-min LMP data."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")
    import pandas as pd

    import src.data_collection as dc

    end_ts = pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)

    # MTLF
    range_df = dc.get_range_data_mtlf(end_ts=end_ts, n_periods=24)
    parquet_files = [pf for pf in range_df if pf.endswith(".parquet")]
    if parquet_files:
        dc.upsert_mtlf_mtrf_lmp(parquet_files, target="mtlf")

    # MTRF
    range_df = dc.get_range_data_mtrf(end_ts=end_ts, n_periods=24)
    parquet_files = [pf for pf in range_df if pf.endswith(".parquet")]
    if parquet_files:
        dc.upsert_mtlf_mtrf_lmp(parquet_files, target="mtrf")

    # LMP 5-min
    range_df = dc.get_range_data_interval_5min_lmps(end_ts=end_ts, n_periods=24 * 12)
    parquet_files = [pf for pf in range_df if pf.endswith(".parquet")]
    if parquet_files:
        dc.upsert_mtlf_mtrf_lmp(parquet_files, target="lmp")


@app.function(
    image=image,
    schedule=modal.Period(days=3),
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1800,
)
def collect_daily():
    """Collect daily LMP settlement data."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")
    import pandas as pd

    import src.data_collection as dc

    end_ts = (
        pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        - pd.Timedelta("2D")
    )

    range_df = dc.get_range_data_interval_daily_lmps(end_ts=end_ts, n_periods=7)
    parquet_files = [pf for pf in range_df if pf.endswith(".parquet")]
    if parquet_files:
        dc.upsert_mtlf_mtrf_lmp(parquet_files, target="lmp")

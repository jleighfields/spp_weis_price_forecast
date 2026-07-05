"""Modal data collection jobs for SPP RTO West / Integrated Marketplace (IM).

Thin wrappers that run the IM marimo notebooks headlessly. The notebooks at
notebooks/data_collection/data_collection_im_*.py contain all collection
logic. Parallel to modal_jobs/data_collection.py (the WEIS pipeline); see
plans/weis_to_rto_west_migration.md.

Test:  modal run modal_jobs/data_collection_im.py::collect_im_hourly
Deploy: modal deploy modal_jobs/data_collection_im.py
"""

import modal

app = modal.App("spp-im-data-collection")

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
        "pytz",
        "joblib",
        "marimo",
        "python-dotenv",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("notebooks", remote_path="/root/notebooks")
)


@app.function(
    image=image,
    schedule=modal.Period(hours=4),
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1800,
    cpu=16.0,  # 16 physical cores for joblib parallel processing
    memory=4096,  # 4 GiB
    env={"MAX_JOBS": "15"},
)
def collect_im_hourly():
    """Collect IM MTLF, MTRF, RF_RESERVE_ZONE, and 5-min LMP data."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from notebooks.data_collection.data_collection_im_hourly import app as notebook_app

    notebook_app.run()


@app.function(
    image=image,
    schedule=modal.Period(days=1),
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1800,
    cpu=16.0,  # 16 physical cores for joblib parallel processing
    memory=4096,  # 4 GiB
    env={"MAX_JOBS": "15"},
)
def collect_im_daily():
    """Run the daily-LMP repair sweep and collect Day-Ahead LMP data."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from notebooks.data_collection.data_collection_im_daily import app as notebook_app

    notebook_app.run()

"""Modal data collection jobs for SPP WEIS price forecast.

Thin wrappers that run the marimo notebooks headlessly.
The notebooks at notebooks/data_collection/ contain all collection logic.

Test:  modal run modal_jobs/data_collection.py::collect_hourly
Deploy: modal deploy modal_jobs/data_collection.py
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
def collect_hourly():
    """Collect MTLF, MTRF, and 5-min LMP data."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from notebooks.data_collection.data_collection_hourly import app as notebook_app

    notebook_app.run()


@app.function(
    image=image,
    schedule=modal.Period(days=3),
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=1800,
    cpu=16.0,  # 16 physical cores for joblib parallel processing
    memory=4096,  # 4 GiB
    env={"MAX_JOBS": "15"},
)
def collect_daily():
    """Collect daily LMP settlement data."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from notebooks.data_collection.data_collection_daily import app as notebook_app

    notebook_app.run()

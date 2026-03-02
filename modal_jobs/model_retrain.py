"""Modal model retrain job for SPP WEIS price forecast.

Thin wrapper that runs the marimo notebook headlessly.
The notebook at notebooks/model_training/model_retrain.py contains all
retrain logic (data prep, training, S3 upload, champion promotion).

Test:  modal run modal_jobs/model_retrain.py::model_retrain_weekly
Deploy: modal deploy modal_jobs/model_retrain.py
"""

import modal

app = modal.App("spp-weis-model-retrain")

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
        "torch==2.5.1",
        "darts==0.41.0",
        "pytorch-lightning",
        "scikit-learn",
        "marimo",
        "python-dotenv",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("notebooks", remote_path="/root/notebooks")
)


@app.function(
    image=image,
    schedule=modal.Cron("0 20 * * 0"),  # Sundays at 8 PM UTC
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=7200,  # 2 hours
    cpu=8.0,  # 8 physical cores
    memory=32768,  # 32 GiB
    gpu="A10G",
)
def model_retrain_weekly():
    """Retrain ensemble models (TiDE, TSMixer, TFT) and promote champion."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from notebooks.model_training.model_retrain import app as notebook_app

    notebook_app.run()

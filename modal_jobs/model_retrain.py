"""Modal model retrain jobs for the SPP price forecast (day-ahead + real-time).

Thin wrappers that run the marimo retrain notebook headlessly, once per
forecast target. The notebook at notebooks/model_training/model_retrain.py
contains all retrain logic (data prep, training, S3 upload, champion promotion)
and selects its target from the TARGET env var set below; each target trains
from its own source table into its own models/<target>/ namespace.

Test:   modal run modal_jobs/model_retrain.py::retrain_da_weekly
Deploy: modal deploy modal_jobs/model_retrain.py
"""

import modal

app = modal.App("spp-weis-model-retrain")

# The bucket name is not secret, so it lives here in code (env= below) rather
# than in the aws-secret. Credentials, S3_ENDPOINT_URL, AWS_DEFAULT_REGION, and
# AWS_S3_FOLDER still come from the aws-secret; a bucket change is a redeploy.
S3_BUCKET = "spp-rto"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install the exact deploy pins (requirements.txt) so the retrain stack —
    # Darts / torch / lightning / marimo — can NOT drift from the app. A
    # hand-maintained pin list drifted once (Darts 0.41 vs the app's 0.45) and
    # trained a broken champion; requirements.txt is the single source of truth
    # for these pins, and resolves to x86 wheels (correct for the A10G below).
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("notebooks", remote_path="/root/notebooks")
)

# Shared function config for both targets (only the schedule and TARGET differ).
_COMMON = dict(
    image=image,
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=7200,  # 2 hours
    cpu=8.0,  # 8 physical cores
    memory=32768,  # 32 GiB
    gpu="A10G",
)


def _run_notebook():
    """Run the retrain notebook headlessly; it reads TARGET from the env."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from notebooks.model_training.model_retrain import app as notebook_app

    notebook_app.run()


@app.function(
    **_COMMON,
    schedule=modal.Cron("0 20 * * 0"),  # Sundays 20:00 UTC
    env={"AWS_S3_BUCKET": S3_BUCKET, "TARGET": "da"},
)
def retrain_da_weekly():
    """Retrain the day-ahead (DA) ensemble and promote its champion (primary)."""
    _run_notebook()


@app.function(
    **_COMMON,
    schedule=modal.Cron("0 22 * * 0"),  # Sundays 22:00 UTC (staggered after DA)
    env={"AWS_S3_BUCKET": S3_BUCKET, "TARGET": "rt"},
)
def retrain_rt_weekly():
    """Retrain the real-time (RT) ensemble and promote its champion."""
    _run_notebook()

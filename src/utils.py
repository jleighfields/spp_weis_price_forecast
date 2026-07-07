"""
AWS S3 utility functions for SPP WEIS Price Forecast.

This module provides helper functions for interacting with AWS S3 storage,
including listing bucket contents and retrieving trained model file paths.
These utilities support the migration from hardcoded S3 paths to configurable
bucket/folder locations via environment variables.
"""

import json
import os
from typing import List
import boto3

# logging
import logging

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── R2 model-storage layout ──────────────────────────────────────────────
# Single source of truth for where retrains are saved and where the live
# champion pointer lives, relative to AWS_S3_FOLDER. Shared by the app read
# path below, the model_retrain notebook (write path), and
# scripts/r2_promote_champion.py so a layout change has exactly one home.
RETRAINS_PREFIX = "model_retrains/"
CHAMPION_KEY_SUFFIX = "S3_models/champion.json"


def list_folder_contents_resource(bucket_name: str, folder_prefix: str):
    """
    Lists all objects within a specific 'folder' in an S3 bucket using the resource API.

    Uses boto3's resource API (higher-level abstraction) rather than the client API
    to provide an iterable collection of S3 ObjectSummary objects.

    Args:
        bucket_name: The name of the S3 bucket (e.g., from AWS_S3_BUCKET env var).
        folder_prefix: The prefix (folder path) to list objects under. A trailing
            slash will be appended if not present to ensure folder-level filtering.

    Returns:
        boto3.resources.collection.s3.Bucket.objectsCollection: An iterable collection
            of S3 ObjectSummary objects. Each object has attributes like 'key', 'size',
            'last_modified', etc.
    """
    s3 = boto3.resource("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))
    bucket = s3.Bucket(bucket_name)

    # Ensure the prefix ends with a slash to limit results to a specific "folder"
    if not folder_prefix.endswith("/") and folder_prefix != "":
        folder_prefix += "/"

    print(f"Listing objects in s3://{bucket_name}/{folder_prefix}")

    bucket_contents = bucket.objects.filter(Prefix=folder_prefix)

    return bucket_contents


def get_loaded_models(search_folder: str = "S3_models/") -> List[str]:
    """
    Retrieves a list of trained model file paths from S3.

    Scans the configured S3 bucket/folder for model files stored in the 'S3_models/'
    subdirectory. Supports common model serialization formats: pickle (.pkl),
    PyTorch Lightning checkpoints (.ckpt), and PyTorch state dicts (.pt).

    Environment Variables:
        AWS_S3_BUCKET: The S3 bucket name containing model files.
        AWS_S3_FOLDER: The folder prefix within the bucket.

    Returns:
        List[str]: S3 keys (paths) for all model files found in the search_folder directory.
    """
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER", "")
    folder_prefix = AWS_S3_FOLDER + search_folder
    log.info(f"{AWS_S3_BUCKET = }")
    log.info(f"{AWS_S3_FOLDER = }")
    log.info(f"{folder_prefix = }")

    bucket_contents = list_folder_contents_resource(AWS_S3_BUCKET, folder_prefix)
    # Filter for objects in the S3_models/ subdirectory
    loaded_models = [d.key for d in bucket_contents if search_folder in d.key]
    # Filter for recognized model file extensions
    loaded_models = [
        lm
        for lm in loaded_models
        if ((".pkl" in lm) or (".ckpt" in lm) or (".pt" in lm))
    ]
    log.info(f"loaded_models: {loaded_models}")

    return loaded_models


def download_checkpoints(s3_folder: str, dest_dir: str) -> None:
    """Download model checkpoint files from an S3 folder to a local directory.

    Lists all model files (.pt, .ckpt, .pkl) in the given S3 folder using
    ``get_loaded_models`` and downloads them into ``dest_dir``. Works with
    any model folder — champion, challenger, or historical.

    Environment Variables:
        AWS_S3_BUCKET: The S3 bucket name.
        S3_ENDPOINT_URL: Endpoint URL for S3-compatible storage (e.g. Cloudflare R2).

    Args:
        s3_folder: S3 folder path containing checkpoint files
            (e.g. ``"S3_models/2026-03-01_12-00-00/"``).
        dest_dir: Local directory to download checkpoint files into.
    """
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    s3_client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))

    log.info(f"downloading model checkpoints from: {s3_folder}")
    model_keys = get_loaded_models(s3_folder)
    log.info(f"model files to download: {model_keys}")

    for key in model_keys:
        local_file = os.path.join(dest_dir, key.split("/")[-1])
        log.info(f"downloading: {key} to {local_file}")
        s3_client.download_file(Bucket=AWS_S3_BUCKET, Key=key, Filename=local_file)


def build_champion_config(
    folder_time: str, artifact_folder: str, artifact_path: str
) -> dict:
    """Build the champion.json payload that points the app at a model folder.

    Single home for the champion.json schema: written by the model_retrain
    notebook and scripts/r2_promote_champion.py, and read back by
    download_champion_checkpoints (which consumes champion_artifact_folder).
    Callers pass their already-computed path pieces so nothing is recomputed
    here — this owns only the key names / structure.

    Args:
        folder_time: Timestamped folder name with a trailing slash
            (e.g. "2026-07-06_12-41-45/").
        artifact_folder: RETRAINS_PREFIX + folder_time — the folder relative
            to AWS_S3_FOLDER that the app loads checkpoints from.
        artifact_path: AWS_S3_FOLDER + artifact_folder — the full-prefixed path.

    Returns:
        The three-key champion config dict (champion / champion_artifact_folder
        / champion_artifact_path).
    """
    return {
        "champion": folder_time,
        "champion_artifact_folder": artifact_folder,
        "champion_artifact_path": artifact_path,
    }


# Name of the per-model provenance/validation file saved next to the
# checkpoints in each model_retrains/<ts>/ folder.
TRAINING_CONFIG_FILENAME = "training_config.json"


def build_training_config(
    train_timestamp: str,
    future_covariates: List[str],
    past_covariates: List[str],
    nodes: List[str],
    quantiles: List[float],
    model_name: str,
    model_types: List[str],
    forecast_horizon: int,
    input_chunk_length: int,
    train_start: str,
    train_end: str,
    darts_version: str,
    torch_version: str,
) -> dict:
    """Build the ``training_config.json`` payload saved alongside a model.

    Single home for the schema. Records what the model was trained on — most
    importantly the exact past/future covariate lists — so the serving code can
    verify it still builds the same inputs before trusting the model (see
    ``validate_model_covariates``). Written by the retrain and Optuna notebooks;
    read back by the app at load.

    Args:
        train_timestamp: When the model was trained (ISO string).
        future_covariates: Future covariate column names the model expects
            (``FUTR_COLS``), in order.
        past_covariates: Past covariate column names the model expects
            (``PAST_COLS``), in order.
        nodes: The node scope trained/served (``MODEL_APP_NODES``).
        quantiles: The QuantileRegression quantile levels.
        model_name: Model family name (``parameters.MODEL_NAME``).
        model_types: Which model builders contributed members (e.g. ``["tide"]``).
        forecast_horizon: Output length in hours.
        input_chunk_length: Input length in hours.
        train_start: First timestamp in the training window (ISO string).
        train_end: Last timestamp in the training window (ISO string).
        darts_version: Darts version the checkpoints were written with.
        torch_version: torch version the checkpoints were written with.

    Returns:
        The training-config dict (JSON-serializable).
    """
    return {
        "model_name": model_name,
        "train_timestamp": train_timestamp,
        "train_start": train_start,
        "train_end": train_end,
        "future_covariates": list(future_covariates),
        "past_covariates": list(past_covariates),
        "nodes": list(nodes),
        "quantiles": list(quantiles),
        "model_types": list(model_types),
        "forecast_horizon": forecast_horizon,
        "input_chunk_length": input_chunk_length,
        "darts_version": darts_version,
        "torch_version": torch_version,
    }


def load_training_config(model_dir: str) -> dict | None:
    """Read ``training_config.json`` from a local model dir; None if absent.

    Returns None for models saved before the config existed, so callers can
    treat validation as best-effort on legacy artifacts.
    """
    path = os.path.join(model_dir, TRAINING_CONFIG_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def validate_model_covariates(
    config: dict, future_covariates: List[str], past_covariates: List[str]
) -> None:
    """Raise if the serving code's covariate lists differ from the model's.

    Compares the current code's ``FUTR_COLS`` / ``PAST_COLS`` against what the
    model was trained on (per its ``training_config.json``) and fails up front
    with a clear message, instead of the cryptic ``mismatch between number of
    components ... component_mask`` error darts raises deep inside prediction.

    Args:
        config: A training-config dict from ``load_training_config``.
        future_covariates: The future covariate names the serving code builds.
        past_covariates: The past covariate names the serving code builds.

    Raises:
        ValueError: If either covariate list (order-sensitive) disagrees with
            the config.
    """
    for kind, current, expected in (
        ("future", list(future_covariates), config.get("future_covariates")),
        ("past", list(past_covariates), config.get("past_covariates")),
    ):
        if expected is not None and current != list(expected):
            raise ValueError(
                f"{kind} covariate mismatch: the model was trained on "
                f"{list(expected)} but the serving code builds {current}. "
                "Promote a model retrained with the current covariates, or "
                "align the code."
            )


def download_champion_checkpoints(dest_dir: str) -> None:
    """Download the current champion model's checkpoint files from S3.

    Reads ``S3_models/champion.json`` to determine which model folder is
    the current champion, then delegates to ``download_checkpoints``.

    Environment Variables:
        AWS_S3_BUCKET: The S3 bucket name.
        AWS_S3_FOLDER: The folder prefix within the bucket.
        S3_ENDPOINT_URL: Endpoint URL for S3-compatible storage (e.g. Cloudflare R2).

    Args:
        dest_dir: Local directory to download checkpoint files into.
    """
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER", "")
    s3_client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))

    champion_key = AWS_S3_FOLDER + CHAMPION_KEY_SUFFIX
    log.info(f"loading champion config from: {champion_key}")
    response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=champion_key)
    champion_config = json.loads(response["Body"].read().decode("utf-8"))
    log.info(f"champion_config: {champion_config}")

    champion_folder = champion_config["champion_artifact_folder"]
    download_checkpoints(champion_folder, dest_dir)


def get_parquet_files() -> List[str]:
    """
    Retrieves a list of parquet file paths from the configured S3 bucket.

    Scans the configured S3 bucket/folder for all parquet files and returns
    their S3 keys. Used by upsert functions to check if existing data files
    are available before performing upsert operations.

    Environment Variables:
        AWS_S3_BUCKET: The S3 bucket name containing data files.
        AWS_S3_FOLDER: The folder prefix within the bucket.

    Returns:
        List[str]: S3 keys (paths) for all parquet files found in the configured location.
    """
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER", "")
    bucket_contents = list_folder_contents_resource(AWS_S3_BUCKET, AWS_S3_FOLDER)
    parquet_files = [d.key for d in bucket_contents if ".parquet" in d.key]

    return parquet_files

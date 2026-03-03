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
    s3 = boto3.resource('s3', endpoint_url=os.getenv("S3_ENDPOINT_URL"))
    bucket = s3.Bucket(bucket_name)

    # Ensure the prefix ends with a slash to limit results to a specific "folder"
    if not folder_prefix.endswith('/') and folder_prefix != "":
        folder_prefix += '/'

    print(f"Listing objects in s3://{bucket_name}/{folder_prefix}")

    bucket_contents = bucket.objects.filter(Prefix=folder_prefix)

    return bucket_contents

def get_loaded_models(search_folder: str='S3_models/') -> List[str]:
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
    log.info(f'{AWS_S3_BUCKET = }')
    log.info(f'{AWS_S3_FOLDER = }')
    log.info(f'{folder_prefix = }')

    bucket_contents = list_folder_contents_resource(AWS_S3_BUCKET, folder_prefix)
    # Filter for objects in the S3_models/ subdirectory
    loaded_models = [d.key for d in bucket_contents if search_folder in d.key]
    # Filter for recognized model file extensions
    loaded_models = [lm for lm in loaded_models if (('.pkl' in lm) or ('.ckpt' in lm) or ('.pt' in lm))]
    log.info(f'loaded_models: {loaded_models}')

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
    s3_client = boto3.client('s3', endpoint_url=os.getenv("S3_ENDPOINT_URL"))

    log.info(f'downloading model checkpoints from: {s3_folder}')
    model_keys = get_loaded_models(s3_folder)
    log.info(f'model files to download: {model_keys}')

    for key in model_keys:
        local_file = os.path.join(dest_dir, key.split('/')[-1])
        log.info(f'downloading: {key} to {local_file}')
        s3_client.download_file(Bucket=AWS_S3_BUCKET, Key=key, Filename=local_file)


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
    s3_client = boto3.client('s3', endpoint_url=os.getenv("S3_ENDPOINT_URL"))

    champion_key = AWS_S3_FOLDER + "S3_models/champion.json"
    log.info(f'loading champion config from: {champion_key}')
    response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=champion_key)
    champion_config = json.loads(response['Body'].read().decode('utf-8'))
    log.info(f'champion_config: {champion_config}')

    champion_folder = champion_config['champion_artifact_folder']
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
    parquet_files = [d.key for d in bucket_contents if '.parquet' in d.key]

    return parquet_files
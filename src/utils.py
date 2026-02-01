"""
AWS S3 utility functions for SPP WEIS Price Forecast.

This module provides helper functions for interacting with AWS S3 storage,
including listing bucket contents and retrieving trained model file paths.
These utilities support the migration from hardcoded S3 paths to configurable
bucket/folder locations via environment variables.
"""
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
    s3 = boto3.resource('s3')
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
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER")
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
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER")
    bucket_contents = list_folder_contents_resource(AWS_S3_BUCKET, AWS_S3_FOLDER)
    parquet_files = [d.key for d in bucket_contents if '.parquet' in d.key]

    return parquet_files
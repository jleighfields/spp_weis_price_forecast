"""
Shared, feed-agnostic helpers for SPP data collection.

These utilities are used by both the legacy WEIS collector (data_collection.py)
and the live Integrated Marketplace / RTO West collector
(data_collection_im.py). They know nothing about a specific feed's URLs,
schema, or upsert keys — they cover S3 existence checks, parallel dispatch
with a progress bar, HTTP CSV reads, and small timestamp/column reshapes.

Keeping them here gives each helper a single home, so the live IM collector
does not have to import from the retired WEIS module.

Dependencies:
    - polars / polars_xdt: DataFrame reshapes
    - requests: HTTP CSV reads from the SPP portal
    - boto3: S3 object existence checks
    - joblib: parallel dispatch
"""
# pylint: disable=C0103,W1203,W1201

import os
from time import sleep
from io import StringIO
from typing import List

import tqdm
import requests
import polars as pl
import polars_xdt as xdt
import boto3
from botocore.exceptions import ClientError
from joblib import Parallel, cpu_count

import logging
log = logging.getLogger(__name__)

# Cap parallel fetches at (cores - 1), or MAX_JOBS when set (Modal pins it).
core_count = cpu_count()
max_jobs = int(os.environ.get('MAX_JOBS', 0))
N_JOBS = max_jobs if max_jobs > 0 else max(1, core_count - 1)
log.info(f'number of cores available: {core_count}')
log.info(f'N_JOBS: {N_JOBS}')


# ── R2 market-data layout ────────────────────────────────────────────────
# Single source of truth for the top-level data prefixes (relative to
# AWS_S3_FOLDER). The IM collector, the WEIS collector, and the app's read
# path (data_engineering) all build their base paths from these, so a layout
# change has exactly one home.
IM_PREFIX = "im/"
WEIS_PREFIX = "weis/"


def _s3_storage_options() -> dict:
    """Return Polars storage_options for S3/R2 endpoint, if configured."""
    endpoint = os.getenv("S3_ENDPOINT_URL")
    return {"endpoint_url": endpoint} if endpoint else {}


def check_file_exists_client(bucket_name: str, object_name: str) -> bool:
    """
    Checks if a file (object) exists in an S3 bucket using boto3 client.

    Args:
        bucket_name: The S3 bucket name.
        object_name: The object key to check.

    Returns:
        True if the object exists, False on a 404 (other errors re-raise).
    """
    s3_client = boto3.client('s3', endpoint_url=os.getenv("S3_ENDPOINT_URL"))
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_name)
        return True
    except ClientError as e:
        # If a ClientError is raised, check the error code.
        # A 404 error code indicates the object does not exist.
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Re-raise the exception if it's not a 404
            raise e


# subclass Parallel to get the progress bar to print
# https://github.com/joblib/joblib/issues/972
# https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm.tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)
    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def set_he(
        df: pl.DataFrame,
        time_cols: List[str] = ['Interval', 'GMTIntervalEnd', 'timestamp_mst'],
    ) -> pl.DataFrame:
    """
    Add hour ending columns for grouping 5 minute intervals.

    Args:
        df: Polars DataFrame with datetime columns.
        time_cols: List of column names to create hour ending versions of.

    Returns:
        pl.DataFrame with new *_HE columns added (ceiling to hour).
    """
    for time_col in time_cols:
        he_col = time_col+'_HE'
        log.debug(f'adding hour ending col: {he_col}')
        # df[he_col] = df[time_col].dt.ceil('h')
        df = df.with_columns(xdt.ceil(time_col, '1h').alias(he_col))

    return df


def add_timestamp_mst(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add MST timestamp column derived from GMT interval end time.

    Converts the GMTIntervalEnd column from UTC to Mountain Standard Time
    and adds it as a new 'timestamp_mst' column.

    Args:
        df: DataFrame with 'GMTIntervalEnd' datetime column.

    Returns:
        pl.DataFrame
    """
    df = df.with_columns(
        pl.col("GMTIntervalEnd")
        .dt.offset_by("-7h")
        .alias("timestamp_mst")
    )

    return df


def format_df_colnames(df: pl.DataFrame) -> None:
    """
    Format dataframe column names for database compatibility.

    Strips whitespace and replaces spaces with underscores.

    Args:
        df: Polars DataFrame to update column names.

    Returns:
        None - column names are modified in place.
    """
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]


def get_csv_from_url(
        url: str,
        timeout: int=120,
        connect_timeout: int=10,
) -> pl.DataFrame:
    """
    Read a CSV file from an SPP portal URL into a Polars DataFrame.

    Args:
        url: URL path to the CSV file.
        timeout: Read timeout in seconds (once connected).
        connect_timeout: Connection timeout in seconds. Kept short so an
            unreachable portal fails fast instead of blocking the full read
            timeout on every file — when the portal is down, a slow-connect
            per file otherwise multiplies across hundreds of files and blows
            the collection job's Modal timeout.

    Returns:
        pl.DataFrame created from reading in the csv from the url;
        if there is an error reading the url an empty dataframe
        is returned.
    """
    try:
        response = requests.get(url, timeout=(connect_timeout, timeout))
        if response.ok:
            # infer dtypes from the whole file, not the default 100-row
            # sample: price components (MCC, MLC, ...) can be integer-valued
            # for the first rows and float later, which mis-infers as i64.
            df = pl.read_csv(StringIO(response.text), infer_schema_length=None)
            log.debug(f'df.shape: {df.shape}')
        else:
            df = pl.DataFrame()
            log.error(f'ERROR READING URL: {url}')
            log.error(response.reason)
        # Be polite to the portal between requests we actually reached. Skip
        # this pause on a connection failure (below) so a portal outage fails
        # fast rather than adding 2s to every timed-out file.
        sleep(2)

    except Exception as e:
        # By this way we can know about the type of error occurring
        log.error(e)
        df = pl.DataFrame()

    return df

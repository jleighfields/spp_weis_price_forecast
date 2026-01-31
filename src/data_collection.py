"""
Data collection module for SPP WEIS price forecasting.

This module provides ETL functions for gathering data from SPP (Southwest Power Pool)
and storing it directly to S3 as parquet files. Data types collected include:

- MTLF: Mid-Term Load Forecast (hourly load forecasts and actuals)
- MTRF: Mid-Term Resource Forecast (wind and solar generation forecasts)
- LMP: Locational Marginal Prices (5-minute intervals aggregated to hourly)
- Generation Capacity: Hourly generation by fuel type

Key features:
    - Parallel data collection using joblib for efficiency
    - Individual parquet files written to S3 per time interval
    - Upsert operations to consolidate data into unified parquet files
    - Polars for high-performance data manipulation

Dependencies:
    - polars: Data manipulation and parquet I/O
    - pandas: Timestamp handling
    - boto3: S3 file operations
    - requests: HTTP requests to SPP portal
    - joblib: Parallel processing
"""
# pylint: disable=C0103,W1203,W1201

# base imports
import os
import sys
from time import sleep
from io import StringIO
from typing import List, Union, Callable
import tqdm

# logging
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# data
import requests
import pandas as pd
import polars as pl
import polars_xdt as xdt
import duckdb
# from meteostat import Hourly, Point

# parallel processing
from joblib import Parallel, delayed, cpu_count
core_count = cpu_count()
N_JOBS = max(1, core_count - 1)
log.info(f'number of cores available: {core_count}')
log.info(f'N_JOBS: {N_JOBS}')

# adding module folder to system path
# needed for running scripts as jobs
home = os.getenv('HOME')
module_paths = [
    f'{home}/spp_weis_price_forecast/src',
    f'{home}/Documents/github/spp_weis_price_forecast/src',
    '/cloud/project/src'
]
for module_path in module_paths:
    if os.path.isdir(module_path):
        log.info('adding module path')
        sys.path.insert(0, module_path)
        
import utils


# AWS S3 configuration - allows deployment to different environments
# by configuring bucket and folder prefix via environment variables
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER")

###########################################################
# HELPER FUNCTIONS
###########################################################

import boto3
from botocore.exceptions import ClientError

def check_file_exists_client(bucket_name, object_name):
    """
    Checks if a file (object) exists in an S3 bucket using boto3 client.
    """
    s3_client = boto3.client('s3')
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


def convert_datetime_cols(
        df: pl.DataFrame,
        dt_cols: List[str] = ['Interval', 'GMTIntervalEnd'],
) -> pl.DataFrame:
    """
    Convert string columns to datetime values.

    Args:
        df: Polars DataFrame with datetimes as strings.
        dt_cols: List of column names to convert.

    Returns:
        pl.DataFrame with converted timestamp columns.
    """

    for col in dt_cols:
        log.debug(f'converting {col} to datetime')
        df = df.with_columns(pl.col(col).str.to_datetime(format = '%m/%d/%Y %H:%M:%S'))
        # df[col] = pd.to_datetime(df[col], format = '%m/%d/%Y %H:%M:%S')
        
    return df


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


def get_time_components(
        time_str: str = None,
        five_min_ceil: bool = False,
) -> dict:
    """
    Function to get the formated time components needed to read csv file from SPP url.
    Args:
        time_str: str - string for date to be converted to timestamp i.e. 4/1/2023 07:00:00
            if None the current time will be used
        five_min_ceil: bool - if True time_stamp is rounded up to next 5 minute
            interval (5 minute ending interval), otherwise time_stamp is rounded
            up to the next hour (hour ending interval)
    Returns:
        dict - with time components formatted to insert into url for csv files
    """

    # dict to store time components
    tc = {}

    # get time stamp from time_str or current time if None
    # weis time are in central time
    log.debug(f'time_str: {time_str}')

    if time_str:
        time_stamp = pd.to_datetime(time_str)
    else:
        time_stamp = pd.Timestamp.now()

    log.debug(f'time_stamp: {time_stamp}')

    # ceiling time to 5minute or hour ending interval
    if five_min_ceil:
        time_stamp = time_stamp.ceil(freq='5min')
    else:
        time_stamp = time_stamp.ceil(freq='h')

    log.debug(f'time_stamp after ceil: {time_stamp}')

    try:
        time_stamp_ct = time_stamp.tz_localize("America/Chicago")
        log.debug(f'time_stamp localized: {time_stamp}')

        # get time components for formatting url path
        # times are in central time
        tc['YEAR'] = str(time_stamp_ct.year)
        tc['MONTH'] = str(time_stamp_ct.month).zfill(2)
        tc['DAY'] = str(time_stamp_ct.day).zfill(2)
        tc['HOUR'] = str(time_stamp_ct.hour).zfill(2)
        tc['MINUTE'] = str(time_stamp_ct.minute).zfill(2)
        tc['YM'] = tc['YEAR']+tc['MONTH']
        tc['YMD'] = tc['YM']+tc['DAY']
        tc['COMBINED'] = tc['YMD']+tc['HOUR']+tc['MINUTE']
        tc['timestamp'] = time_stamp_ct
        tc['timestamp_utc'] = time_stamp_ct.tz_convert(None)
        return tc

    except Exception:
        log.error(f'error parsing: {time_stamp}')
        return None
    

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

###########################################################
# GET FILE URLS
###########################################################
def get_csv_from_url(
        url: str,
        timeout: int=120,
) -> pl.DataFrame:
    """
    Function to read csv file from SPP url.
    Args:
        url: str - url path to csv
    Returns:
        pl.DataFrame created from reading in the csv from the url,
            if there is an error reading the url an empty dataframe
            is returned
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.ok:
            df = pl.read_csv(StringIO(response.text))
            log.debug(f'df.shape: {df.shape}')
        else:
            df = pl.DataFrame()
            log.error(f'ERROR READING URL: {url}')
            log.error(response.reason)

    except Exception as e:
        # By this way we can know about the type of error occurring
        log.error(e)
        df = pl.DataFrame()

    sleep(2)
    return df


def get_hourly_mtlf_url(tc: dict) -> str:
    """
    Function to create url for mid-term load forecast csv from time components.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        str - url used for reading in the csv

    """
    base_url = 'https://portal.spp.org/file-browser-api/download/systemwide-hourly-load-forecast-mtlf-vs-actual-weis?'
    # base_url = 'https://marketplace.spp.org/file-browser-api/download/systemwide-hourly-load-forecast-mtlf-vs-actual-weis?'
    path_url = f"path=%2F{tc['YEAR']}%2F{tc['MONTH']}%2F{tc['DAY']}%2FWEIS-OP-MTLF-{tc['COMBINED']}.csv"
    url = base_url + path_url
    return url


def get_hourly_mtrf_url(tc: dict) -> str:
    """
    Function to create url for mid-term resource forecast csv from time components.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        str - url used for reading in the csv
    """
    base_url = 'https://portal.spp.org/file-browser-api/download/mid-term-resource-forecast-mtrf-weis?'
    # base_url = 'https://marketplace.spp.org/file-browser-api/download/mid-term-resource-forecast-mtrf-weis?'
    path_url = f"path=%2F{tc['YEAR']}%2F{tc['MONTH']}%2F{tc['DAY']}%2FWEIS-OP-MTRF-{tc['COMBINED']}.csv"
    url = base_url + path_url
    return url


def get_5min_lmp_url(tc: dict) -> str:
    """
    Function to create url for 5 minute lmp csv from time components
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        str - url used for reading in the csv
    """

    base_url = 'https://portal.spp.org/file-browser-api/download/lmp-by-settlement-location-weis?'
    # base_url = 'https://marketplace.spp.org/file-browser-api/download/lmp-by-settlement-location-weis?'
    path_url = f"path=%2F{tc['YEAR']}%2F{tc['MONTH']}%2FBy_Interval%2F{tc['DAY']}%2FWEIS-RTBM-LMP-SL-{tc['COMBINED']}.csv"
    url = base_url + path_url
    return url

# https://portal.spp.org/file-browser-api/download/lmp-by-settlement-location-weis?path=%2F2025%2F01%2FBy_Interval%2F16%2FWEIS-RTBM-LMP-SL-202501161925.csv
# https://portal.spp.org/file-browser-api/download/lmp-by-settlement-location-weis?path=%2F2025%2F01%2FBy_Interval%2F16%2FWEIS-RTBM-LMP-SL-202501161925.csv

def get_daily_lmp_url(tc: dict) -> str:
    """
    Function to create url for daily 5 minute lmp csv from time components.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        str - url used for reading in the csv
    """

    base_url = 'https://portal.spp.org/file-browser-api/download/lmp-by-settlement-location-weis?'
    # base_url = 'https://marketplace.spp.org/file-browser-api/download/lmp-by-settlement-location-weis?'
    path_url = f"path=%2F{tc['YEAR']}%2F{tc['MONTH']}%2FBy_Day%2FWEIS-RTBM-LMP-DAILY-SL-{tc['YMD']}.csv"
    url = base_url + path_url
    return url


def get_gen_cap_url(tc: dict) -> str:
    """
    Function to create url for daily 5 minute lmp csv from time components.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        str - url used for reading in the csv
    """
    # https://portal.spp.org/file-browser-api/download/hourly-gen-capacity-by-fuel-type-weis?path=%2F2024%2F11%2FWEIS-HRLY-GEN-CAP-BY-FUEL-TYPE-20241108.csv
    base_url = 'https://portal.spp.org/file-browser-api/download/hourly-gen-capacity-by-fuel-type-weis?'
    # base_url = 'https://marketplace.spp.org/file-browser-api/download/lmp-by-settlement-location-weis?'
    path_url = f"path=%2F{tc['YEAR']}%2F{tc['MONTH']}%2FWEIS-HRLY-GEN-CAP-BY-FUEL-TYPE-{tc['YMD']}.csv"
    url = base_url + path_url
    return url


###########################################################
# PROCESS AND COLLECT DATA
###########################################################
# get_range_data() is the base function that gathers
# data from a range of datetimes and calls a
# function (get_process_func) to process the data
# each type of data needs different processing so
# there is a get_process_* function for
# MTLF, MTRF, single 5 min LMP intervals, and daily
# 5 minute LMP files.  A helper function for each
# type of data (get_range_data_*) is defined to tie
# these functions together using data specific parameters
###########################################################

def get_range_data(
        end_ts: pd.Timestamp,
        n_periods: int,
        freq: str,
        get_process_func: Callable,
        do_parallel: bool=True,
    ) -> List[str]:
    """
    Collect data for a range of time periods and write to S3.

    Downloads data from SPP for each time period, processes it, and writes
    individual parquet files to S3. Returns list of S3 paths or URLs for
    failed downloads.

    Args:
        end_ts: The last time period to get data (e.g., pd.Timestamp('6/4/2023')).
        n_periods: Number of time periods to gather prior to end_ts.
        freq: Frequency - 'D' for daily, 'h' for hourly, '5min' for 5 minute.
        get_process_func: Processing function for the data type:
            - MTLF: get_process_mtlf
            - MTRF: get_process_mtrf
            - 5min LMP intervals: get_process_5min_lmp
            - daily LMP files: get_process_daily_lmp
        do_parallel: If True, use parallel processing with joblib.

    Returns:
        List of S3 paths for successful writes, or URLs for failed downloads.
    """

    # boolean to determine if we need 5 minute intervals
    five_min_ceil = freq == '5min'

    # get list of time strings to process for file urls
    time_str_list = [str(dt) for dt in pd.date_range(end=end_ts, periods=n_periods, freq=freq)]

    # get list of time components for urls from time str list
    tc_list = [get_time_components(time_str, five_min_ceil=five_min_ceil) for time_str in time_str_list]
    # tc can be None if error during daylight savings
    tc_list = [tc for tc in tc_list if tc is not None]
    N = len(tc_list)

    # get list of dataframes from time components
    if do_parallel:
        df_list = (
            ProgressParallel(n_jobs=N_JOBS, total=N)
            (delayed(get_process_func)(tc) for tc in tc_list)
        )

    else:
        df_list = []
        for tc in tqdm.tqdm(tc_list):
            df_list+=[get_process_func(tc)]

    return df_list


def get_process_mtlf(tc: dict) -> str:
    """
    Download, process, and write MTLF data to S3.

    Fetches mid-term load forecast CSV from SPP, converts to Polars DataFrame,
    adds timestamps, and writes to S3 as parquet.

    Args:
        tc: Time components dictionary from get_time_components().

    Returns:
        S3 path if successful, or the source URL if download failed.
    """
    data_category = 'mtlf'
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER')
    assert AWS_S3_BUCKET
    assert AWS_S3_FOLDER
    
    url = get_hourly_mtlf_url(tc)
    log.debug(f'{data_category} url: {url}')

    df = get_csv_from_url(url)

    if df.shape[0] > 0:
        parquet_filename = url.split('WEIS-')[-1].replace('.csv','.parquet')
        s3_path = f's3://{AWS_S3_BUCKET}/{AWS_S3_FOLDER}data/{data_category}/{parquet_filename}'
        format_df_colnames(df)
        df = convert_datetime_cols(df)
        df = add_timestamp_mst(df)
        df = df.with_columns(
            pl.col.MTLF.cast(pl.Float32),
            pl.col.Averaged_Actual.cast(pl.Float32),
            pl.lit(tc['timestamp_utc']).alias('file_create_time_utc'),
            pl.lit(url).alias('url'),
        )
        df.unique().write_parquet(s3_path)
        return s3_path
    else:
        return url


def get_range_data_mtlf(
        end_ts: pd.Timestamp,
        n_periods: int,
    ) -> List[str]:
    """
    Collect MTLF data for a range of hourly time periods.

    Args:
        end_ts: The last hour to get data.
        n_periods: Number of hours to gather prior to end_ts.

    Returns:
        List of S3 paths for successful writes, or URLs for failed downloads.
    """
    freq = 'h'
    get_process_func = get_process_mtlf
    # dup_cols = ['GMTIntervalEnd']
    return get_range_data(end_ts, n_periods, freq, get_process_func)


def get_process_mtrf(tc: dict) -> str:
    """
    Download, process, and write MTRF data to S3.

    Fetches mid-term resource forecast (wind/solar) CSV from SPP, converts to
    Polars DataFrame, adds timestamps, and writes to S3 as parquet.

    Args:
        tc: Time components dictionary from get_time_components().

    Returns:
        S3 path if successful, or the source URL if download failed.
    """
    data_category = 'mtrf'
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER')
    assert AWS_S3_BUCKET
    assert AWS_S3_FOLDER
    
    url = get_hourly_mtrf_url(tc)
    log.debug(f'{data_category} url: {url}')

    df = get_csv_from_url(url)

    if df.shape[0] > 0:
        parquet_filename = url.split('WEIS-')[-1].replace('.csv','.parquet')
        s3_path = f's3://{AWS_S3_BUCKET}/{AWS_S3_FOLDER}data/{data_category}/{parquet_filename}'
        format_df_colnames(df)
        df = convert_datetime_cols(df)
        df = add_timestamp_mst(df)
        df = df.with_columns(
            pl.col.Wind_Forecast_MW.cast(pl.Float32),
            pl.col.Solar_Forecast_MW.cast(pl.Float32),
            pl.lit(tc['timestamp_utc']).alias('file_create_time_utc'),
            pl.lit(url).alias('url'),
        )
        df.unique().write_parquet(s3_path)
        return s3_path
    else:
        return url



def get_range_data_mtrf(
        end_ts: pd.Timestamp,
        n_periods: int,
    ) -> List[str]:
    """
    Collect MTRF data for a range of hourly time periods.

    Args:
        end_ts: The last hour to get data.
        n_periods: Number of hours to gather prior to end_ts.

    Returns:
        List of S3 paths for successful writes, or URLs for failed downloads.
    """
    freq = 'h'
    get_process_func = get_process_mtrf
    # dup_cols = ['GMTIntervalEnd']
    return get_range_data(end_ts, n_periods, freq, get_process_func)


def agg_lmp(five_min_lmp_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate 5-minute LMPs to hour ending averages.

    Args:
        five_min_lmp_df: Polars DataFrame with 5-minute LMP data.

    Returns:
        Polars DataFrame with hourly averaged LMP values.
    """
    group_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name'
    ]
    value_cols = ['LMP', 'MLC', 'MCC', 'MEC']
    he_lmp_df = (
        five_min_lmp_df.select(group_cols + value_cols)
        .group_by(group_cols)
        .mean()
    )
    return he_lmp_df


def get_process_5min_lmp(tc: dict) -> str:
    """
    Download, process, and write 5-minute LMP interval data to S3.

    Fetches single 5-minute LMP CSV from SPP, aggregates to hourly, and writes
    to S3 as parquet. Note: 5-minute files have different column names than
    daily files.

    Args:
        tc: Time components dictionary from get_time_components().

    Returns:
        S3 path if successful, or the source URL if download failed.
    """
    data_category = 'lmp_5min'
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER')
    assert AWS_S3_BUCKET
    assert AWS_S3_FOLDER

    url = get_5min_lmp_url(tc)
    log.debug(f'{data_category} url: {url}')

    df = get_csv_from_url(url)

    if df.shape[0] > 0:
        parquet_filename = url.split('WEIS-')[-1].replace('.csv','.parquet')
        s3_path = f's3://{AWS_S3_BUCKET}/{AWS_S3_FOLDER}data/{data_category}/{parquet_filename}'
        format_df_colnames(df)
        log.debug(f'{df.columns = }')
        # df.columns = ['Interval', 'GMTIntervalEnd', 'Settlement_Location', 'Pnode', 'LMP', 'MLC', 'MCC', 'MEC']
        df = df.rename(
            {
                'Settlement_Location':'Settlement_Location_Name',
                'Pnode':'PNODE_Name',
            },
        )
        df = convert_datetime_cols(df)
        df = add_timestamp_mst(df)
        df = set_he(df)
        df = agg_lmp(df)
        
        df = df.with_columns(
            pl.col.LMP.cast(pl.Float32),
            pl.col.MLC.cast(pl.Float32),
            pl.col.MCC.cast(pl.Float32),
            pl.col.MEC.cast(pl.Float32),
            pl.lit(tc['timestamp_utc']).alias('file_create_time_utc'),
            pl.lit(url).alias('url'),
        )
        df.unique().write_parquet(s3_path)
        return s3_path
    else:
        return url


#TODO: combine daily and interval lmp get functions
def get_range_data_interval_5min_lmps(
        end_ts: pd.Timestamp,
        n_periods: int,
    ) -> List[str]:
    """
    Collect 5-minute LMP interval data for a range of time periods.

    Args:
        end_ts: The last 5-minute interval to get data.
        n_periods: Number of 5-minute intervals to gather prior to end_ts.

    Returns:
        List of S3 paths for successful writes, or URLs for failed downloads.
    """
    freq = '5min'
    get_process_func = get_process_5min_lmp
    # dup_cols = ['GMTIntervalEnd_HE', 'Settlement_Location_Name', 'PNODE_Name']
    return get_range_data(end_ts, n_periods, freq, get_process_func)


def get_process_daily_lmp(tc: dict) -> str:
    """
    Download, process, and write daily LMP file to S3.

    Fetches daily 5-minute LMP CSV from SPP (contains full day of intervals),
    aggregates to hourly, and writes to S3 as parquet. Note: daily files have
    different column names than individual 5-minute interval files.

    Args:
        tc: Time components dictionary from get_time_components().

    Returns:
        S3 path if successful, or the source URL if download failed.
    """
    data_category = 'lmp_daily'
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER')
    assert AWS_S3_BUCKET
    assert AWS_S3_FOLDER

    url = get_daily_lmp_url(tc)
    log.debug(f'{data_category} url: {url}')

    df = get_csv_from_url(url)

    if df.shape[0] > 0:
        parquet_filename = url.split('WEIS-')[-1].replace('.csv','.parquet')
        s3_path = f's3://{AWS_S3_BUCKET}/{AWS_S3_FOLDER}data/{data_category}/{parquet_filename}'
        format_df_colnames(df)
        log.debug(f'{df.columns = }')
        # df.columns = ['Interval', 'GMT_Interval', 'Settlement_Location_Name', 'PNODE_Name', 'LMP', 'MLC', 'MCC', 'MEC']
        df = df.rename({'GMT_Interval':'GMTIntervalEnd'})
        df = convert_datetime_cols(df)
        df = add_timestamp_mst(df)
        df = set_he(df)
        df = agg_lmp(df)
        
        df = df.with_columns(
            pl.col.LMP.cast(pl.Float32),
            pl.col.MLC.cast(pl.Float32),
            pl.col.MCC.cast(pl.Float32),
            pl.col.MEC.cast(pl.Float32),
            pl.lit(tc['timestamp_utc']).alias('file_create_time_utc'),
            pl.lit(url).alias('url'),
        )
        df.unique().write_parquet(s3_path)
        return s3_path
    else:
        return url

#TODO: combine daily and interval lmp get functions
def get_range_data_interval_daily_lmps(
        end_ts: pd.Timestamp,
        n_periods: int,
    ) -> List[str]:
    """
    Collect daily LMP files for a range of days.

    Args:
        end_ts: The last day to get data.
        n_periods: Number of days to gather prior to end_ts.

    Returns:
        List of S3 paths for successful writes, or URLs for failed downloads.
    """
    freq = 'D'
    get_process_func = get_process_daily_lmp
    # dup_cols = ['GMTIntervalEnd_HE', 'Settlement_Location_Name', 'PNODE_Name']
    return get_range_data(end_ts, n_periods, freq, get_process_func)


###########################################################
# UPSERT DATA
###########################################################

def upsert_mtlf_mtrf_lmp(
    parquet_files: List[str],
    target: str,
) -> None:
    """
    Upsert data from individual parquet files into a consolidated target file.

    Reads multiple parquet files, deduplicates by primary key (keeping the latest
    by file_create_time_utc), merges with existing target file if present, and
    writes the consolidated result back to S3.

    Args:
        parquet_files: List of S3 paths to individual parquet files to upsert.
        target: Target data type - 'mtlf', 'mtrf', or 'lmp'. Determines the
            primary key columns used for deduplication:
            - mtlf/mtrf: GMTIntervalEnd
            - lmp: GMTIntervalEnd_HE, Settlement_Location_Name, PNODE_Name

    Returns:
        None - writes consolidated parquet to S3 at data/{target}.parquet.

    Environment Variables:
        AWS_S3_BUCKET: S3 bucket for data storage.
        AWS_S3_FOLDER: Folder prefix within the bucket.
    """

    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER')
    assert AWS_S3_BUCKET
    assert AWS_S3_FOLDER

    if target == 'lmp':
        key_cols = [
            'GMTIntervalEnd_HE',
            'Settlement_Location_Name', 
            'PNODE_Name',
        ]
    elif target in ['mtrf', 'mtlf']:
        key_cols = ['GMTIntervalEnd']
    else:
        raise ValueError(f"{target = } - expected one of (mtlf, mtrf, lmp)")


    object_name = f'{AWS_S3_FOLDER}data/{target}.parquet'
    s3_path_target = f's3://{AWS_S3_BUCKET}/{object_name}'
    file_exists = check_file_exists_client(AWS_S3_BUCKET, object_name)
    log.info(f'{s3_path_target = }')
    log.info(f'{file_exists = }')
    log.info(f'number of files upserting: {len(parquet_files)}')

    upsert_df = (
        pl.scan_parquet(parquet_files)
        .sort(key_cols + ['file_create_time_utc'], descending=False)
        .unique(
            subset=key_cols, 
            keep='last',
            maintain_order=True,
        )
        .collect()
    )

    num_dups = upsert_df.select(key_cols).is_duplicated().sum()
    assert num_dups == 0, print(f'{num_dups = }')
    log.info(f'{upsert_df.shape = }')


    update_count = upsert_df.shape[0]

    if 'GMTIntervalEnd' in upsert_df:
        min_max = (
            upsert_df
            .select(
                pl.col.GMTIntervalEnd.min().alias('min_date'),
                pl.col.GMTIntervalEnd.max().alias('max_date'),
            )
        )
    else:
        min_max = (
            upsert_df
            .select(
                pl.col.GMTIntervalEnd_HE.min().alias('min_date'),
                pl.col.GMTIntervalEnd_HE.max().alias('max_date'),
            )
        )

    log.info(f'min/max update times: \n{min_max}')

    if file_exists:
        target_df = pl.read_parquet(s3_path_target)
        log.info(f'{target_df.shape = }')
        start_count = target_df.shape[0]
        log.info(f'starting count: {start_count:,}')

        target_df = (
            pl.concat([target_df, upsert_df])
            .sort(key_cols + ['file_create_time_utc'], descending=False)
            .unique(
                subset=key_cols, 
                keep='last',
                maintain_order=True,
            )
        )
    else:
        start_count = 0
        target_df = upsert_df

    num_dups = target_df.select(key_cols).is_duplicated().sum()
    assert num_dups == 0, print(f'{num_dups = }')

    log.info(f'starting count: {start_count:,}')
    target_df.write_parquet(s3_path_target)

    end_count = target_df.shape[0]
    insert_count = end_count - start_count
    rows_updated = update_count - insert_count
    log.info(
        f'ROWS INSERTED: {insert_count:,} - ROWS UPDATED: {rows_updated :,} - TOTAL: {end_count:,}'
        )


def rebuild_mtlf_mtrf_lmp_from_s3(src_dir: str):
    
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER')
    assert AWS_S3_BUCKET
    assert AWS_S3_FOLDER
    
    if src_dir in ['lmp_daily', 'lmp_5min']:
        key_cols = [
            'GMTIntervalEnd_HE',
            'Settlement_Location_Name', 
            'PNODE_Name',
        ]
    elif src_dir in ['mtrf', 'mtlf']:
        key_cols = ['GMTIntervalEnd']
    else:
        raise ValueError(f"{src_dir = } - expected one of (mtlf, mtrf, lmp_daily, lmp_5min)")
    
    if 'lmp_' in src_dir:
        object_name = f'{AWS_S3_FOLDER}data/lmp.parquet'
    else: # mtrf, mtlf
        object_name = f'{AWS_S3_FOLDER}data/{src_dir}.parquet'

    s3_path_target = f's3://{AWS_S3_BUCKET}/{object_name}'
    file_exists = check_file_exists_client(AWS_S3_BUCKET, object_name)
    log.info(f'{s3_path_target = }')
    log.info(f'{file_exists = }')
    
    upsert_df = (
        pl.scan_parquet(f's3://{AWS_S3_BUCKET}/{AWS_S3_FOLDER}data/{src_dir}/*.parquet')
        .sort(key_cols + ['file_create_time_utc'], descending=False)
        .unique(
            subset=key_cols, 
            keep='last',
            maintain_order=True,
        )
        .collect()
    )
    
    num_dups = upsert_df.select(key_cols).is_duplicated().sum()
    assert num_dups == 0, print(f'{num_dups = }')
    log.info(f'{upsert_df.shape = }')
    
    update_count = upsert_df.shape[0]
    
    if 'GMTIntervalEnd' in upsert_df:
        min_max = (
            upsert_df
            .select(
                pl.col.GMTIntervalEnd.min().alias('min_date'),
                pl.col.GMTIntervalEnd.max().alias('max_date'),
            )
        )
    else:
        min_max = (
            upsert_df
            .select(
                pl.col.GMTIntervalEnd_HE.min().alias('min_date'),
                pl.col.GMTIntervalEnd_HE.max().alias('max_date'),
            )
        )
    
    log.info(f'min/max update times: \n{min_max}')
    
    if file_exists:
        target_df = pl.read_parquet(s3_path_target)
        start_count = target_df.shape[0]
    
        target_df = (
            pl.concat([target_df, upsert_df])
            .sort(key_cols + ['file_create_time_utc'], descending=False)
            .unique(
                subset=key_cols, 
                keep='last',
                maintain_order=True,
            )
        )
    else:
        start_count = 0
        target_df = upsert_df
    
    num_dups = target_df.select(key_cols).is_duplicated().sum()
    assert num_dups == 0, print(f'{num_dups = }')
    
    log.info(f'starting count: {start_count:,}')
    target_df.write_parquet(s3_path_target)
    
    end_count = target_df.shape[0]
    insert_count = end_count - start_count
    rows_updated = update_count - insert_count
    log.info(
        f'ROWS INSERTED: {insert_count:,} - ROWS UPDATED: {rows_updated :,} - TOTAL: {end_count:,}'
        )
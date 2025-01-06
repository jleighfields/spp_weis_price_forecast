'''
Functions for ETL for gathering data from SPP
and loading to databricks
'''
# pylint: disable=C0103,W1203,W1201

# base imports
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
import duckdb

# parallel processing
from joblib import Parallel, delayed, cpu_count
core_count = cpu_count()
N_JOBS = max(1, core_count - 1)
log.info(f'number of cores available: {core_count}')
log.info(f'N_JOBS: {N_JOBS}')

###########################################################
# HELPER FUNCTIONS
###########################################################

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
        df: pd.DataFrame,
        dt_cols: List[str] = ['Interval', 'GMTIntervalEnd'],
) -> None:
    """
    Function to convert string columns to datetime values.
    Args:
        df: pd.DataFrame - dataframe with datetimes as strings
        dt_cols: List[str] - list of column names to convert
    Returns:
        None - columns are updated in place
    """

    for col in dt_cols:
        log.debug(f'converting {col} to datetime')
        df[col] = pd.to_datetime(df[col], format = '%m/%d/%Y %H:%M:%S')


def set_he(
        df: pd.DataFrame,
        time_cols: List[str] = ['Interval', 'GMTIntervalEnd', 'timestamp_mst'],
    ) -> None:
    """
    Function to add hour ending column for grouping 5 minute intervals.
    Args:
        df: pd.DataFrame - dataframe with datetime to
            get hour ending intervals
        dt_cols: List[str] - list of column names to convert
    Returns:
        None - columns are updated in place
    """
    for time_col in time_cols:
        he_col = time_col+'_HE'
        log.debug(f'adding hour ending col: {he_col}')
        df[he_col] = df[time_col].dt.ceil('h')


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
        return tc

    except:
        log.error(f'error parsing: {time_stamp}')
        return None
    
def add_timestamp_mst(df: pd.DataFrame):
    df['timestamp_mst'] = (
        df.GMTIntervalEnd.dt.tz_localize("UTC")
        .dt.tz_convert('MST')
        .dt.tz_localize(None)
    )


def format_df_colnames(df: pd.DataFrame) -> None:
    """
    Function to format dataframe column names.  Used after reading
    csv data from url to prepare for loading into a database.
    Args:
        df: pd.DataFrame - dataframe to update column names
            to a database compliant format
    Returns:
        None - column names are modified in place
    """

    df.columns = [col.strip().replace(' ', '_') for col in df.columns]


###########################################################
# GET FILE URLS
###########################################################
def get_csv_from_url(
        url: str,
        timeout: int=60,
) -> pd.DataFrame:
    """
    Function to read csv file from SPP url.
    Args:
        url: str - url path to csv
    Returns:
        pd.DataFrame created from reading in the csv from the url,
            if there is an error reading the url an empty dataframe
            is returned
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.ok:
            df = pd.read_csv(StringIO(response.text))
            log.debug(f'df.shape: {df.shape}')
        else:
            df = pd.DataFrame()
            log.error(f'ERROR READING URL: {url}')
            log.error(response.reason)

    except Exception as e:
        # By this way we can know about the type of error occurring
        log.error(e)
        df = pd.DataFrame()
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
        dup_cols: List[str],
        do_parallel: bool=True,
    ) -> pd.DataFrame:
    """
    Function to get a range of data based on an ending timestamp (end_ts)
    and the number of preceeding periods (n_periods).
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
        freq: str - either 'D' for daily, 'h' for hourly datam '5min' for 5 minute data
            lmp single intervals are '5min', lmp daily file are 'D' freq
            load and resource forecasts are 'h'
        get_process_func: Callable - function to process data, each type of
            data has a function to process the downloaded data
            MTLF: get_process_mtlf
            MTRF: get_process_mtrf
            single LMP intervals: get_process_5min_lmp
            daily LMP intervals: get_process_daily_lmp
            NOTE: there two processing functions for LMPs since they are
                formatted differently.
        dup_cols: List[str] - list of columns that create a unique id, this
            is used to dedup the data
    Returns:
        pd.DataFrame containing all the processed data for the range of datetimes
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
        for tc in tc_list:
            df_list+=[get_process_func(tc)]
            # time.sleep(0.5) # be kind to the server, maybe

    df = pd.concat(df_list)

    if df.shape[0] > 0:
        # using keep='last'
        # assumes the last items are the most recent
        # this will be true given the use of pd.date_range
        dups = df[dup_cols].duplicated(keep='last')
        df = df[~dups]
        df = df.sort_values(dup_cols).reset_index(drop=True)

    return df


def get_process_mtlf(tc: dict) -> pd.DataFrame:
    """
    Function to get and process MTLF data.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    mtlf_url = get_hourly_mtlf_url(tc)
    log.debug(f'mtlf_url: {mtlf_url}')

    df = get_csv_from_url(mtlf_url)

    if df.shape[0] > 0:
        format_df_colnames(df)
        convert_datetime_cols(df)
        add_timestamp_mst(df)

    return df


def get_range_data_mtlf(
        end_ts: pd.Timestamp,
        n_periods: int,
    ) -> pd.DataFrame:
    """
    Helper function to get and process MTLF data for a range of datetimes.
    Default args are set for MTLF and passed to get_range_data()
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    freq = 'h'
    get_process_func = get_process_mtlf
    dup_cols = ['GMTIntervalEnd']
    return get_range_data(end_ts, n_periods, freq, get_process_func, dup_cols)


def get_process_mtrf(tc: dict) -> pd.DataFrame:
    """
    Function to get and process MTRF data.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    mtrf_url = get_hourly_mtrf_url(tc)
    log.debug(f'mtrf_url: {mtrf_url}')

    df = get_csv_from_url(mtrf_url)

    if df.shape[0] > 0:
        format_df_colnames(df)
        convert_datetime_cols(df)
        add_timestamp_mst(df)

    return df


def get_range_data_mtrf(
        end_ts: pd.Timestamp,
        n_periods: int,
    ):
    """
    Helper function to get and process MTRF data for a range of datetimes.
    Default args are set for MTLF and passed to get_range_data()
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    freq = 'h'
    get_process_func = get_process_mtrf
    dup_cols = ['GMTIntervalEnd']
    return get_range_data(end_ts, n_periods, freq, get_process_func, dup_cols)


def agg_lmp(five_min_lmp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to aggregate 5 minute LMPs to hour ending LMPs.
    Args:
        five_min_lmp_df: pd.DataFrame - 5 minute data aggregate
    Returns:
        he_lmp_df: pd.DataFrame hour ending aggregates
    """
    group_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name'
    ]
    value_cols = ['LMP', 'MLC', 'MCC', 'MEC']
    he_lmp_df = (
        five_min_lmp_df[group_cols + value_cols]
        .groupby(group_cols)
        .mean()
        .reset_index()
    )
    return he_lmp_df


def get_process_5min_lmp(tc: dict) -> pd.DataFrame:
    """
    Function to get and process single 5 minute LMP data files.  Daily and
    5 minute files have different column names and need different processing.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    lmp_url = get_5min_lmp_url(tc)
    log.debug(f'lmp_url: {lmp_url}')

    df = get_csv_from_url(lmp_url)

    if df.shape[0] > 0:
        format_df_colnames(df)
        df.rename(
            columns={
                'GMT_Interval':'GMTIntervalEnd',
                'Settlement_Location':'Settlement_Location_Name',
                'Pnode':'PNODE_Name',
            },
            inplace=True
        )
        convert_datetime_cols(df)
        add_timestamp_mst(df)
        set_he(df)
        df = agg_lmp(df)

    return df

#TODO: combine daily and interval lmp get functions
def get_range_data_interval_5min_lmps(
        end_ts: pd.Timestamp,
        n_periods: int,
    ):
    """
    Helper function to get and process single LMP intervals data for a range of datetimes.
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    freq = '5min'
    get_process_func = get_process_5min_lmp
    dup_cols = ['GMTIntervalEnd_HE', 'Settlement_Location_Name', 'PNODE_Name']
    return get_range_data(end_ts, n_periods, freq, get_process_func, dup_cols)


def get_process_daily_lmp(tc) -> pd.DataFrame:
    """
    Function to get and process daily 5 minute LMP data. Daily and
    5 minute files have different column names and need different processing.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    lmp_url = get_daily_lmp_url(tc)
    log.debug(f'lmp_url: {lmp_url}')

    df = get_csv_from_url(lmp_url)

    if df.shape[0] > 0:
        format_df_colnames(df)
        df.rename(columns={'GMT_Interval':'GMTIntervalEnd'}, inplace=True)
        convert_datetime_cols(df)
        add_timestamp_mst(df)
        set_he(df)
        df = agg_lmp(df)

    return df

#TODO: combine daily and interval lmp get functions
def get_range_data_interval_daily_lmps(
        end_ts: pd.Timestamp,
        n_periods: int,
    ):
    """
    Helper function to get and process daily LMP intervals files.
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    freq = 'D'
    get_process_func = get_process_daily_lmp
    dup_cols = ['GMTIntervalEnd_HE', 'Settlement_Location_Name', 'PNODE_Name']
    return get_range_data(end_ts, n_periods, freq, get_process_func, dup_cols)


def get_process_gen_cap(tc: dict) -> pd.DataFrame:
    """
    Function to get and process MTLF data.
    Args:
        tc: dict - dictionary returned from get_time_components()
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    gen_cap_url = get_gen_cap_url(tc)
    log.debug(f'gen_cap_url: {gen_cap_url}')

    df = get_csv_from_url(gen_cap_url)

    if df.shape[0] > 0:
        format_df_colnames(df)
        df['GMT_TIME'] = pd.to_datetime(df['GMT_TIME'], format='ISO8601')
        df.rename(columns={'GMT_TIME': 'GMTIntervalEnd'}, inplace=True)
        df['timestamp_mst'] = (
            df.GMTIntervalEnd
            .dt.tz_convert('MST')
            .dt.tz_localize(None)
        )

    return df
    

def get_range_data_gen_cap(
        end_ts: pd.Timestamp,
        n_periods: int,
    ):
    """
    Helper function to get and process MTRF data for a range of datetimes.
    Default args are set for MTLF and passed to get_range_data()
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
    Returns:
        pd.DataFrame with processed data for file corresponding to
            url created from tc
    """
    freq = 'D'
    get_process_func = get_process_gen_cap
    dup_cols = ['GMTIntervalEnd']
    return get_range_data(end_ts, n_periods, freq, get_process_func, dup_cols)


###########################################################
# UPSERT DATA
###########################################################

def upsert_mtlf(
        mtlf_upsert: pd.DataFrame,
        backfill: bool=False,
) -> None:
    """
    Function to upsert new/backfilled MTLF data into duckdb database.
    Args:
        mtlf_upsert: pd.DataFrame - dataframe to upsert to MTLF table in database.
        backfill: bool = False - if true removes rows with missing values before
            upsert.  This removes rows where average actual is missing because
            the time period is forecasted and prevents overwriting actual values
            with forecasted values.
    Returns:
        None - new data is upserted to table
    """

    # remove missing values if backfilling
    if backfill:
        mtlf_upsert.dropna(axis=0, how='any', inplace=True)
    # remove any duplicated primary keys
    mtlf_upsert = mtlf_upsert[~mtlf_upsert.GMTIntervalEnd.duplicated()]
    update_count = len(mtlf_upsert)
    # NOTE: the df col order must match the order in the table
    ordered_cols = [
        'Interval', 'GMTIntervalEnd', 'timestamp_mst',
        'MTLF', 'Averaged_Actual',
    ]
    mtlf_upsert = mtlf_upsert[ordered_cols]
    log.info(f'mtlf_upsert.timestamp_mst.min(): {mtlf_upsert.timestamp_mst.min()}')
    log.info(f'mtlf_upsert.timestamp_mst.max(): {mtlf_upsert.timestamp_mst.max()}')

    # upsert with duckdb
    with duckdb.connect('~/spp_weis_price_forecast/data/spp.ddb') as con_ddb:
        create_mtlf = '''
        CREATE TABLE IF NOT EXISTS mtlf (
             Interval TIMESTAMP,
             GMTIntervalEnd TIMESTAMP PRIMARY KEY,
             timestamp_mst TIMESTAMP,
             MTLF INTEGER, 
             Averaged_Actual DOUBLE
             );
        '''
        con_ddb.sql(create_mtlf)

        res = con_ddb.sql('select count(*) from mtlf')
        start_count = res.fetchall()[0][0]
        log.info(f'starting count: {start_count:,}')

        mtlf_insert_update = '''
        INSERT INTO mtlf
            SELECT * FROM mtlf_upsert
            ON CONFLICT DO UPDATE SET MTLF = EXCLUDED.MTLF, 
            Averaged_Actual = EXCLUDED.Averaged_Actual;
        '''

        con_ddb.sql(mtlf_insert_update)

        res = con_ddb.sql('select count(*) from mtlf')
        end_count = res.fetchall()[0][0]
        insert_count = end_count - start_count
        rows_updated = update_count - insert_count
        log.info(
            f'ROWS INSERTED: {insert_count:,} ROWS UPDATED: {rows_updated :,} TOTAL: {end_count:,}')

        # copy to s3
        con_ddb.sql("INSTALL httpfs;")
        con_ddb.sql("LOAD httpfs;")
        con_ddb.sql("COPY mtlf TO 's3://spp-weis/data/mtlf.parquet';")


def upsert_mtrf(
        mtrf_upsert: pd.DataFrame,
        backfill: bool=False,
) -> None:
    """
    Function to upsert new/backfilled MTRF data into duckdb database.
    Args:
        mtrf_upsert: pd.DataFrame - dataframe to upsert to MTRF table in database.
        backfill: bool = False - if true removes rows with missing values before
            upsert.  This removes rows where average actual is missing because
            the time period is forecasted and prevents overwriting actual values
            with forecasted values.
    Returns:
        None - new data is upserted to table
    """
    # remove missing values if backfilling
    if backfill:
        mtrf_upsert.dropna(axis=0, how='any', inplace=True)
    # remove any duplicated primary keys
    mtrf_upsert = mtrf_upsert[~mtrf_upsert.GMTIntervalEnd.duplicated()]
    update_count = len(mtrf_upsert)
    # NOTE: the df col order must match the order in the table
    ordered_cols = [
        'Interval', 'GMTIntervalEnd', 'timestamp_mst',
        'Wind_Forecast_MW', 'Solar_Forecast_MW',
    ]
    mtrf_upsert = mtrf_upsert[ordered_cols]
    log.info(f'mtrf_upsert.timestamp_mst.min(): {mtrf_upsert.timestamp_mst.min()}')
    log.info(f'mtrf_upsert.timestamp_mst.max(): {mtrf_upsert.timestamp_mst.max()}')

    # upsert with duckdb
    with duckdb.connect('~/spp_weis_price_forecast/data/spp.ddb') as con_ddb:
        create_mtrf = '''
        CREATE TABLE IF NOT EXISTS mtrf (
             Interval TIMESTAMP,
             GMTIntervalEnd TIMESTAMP PRIMARY KEY,
             timestamp_mst TIMESTAMP,
             Wind_Forecast_MW DOUBLE, 
             Solar_Forecast_MW DOUBLE
             );
        '''
        con_ddb.sql(create_mtrf)

        res = con_ddb.sql('select count(*) from mtrf')
        start_count = res.fetchall()[0][0]
        log.info(f'starting count: {start_count:,}')

        mtrf_insert_update = '''
        INSERT INTO mtrf
            SELECT * FROM mtrf_upsert
            ON CONFLICT DO UPDATE SET Wind_Forecast_MW = EXCLUDED.Wind_Forecast_MW, 
            Solar_Forecast_MW = EXCLUDED.Solar_Forecast_MW;
        '''

        con_ddb.sql(mtrf_insert_update)

        res = con_ddb.sql('select count(*) from mtrf')
        end_count = res.fetchall()[0][0]
        insert_count = end_count - start_count
        rows_updated = update_count - insert_count
        log.info(
            f'ROWS INSERTED: {insert_count:,} ROWS UPDATED: {rows_updated :,} TOTAL: {end_count:,}')

        # copy to s3
        con_ddb.sql("INSTALL httpfs;")
        con_ddb.sql("LOAD httpfs;")
        con_ddb.sql("COPY mtrf TO 's3://spp-weis/data/mtrf.parquet';")

def upsert_lmp(
    lmp_upsert: pd.DataFrame,
    backfill: bool=False, #NOOP
) -> None:
    """
    Function to upsert new/backfilled LMP data into duckdb database.
    NOTE: no backfill parameter is used (as opposed to MTLF and MTLF data) since
    the prices aren't forecasted we don't need to worry about overwriting actuals
    with forecasts.
    Args:
        lmp_upsert: pd.DataFrame - dataframe to upsert to LMP table in database.
        backfill: bool=False - not used included for compatibility with collect_upsert_data()
    Returns:
        None - new data is upserted to table
    """
    # remove any duplicated primary keys
    idx = lmp_upsert[['GMTIntervalEnd_HE', 'Settlement_Location_Name', 'PNODE_Name']].duplicated()
    lmp_upsert = lmp_upsert[~idx]
    # NOTE: the df col order must match the order in the table
    ordered_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name',
        'LMP', 'MLC', 'MCC', 'MEC'
    ]
    lmp_upsert = lmp_upsert[ordered_cols]
    update_count = len(lmp_upsert)
    log.info(f'lmp_upsert.timestamp_mst_HE.min(): {lmp_upsert.timestamp_mst_HE.min()}')
    log.info(f'lmp_upsert.timestamp_mst_HE.max(): {lmp_upsert.timestamp_mst_HE.max()}')

    # upsert with duckdb
    with duckdb.connect('~/spp_weis_price_forecast/data/spp.ddb') as con_ddb:
        create_lmp = '''
        CREATE TABLE IF NOT EXISTS lmp (
             Interval_HE TIMESTAMP,
             GMTIntervalEnd_HE TIMESTAMP,
             timestamp_mst_HE TIMESTAMP,
             Settlement_Location_Name STRING, 
             PNODE_Name STRING,
             LMP DOUBLE,
             MLC DOUBLE,
             MCC DOUBLE,
             MEC DOUBLE,
             PRIMARY KEY (GMTIntervalEnd_HE, Settlement_Location_Name, PNODE_Name)
             );
        '''
        con_ddb.sql(create_lmp)

        res = con_ddb.sql('select count(*) from lmp')
        start_count = res.fetchall()[0][0]
        log.info(f'starting count: {start_count:,}')

        lmp_insert_update = '''
        INSERT INTO lmp
            SELECT * FROM lmp_upsert
            ON CONFLICT DO UPDATE SET 
            LMP = EXCLUDED.LMP, 
            MLC = EXCLUDED.MLC,
            MCC = EXCLUDED.MCC,
            MEC = EXCLUDED.MEC;
        '''

        con_ddb.sql(lmp_insert_update)

        res = con_ddb.sql('select count(*) from lmp')
        end_count = res.fetchall()[0][0]
        insert_count = end_count - start_count
        rows_updated = update_count - insert_count
        log.info(
            f'ROWS INSERTED: {insert_count:,} ROWS UPDATED: {rows_updated :,} TOTAL: {end_count:,}')

        # copy to s3
        con_ddb.sql("INSTALL httpfs;")
        con_ddb.sql("LOAD httpfs;")
        con_ddb.sql("COPY lmp TO 's3://spp-weis/data/lmp.parquet';")


def upsert_gen_cap(
        gen_cap_upsert: pd.DataFrame,
        backfill: bool=False,
) -> None:
    """
    Function to upsert new/backfilled generation capacity data into duckdb database.
    Args:
        gen_cap_upsert: pd.DataFrame - dataframe to upsert to MTRF table in database.
        backfill: bool = False - if true removes rows with missing values before
            upsert.  This removes rows where average actual is missing because
            the time period is forecasted and prevents overwriting actual values
            with forecasted values.
    Returns:
        None - new data is upserted to table
    """
    # remove missing values if backfilling
    if backfill:
        gen_cap_upsert.dropna(axis=0, how='any', inplace=True)
    # remove any duplicated primary keys
    gen_cap_upsert = gen_cap_upsert[~gen_cap_upsert.GMTIntervalEnd.duplicated()]
    update_count = len(gen_cap_upsert)
    # NOTE: the df col order must match the order in the table
    ordered_cols = [
        'GMTIntervalEnd', 'timestamp_mst',
        'Coal_Market', 'Coal_Self', 'Hydro', 
        'Natural_Gas', 'Nuclear', 'Solar', 'Wind',
    ]
    gen_cap_upsert = gen_cap_upsert[ordered_cols]
    log.info(f'gen_cap_upsert.timestamp_mst.min(): {gen_cap_upsert.timestamp_mst.min()}')
    log.info(f'gen_cap_upsert.timestamp_mst.max(): {gen_cap_upsert.timestamp_mst.max()}')

    # upsert with duckdb
    with duckdb.connect('~/spp_weis_price_forecast/data/spp.ddb') as con_ddb:
        create_gen_cap = '''
        CREATE TABLE IF NOT EXISTS gen_cap (
             GMTIntervalEnd TIMESTAMP PRIMARY KEY,
             timestamp_mst TIMESTAMP,
             Coal_Market DOUBLE, 
             Coal_Self DOUBLE,
             Hydro DOUBLE,
             Natural_Gas DOUBLE,
             Nuclear DOUBLE,
             Solar DOUBLE,
             Wind DOUBLE
             );
        '''
        con_ddb.sql(create_gen_cap)

        res = con_ddb.sql('select count(*) from gen_cap')
        start_count = res.fetchall()[0][0]
        log.info(f'starting count: {start_count:,}')

        gen_cap_insert_update = '''
        INSERT INTO gen_cap
            SELECT * FROM gen_cap_upsert
            ON CONFLICT DO UPDATE SET Coal_Market = EXCLUDED.Coal_Market, 
            Coal_Self = EXCLUDED.Coal_Self,
            Hydro = EXCLUDED.Hydro,
            Natural_Gas = EXCLUDED.Natural_Gas,
            Nuclear = EXCLUDED.Nuclear,
            Solar = EXCLUDED.Solar,
            Wind = EXCLUDED.Wind;
        '''

        con_ddb.sql(gen_cap_insert_update)

        res = con_ddb.sql('select count(*) from gen_cap')
        end_count = res.fetchall()[0][0]
        insert_count = end_count - start_count
        rows_updated = update_count - insert_count
        log.info(
            f'ROWS INSERTED: {insert_count:,} ROWS UPDATED: {rows_updated :,} TOTAL: {end_count:,}')


###########################################################
# COLLECT AND UPSERT DATA
###########################################################

def collect_upsert_data(
    get_range_data_func: Callable,
    upsert_func: Callable,
    n_periods: int,
    primary_key_cols: List[str],
    end_ts: Union[pd.Timestamp, None] = None,
    backfill: bool=False,
) -> None:
    """
    Function to combine data collection and upsert.  This is used as
    a base function.  Helper functions will wrap this function and
    pass in data specific parameters needed to collect and upsert
    new data.
    Args:
        get_range_data_func: Callable - data specific function
            to get a range of data, i.e. get_range_data_mtrf()
        n_periods: int - the number of time periods to gather prior to end_ts
        primary_key_cols: List[str] - columns that define unique
            rows.  This is used to find duplicate rows and update
            those rows with new data.
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        backfill: bool = False - if true removes rows with missing values before
            upsert.  This removes rows where average actual is missing because
            the time period is forecasted and prevents overwriting actual values
            with forecasted values.
    Returns:
        None - new data is upserted to table
    """

    # set end_ts to current time if None
    if end_ts is None:
        end_ts = pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)

    log.info(f'end_ts: {end_ts}')
    log.info(f'n_periods: {n_periods}')

    # get data
    range_df = get_range_data_func(end_ts, n_periods=n_periods)

    # upsert data
    if range_df.shape[0] > 0:
        range_df.info()
        assert not range_df[primary_key_cols].duplicated().any()
        upsert_func(range_df, backfill=backfill)

    else:
        log.info(f'no data to upsert for end_ts: {end_ts} - n_periods: {n_periods}')


def collect_upsert_mtlf(
    end_ts: Union[pd.Timestamp, None] = None,
    n_periods: int = 6,
    backfill: bool = False,
) -> None:
    """
    Helper function to wrap collect_upsert_data() with defaults for
    the MTLF data.
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
            default set to get the previous 6 hours of data
        backfill: bool = False - if true removes rows with missing values before
            upsert.  This removes rows where average actual is missing because
            the time period is forecasted and prevents overwriting actual values
            with forecasted values.
    Returns:
        None - new data is upserted to table
    """

    primary_key_cols = ['GMTIntervalEnd']

    collect_upsert_data(
        get_range_data_mtlf,
        upsert_mtlf,
        n_periods,
        primary_key_cols,
        end_ts=end_ts,
        backfill=backfill,
    )


def collect_upsert_mtrf(
    end_ts: Union[pd.Timestamp, None] = None,
    n_periods: int = 6,
    backfill: bool = False,
) -> None:
    """
    Helper function to wrap collect_upsert_data() with defaults for
    the MTRF data.
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
            default set to get the previous 6 hours of data
        backfill: bool = False - if true removes rows with missing values before
            upsert.  This removes rows where average actual is missing because
            the time period is forecasted and prevents overwriting actual values
            with forecasted values.
    Returns:
        None - new data is upserted to table
    """

    # set default table
    primary_key_cols = ['GMTIntervalEnd']

    collect_upsert_data(
        get_range_data_mtrf,
        upsert_mtrf,
        n_periods,
        primary_key_cols,
        end_ts=end_ts,
        backfill=backfill,
    )


def collect_upsert_lmp(
    daily_file: bool,
    end_ts: Union[pd.Timestamp, None] = None,
    n_periods: int = 6,
) -> None:
    """
    Helper function to wrap collect_upsert_data() with defaults for
    the LMP data.
    Args:
        daily_file: bool - flag for daily files vs 5 min interval files
            since they are formatted differently :(
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
            default set to get the previous 6 hours of data
    Returns:
        None - new data is upserted to table
    """

    primary_key_cols = ['GMTIntervalEnd_HE', 'Settlement_Location_Name', 'PNODE_Name']

    if daily_file:  # loading 5 min lmps for entire day
        lmp_get_range_data_func = get_range_data_interval_daily_lmps
    else:  # working with 5 min interval files
        lmp_get_range_data_func = get_range_data_interval_5min_lmps

    collect_upsert_data(
        lmp_get_range_data_func,
        upsert_lmp,
        n_periods,
        primary_key_cols,
        end_ts=end_ts,
    )


def collect_upsert_gen_cap(
    end_ts: Union[pd.Timestamp, None] = None,
    n_periods: int = 6,
    backfill: bool = False,
) -> None:
    """
    Helper function to wrap collect_upsert_data() with defaults for
    the MTRF data.
    Args:
        end_ts: pd.Timestamp - the last time period to get data
            i.e. pd.Timestamp('6/4/2023') or
            pd.Timestamp.utcnow().tz_convert("America/Chicago").tz_localize(None)
        n_periods: int - the number of time periods to gather prior to end_ts
            default set to get the previous 6 hours of data
        backfill: bool = False - if true removes rows with missing values before
            upsert.  This removes rows where average actual is missing because
            the time period is forecasted and prevents overwriting actual values
            with forecasted values.
    Returns:
        None - new data is upserted to table
    """

    # set default table
    primary_key_cols = ['GMTIntervalEnd']

    collect_upsert_data(
        get_range_data_gen_cap,
        upsert_gen_cap,
        n_periods,
        primary_key_cols,
        end_ts=end_ts,
        backfill=backfill,
    )


"""
Data collection for SPP Integrated Marketplace (RTO West) feeds.

The IM successor to data_collection.py (WEIS feeds, dead since 2026-04-01).
Collects from the Integrated Marketplace portal feeds into the `data_im/`
R2 prefix, alongside the untouched WEIS `data/` prefix. Feeds:

- MTLF: Mid-Term Load Forecast, per BAA (hourly)
- MTRF: Mid-Term Resource Forecast wind/solar, per BAA (hourly)
- RTBM LMP: 5-minute interval files + daily rollup, aggregated to hourly
- RF_RESERVE_ZONE: wind/solar forecasts AND actuals by reserve zone (hourly)
- DA LMP: Day-Ahead hourly LMP (collected for history; not modeled yet)

Key differences from the WEIS collectors
(see plans/weis_to_rto_west_migration.md):
    - Files cover both BAAs (East 'SPP', West 'SWPW') with a `BAA` column;
      pre-launch (< 2026-04-01) files lack the column and are East-only,
      so processors fill BAA='SPP'. Rows with a null BAA (future intervals
      not yet populated) carry no data and are dropped.
    - LMP rows are filtered to the hub/BA node list (node_list.STORED_NODES)
      at storage; other feeds are stored whole.
    - Every upsert dedup key includes BAA (UPSERT_KEYS).
    - DST fall-back publishes a duplicate-hour `...d.csv` variant for the
      interval/hourly feeds; the range generation emits an extra fetch for
      the ambiguous wall-clock times.
    - The daily LMP rollup publishes with a ~5-day lag
      (DAILY_LMP_LAG_DAYS); its range helper shifts the window.
"""
# pylint: disable=C0103,W1203,W1201

import os
import sys
import logging
from typing import List, Callable

import pandas as pd
import polars as pl
import tqdm
from pytz.exceptions import NonExistentTimeError

log = logging.getLogger(__name__)

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
        sys.path.insert(0, module_path)

from data_collection import (  # noqa: E402
    N_JOBS,
    ProgressParallel,
    _s3_storage_options,
    add_timestamp_mst,
    check_file_exists_client,
    format_df_colnames,
    get_csv_from_url,
    set_he,
)
from node_list import STORED_NODES  # noqa: E402

from joblib import delayed  # noqa: E402

PORTAL_DOWNLOAD = 'https://portal.spp.org/file-browser-api/download/'

# The daily LMP rollup for operating day D publishes at ~18:00 on D+5.
DAILY_LMP_LAG_DAYS = 5

# RTO West go-live / WEIS→IM seam: files carry the BAA column from this date
# on, and the West BAA's own market data starts here. Single home for the
# date; the backfill notebook and the WEIS stitch script both read it.
RTO_WEST_LAUNCH = pd.Timestamp('2026-04-01')

# Dedup keys for the consolidated data_im/ tables. Every key includes BAA:
# both BAAs share timestamps, so without it East and West rows clobber
# each other in the upsert.
UPSERT_KEYS = {
    'lmp': ['GMTIntervalEnd_HE', 'Settlement_Location_Name', 'PNODE_Name', 'BAA'],
    'mtlf': ['GMTIntervalEnd', 'BAA'],
    'mtrf': ['GMTIntervalEnd', 'BAA'],
    'rf_reserve_zone': ['GMTIntervalEnd', 'BAA', 'ReserveZone'],
    'da_lmp': ['GMTIntervalEnd', 'Settlement_Location_Name', 'PNODE_Name', 'BAA'],
}


###########################################################
# HELPER FUNCTIONS
###########################################################

def get_s3_base_path_im() -> str:
    """Build the base S3 path for the IM `data_im/` prefix from AWS env vars."""
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER', '')
    if not AWS_S3_BUCKET:
        raise ValueError('AWS_S3_BUCKET env var is not set')
    return f's3://{AWS_S3_BUCKET}/{AWS_S3_FOLDER}data_im/'


def get_time_components_im(
        time_str: str | None = None,
        five_min_ceil: bool = False,
        dst_variant: bool = False,
) -> dict | None:
    """
    Get formatted time components for building IM feed URLs.

    Like data_collection.get_time_components, plus DST support. Filenames
    label *interval endings*, so a file ending at T is duplicated on
    fall-back iff the interval before T is ambiguous — the probe is at
    T - 1s (this is what catches the boundary `...0200d.csv` file).
    `dst_variant=False` resolves to the first occurrence (DST, the normal
    file); `dst_variant=True` to the second (standard, the `d` file). The
    returned dict carries `DST_VARIANT` (URL builders append the suffix)
    and `IS_AMBIGUOUS` (range generation fetches both variants).

    On spring-forward, SPP still publishes the interval ending inside the
    skipped hour (verified: OP-MTLF-...0200.csv exists on 2026-03-08), so
    nonexistent local times shift forward for the provenance timestamp
    while the URL components keep the naive wall-clock label.

    Args:
        time_str: Date string to convert (e.g. '4/1/2026 07:00:00');
            current time if None.
        five_min_ceil: Ceil to the 5-minute ending interval instead of
            the hour ending interval.
        dst_variant: Resolve an ambiguous fall-back time to its second
            occurrence (the duplicate-hour `d` file).

    Returns:
        dict of URL time components, or None if the time can't be parsed.
    """
    tc = {}

    if time_str:
        time_stamp = pd.to_datetime(time_str)
    else:
        time_stamp = pd.Timestamp.now()

    if five_min_ceil:
        time_stamp = time_stamp.ceil(freq='5min')
    else:
        time_stamp = time_stamp.ceil(freq='h')

    # ambiguous: True = first occurrence (DST), False = second (standard)
    try:
        try:
            time_stamp_ct = time_stamp.tz_localize(
                "America/Chicago", ambiguous=not dst_variant
            )
        except NonExistentTimeError:
            # spring-forward gap: keep the wall-clock label, shift the
            # provenance timestamp to the next valid instant
            time_stamp_ct = time_stamp.tz_localize(
                "America/Chicago", ambiguous=not dst_variant,
                nonexistent='shift_forward',
            )
    except Exception:
        log.error(f'error parsing: {time_stamp}')
        return None

    probe = time_stamp - pd.Timedelta(seconds=1)
    try:
        is_ambiguous = (
            probe.tz_localize("America/Chicago", ambiguous=True)
            != probe.tz_localize("America/Chicago", ambiguous=False)
        )
    except NonExistentTimeError:
        # probe fell into the spring-forward gap: nothing is duplicated
        is_ambiguous = False

    # URL components come from the naive wall-clock time (the filename
    # label), not the localized timestamp
    tc['YEAR'] = str(time_stamp.year)
    tc['MONTH'] = str(time_stamp.month).zfill(2)
    tc['DAY'] = str(time_stamp.day).zfill(2)
    tc['HOUR'] = str(time_stamp.hour).zfill(2)
    tc['MINUTE'] = str(time_stamp.minute).zfill(2)
    tc['YM'] = tc['YEAR'] + tc['MONTH']
    tc['YMD'] = tc['YM'] + tc['DAY']
    tc['COMBINED'] = tc['YMD'] + tc['HOUR'] + tc['MINUTE']
    tc['timestamp'] = time_stamp_ct
    tc['timestamp_utc'] = time_stamp_ct.tz_convert(None)
    tc['DST_VARIANT'] = dst_variant
    tc['IS_AMBIGUOUS'] = is_ambiguous
    return tc


def ensure_baa(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the BAA column across pre- and post-launch files.

    Pre-launch (< 2026-04-01) files have no BAA column and are East-only:
    fill BAA='SPP'. Post-launch files carry leading rows with a null BAA
    and no values (future intervals not yet populated): drop them.

    Args:
        df: DataFrame after format_df_colnames.

    Returns:
        pl.DataFrame with a fully populated BAA column.
    """
    if 'BAA' not in df.columns:
        return df.with_columns(pl.lit('SPP').alias('BAA'))
    return df.drop_nulls(subset=['BAA'])


def convert_datetime_cols(df: pl.DataFrame, dt_cols: List[str]) -> pl.DataFrame:
    """
    Convert string datetime columns that mix two timestamp formats.

    IM feeds vary in timestamp format across their history — some files use
    '%m/%d/%Y %H:%M:%S' and others the seconds-less '%m/%d/%Y %H:%M' (also
    unpadded, e.g. '3/20/2026 0:05'). This is the single datetime parser for
    every IM feed: each value is tried against both formats, first match wins.

    Args:
        df: DataFrame with datetime strings.
        dt_cols: Column names to convert.

    Returns:
        pl.DataFrame with converted columns.

    Raises:
        ValueError: If any value matches neither format.
    """
    for col in dt_cols:
        df = df.with_columns(
            pl.coalesce(
                pl.col(col).str.to_datetime(format='%m/%d/%Y %H:%M:%S', strict=False),
                pl.col(col).str.to_datetime(format='%m/%d/%Y %H:%M', strict=False),
            ).alias(col)
        )
        null_count = df[col].null_count()
        if null_count:
            raise ValueError(f'{col}: {null_count} values matched no known datetime format')
    return df


def _parquet_output_path(url: str, base_path: str, data_category: str) -> str:
    """Derive the parquet path from the URL's csv filename (no WEIS- prefix)."""
    parquet_filename = url.split('%2F')[-1].replace('.csv', '.parquet')
    return f'{base_path}{data_category}/{parquet_filename}'


def _stamp_and_write(df: pl.DataFrame, tc: dict, url: str, output_path: str) -> str:
    """Add provenance columns (file_create_time_utc, url, source='im'), then
    write the parquet and return its path. The WEIS stitch-fill writes
    source='weis' rows separately so the consolidated tables stay traceable."""
    df = df.with_columns(
        pl.lit(tc['timestamp_utc']).alias('file_create_time_utc'),
        pl.lit(url).alias('url'),
        pl.lit('im').alias('source'),
    )
    df.unique().write_parquet(output_path, storage_options=_s3_storage_options())
    return output_path


def agg_lmp_im(five_min_lmp_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate 5-minute IM LMPs to hour ending averages, grouped per BAA.

    Args:
        five_min_lmp_df: 5-minute LMP rows with a BAA column.

    Returns:
        pl.DataFrame with hourly averaged LMP values.
    """
    group_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name', 'BAA',
    ]
    value_cols = ['LMP', 'MLC', 'MCC', 'MEC']
    return (
        five_min_lmp_df.select(group_cols + value_cols)
        .group_by(group_cols)
        .mean()
    )


###########################################################
# GET FILE URLS
###########################################################

def _dst_suffix(tc: dict) -> str:
    """Filename suffix for the DST duplicate-hour variant ('d' or '')."""
    return 'd' if tc.get('DST_VARIANT') else ''


def get_hourly_mtlf_url(tc: dict) -> str:
    """Build the IM MTLF csv url (per-BAA load forecast vs actual)."""
    path = f"%2F{tc['YEAR']}%2F{tc['MONTH']}%2F{tc['DAY']}%2FOP-MTLF-{tc['COMBINED']}{_dst_suffix(tc)}.csv"
    return f'{PORTAL_DOWNLOAD}mtlf-vs-actual?path={path}'


def get_hourly_mtrf_url(tc: dict) -> str:
    """Build the IM MTRF csv url (per-BAA wind/solar forecast)."""
    path = f"%2F{tc['YEAR']}%2F{tc['MONTH']}%2F{tc['DAY']}%2FOP-MTRF-{tc['COMBINED']}{_dst_suffix(tc)}.csv"
    return f'{PORTAL_DOWNLOAD}midterm-resource-forecast?path={path}'


def get_5min_lmp_url(tc: dict) -> str:
    """Build the IM 5-minute RTBM LMP interval csv url."""
    path = (
        f"%2F{tc['YEAR']}%2F{tc['MONTH']}%2FBy_Interval%2F{tc['DAY']}"
        f"%2FRTBM-LMP-SL-{tc['COMBINED']}{_dst_suffix(tc)}.csv"
    )
    return f'{PORTAL_DOWNLOAD}rtbm-lmp-by-location?path={path}'


def get_daily_lmp_url(tc: dict) -> str:
    """Build the IM daily RTBM LMP rollup csv url (publishes at ~D+5)."""
    path = f"%2F{tc['YEAR']}%2F{tc['MONTH']}%2FBy_Day%2FRTBM-LMP-DAILY-SL-{tc['YMD']}.csv"
    return f'{PORTAL_DOWNLOAD}rtbm-lmp-by-location?path={path}'


def get_rf_reserve_zone_url(tc: dict) -> str:
    """Build the RF_RESERVE_ZONE csv url (hourly wind/solar forecast + actuals)."""
    path = (
        f"%2F{tc['YEAR']}%2F{tc['MONTH']}%2F{tc['DAY']}"
        f"%2FRF_RESERVE_ZONE-{tc['COMBINED']}{_dst_suffix(tc)}.csv"
    )
    return f'{PORTAL_DOWNLOAD}resource-forecast-by-reserve-zone?path={path}'


def get_da_lmp_url(tc: dict) -> str:
    """Build the DA LMP csv url (one file per operating day, published D-1)."""
    path = f"%2F{tc['YEAR']}%2F{tc['MONTH']}%2FBy_Day%2FDA-LMP-SL-{tc['YMD']}0100.csv"
    return f'{PORTAL_DOWNLOAD}da-lmp-by-settlement-location?path={path}'


###########################################################
# PROCESS AND COLLECT DATA
###########################################################
# Mirrors the WEIS layout: get_range_data_im() fans a range of datetimes
# out to a get_process_* function per feed, adding a second fetch for the
# DST duplicate-hour ...d.csv on ambiguous wall-clock times.
###########################################################

def get_range_data_im(
        end_ts: pd.Timestamp,
        n_periods: int,
        freq: str,
        get_process_func: Callable,
        base_path: str | None = None,
        do_parallel: bool = True,
    ) -> List[str]:
    """
    Collect IM data for a range of time periods and write to storage.

    Args:
        end_ts: The last time period to get data.
        n_periods: Number of time periods to gather prior to end_ts.
        freq: Frequency - 'D' for daily, 'h' for hourly, '5min' for 5 minute.
        get_process_func: The feed's get_process_* function.
        base_path: Optional base path for output files. If None, uses the
            data_im/ S3 path from AWS env vars.
        do_parallel: If True, use parallel processing with joblib.

    Returns:
        List of file paths for successful writes, or URLs for files that failed to download or process.
    """
    five_min_ceil = freq == '5min'
    time_str_list = [str(dt) for dt in pd.date_range(end=end_ts, periods=n_periods, freq=freq)]

    tc_list = []
    for time_str in time_str_list:
        tc = get_time_components_im(time_str, five_min_ceil=five_min_ceil)
        if tc is None:
            continue
        tc_list.append(tc)
        if tc['IS_AMBIGUOUS']:
            # fall-back hour: also fetch the duplicate-hour ...d.csv
            tc_list.append(
                get_time_components_im(time_str, five_min_ceil=five_min_ceil, dst_variant=True)
            )
    N = len(tc_list)

    if do_parallel:
        results = (
            ProgressParallel(n_jobs=N_JOBS, total=N)
            (delayed(get_process_func)(tc, base_path=base_path) for tc in tc_list)
        )
    else:
        results = []
        for tc in tqdm.tqdm(tc_list):
            results += [get_process_func(tc, base_path=base_path)]

    # Surface the batch success rate so a systematic break (every file
    # missing or failing to parse) is distinguishable from one bad file:
    # skip-and-log per file would otherwise let a whole feed silently
    # collect nothing while the job still "succeeds".
    n_success = sum(1 for r in results if r.endswith('.parquet'))
    if results and not n_success:
        log.warning(
            f'{get_process_func.__name__}: collected 0/{len(results)} files — '
            'every fetch missed (feed outage or schema change?)'
        )
    else:
        log.info(f'{get_process_func.__name__}: collected {n_success}/{len(results)} files')
    return results


def _get_process_feed(
        tc: dict,
        url_builder: Callable,
        data_category: str,
        transform: Callable,
        base_path: str | None = None,
) -> str:
    """
    Shared scaffold for every IM feed: download, transform, write parquet.

    Args:
        tc: Time components from get_time_components_im().
        url_builder: The feed's get_*_url function.
        data_category: Output subfolder under the data_im/ prefix.
        transform: Feed-specific pl.DataFrame -> pl.DataFrame processing.
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        File path if successful, or the source URL if the download or the transform failed.
    """
    if base_path is None:
        base_path = get_s3_base_path_im()

    url = url_builder(tc)
    df = get_csv_from_url(url)
    if df.shape[0] == 0:
        return url

    # Skip an isolated malformed file (e.g. SPP's DA file for 2026-06-04
    # shipped an all-caps header) rather than aborting the whole batch;
    # the caller filters on the .parquet suffix, so a returned URL is a miss.
    try:
        df = transform(df)
    except Exception as e:
        log.error(f'transform failed, skipping {url}: {e}')
        return url
    return _stamp_and_write(df, tc, url, _parquet_output_path(url, base_path, data_category))


def _transform_mtlf(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize one MTLF file: BAA fill/drop, datetimes, Float32 casts."""
    format_df_colnames(df)
    df = ensure_baa(df)
    df = convert_datetime_cols(df, ['Interval', 'GMTIntervalEnd'])
    df = add_timestamp_mst(df)
    return df.with_columns(
        pl.col.MTLF.cast(pl.Float32),
        pl.col.Averaged_Actual.cast(pl.Float32),
    )


def _transform_mtrf(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize one MTRF file: BAA fill/drop, datetimes, Float32 casts."""
    format_df_colnames(df)
    df = ensure_baa(df)
    df = convert_datetime_cols(df, ['Interval', 'GMTIntervalEnd'])
    df = add_timestamp_mst(df)
    return df.with_columns(
        pl.col.Wind_Forecast_MW.cast(pl.Float32),
        pl.col.Solar_Forecast_MW.cast(pl.Float32),
    )


def get_process_mtlf(tc: dict, base_path: str | None = None) -> str:
    """
    Download, process, and write one IM MTLF file to storage.

    Keeps all BAAs (East 'SPP' + West 'SWPW'); pre-launch files without a
    BAA column are filled with 'SPP'.

    Args:
        tc: Time components from get_time_components_im().
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        File path if successful, or the source URL if the download or the transform failed.
    """
    return _get_process_feed(tc, get_hourly_mtlf_url, 'mtlf', _transform_mtlf, base_path)


def get_process_mtrf(tc: dict, base_path: str | None = None) -> str:
    """
    Download, process, and write one IM MTRF file to storage.

    Keeps all BAAs; pre-launch files without a BAA column are filled with
    'SPP', and unpopulated null-BAA rows are dropped.

    Args:
        tc: Time components from get_time_components_im().
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        File path if successful, or the source URL if the download or the transform failed.
    """
    return _get_process_feed(tc, get_hourly_mtrf_url, 'mtrf', _transform_mtrf, base_path)


def _process_lmp(df: pl.DataFrame, rename_map: dict) -> pl.DataFrame:
    """Shared IM LMP pipeline: normalize, filter to hub/BA nodes, agg to HE."""
    format_df_colnames(df)
    df = df.rename(rename_map)
    df = ensure_baa(df)
    df = df.filter(pl.col('Settlement_Location_Name').is_in(STORED_NODES))
    # Daily-rollup files across history mix seconded/unpadded timestamp
    # formats (e.g. '3/20/2026 0:05'), so parse flexibly like the DA feed.
    df = convert_datetime_cols(df, ['Interval', 'GMTIntervalEnd'])
    df = add_timestamp_mst(df)
    df = set_he(df)
    df = agg_lmp_im(df)
    return df.with_columns(
        pl.col.LMP.cast(pl.Float32),
        pl.col.MLC.cast(pl.Float32),
        pl.col.MCC.cast(pl.Float32),
        pl.col.MEC.cast(pl.Float32),
    )


def get_process_5min_lmp(tc: dict, base_path: str | None = None) -> str:
    """
    Download, process, and write one IM 5-minute LMP interval to storage.

    Rows are filtered to the hub/BA node list (both BAAs) before the
    hourly aggregation.

    Args:
        tc: Time components from get_time_components_im().
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        File path if successful, or the source URL if the download or the transform failed.
    """
    def transform(df: pl.DataFrame) -> pl.DataFrame:
        return _process_lmp(
            df,
            rename_map={'Settlement_Location': 'Settlement_Location_Name', 'Pnode': 'PNODE_Name'},
        )

    return _get_process_feed(tc, get_5min_lmp_url, 'lmp_5min', transform, base_path)


def get_process_daily_lmp(tc: dict, base_path: str | None = None) -> str:
    """
    Download, process, and write one IM daily LMP rollup to storage.

    The daily file holds the full day of 5-minute intervals (the trailing
    repair sweep); it publishes at ~D+5 — use the lag-aware range helper.

    Args:
        tc: Time components from get_time_components_im().
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        File path if successful, or the source URL if the download or the transform failed.
    """
    def transform(df: pl.DataFrame) -> pl.DataFrame:
        return _process_lmp(df, rename_map={'GMT_Interval': 'GMTIntervalEnd'})

    return _get_process_feed(tc, get_daily_lmp_url, 'lmp_daily', transform, base_path)


def get_process_rf_reserve_zone(tc: dict, base_path: str | None = None) -> str:
    """
    Download, process, and write one RF_RESERVE_ZONE file to storage.

    Stores all reserve zones — filtering to ReserveZone 21 (the West BAA)
    is a downstream concern. This is the only feed carrying wind/solar
    actuals.

    Args:
        tc: Time components from get_time_components_im().
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        File path if successful, or the source URL if the download or the transform failed.
    """
    def transform(df: pl.DataFrame) -> pl.DataFrame:
        format_df_colnames(df)
        df = ensure_baa(df)
        df = convert_datetime_cols(df, ['IntervalEnd', 'GMTIntervalEnd'])
        df = add_timestamp_mst(df)
        return df.with_columns(
            pl.col.WindForecastMW.cast(pl.Float32),
            pl.col.WindActualMW.cast(pl.Float32),
            pl.col.SolarForecastMW.cast(pl.Float32),
            pl.col.SolarActualMW.cast(pl.Float32),
        )

    return _get_process_feed(tc, get_rf_reserve_zone_url, 'rf_reserve_zone', transform, base_path)


def get_process_da_lmp(tc: dict, base_path: str | None = None) -> str:
    """
    Download, process, and write one DA LMP file to storage.

    Hourly day-ahead prices, one file per operating day (published the
    prior afternoon). Filtered to the hub/BA node list like RTBM LMP.
    Collected for history accrual; not consumed by the model yet.

    Args:
        tc: Time components from get_time_components_im().
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        File path if successful, or the source URL if the download or the transform failed.
    """
    def transform(df: pl.DataFrame) -> pl.DataFrame:
        format_df_colnames(df)
        df = df.rename({'Settlement_Location': 'Settlement_Location_Name', 'Pnode': 'PNODE_Name'})
        df = ensure_baa(df)
        df = df.filter(pl.col('Settlement_Location_Name').is_in(STORED_NODES))
        # DA files mix timestamp formats: some have seconds, some don't
        df = convert_datetime_cols(df, ['Interval', 'GMTIntervalEnd'])
        df = add_timestamp_mst(df)
        return df.with_columns(
            pl.col.LMP.cast(pl.Float32),
            pl.col.MLC.cast(pl.Float32),
            pl.col.MCC.cast(pl.Float32),
            pl.col.MEC.cast(pl.Float32),
        )

    return _get_process_feed(tc, get_da_lmp_url, 'da_lmp', transform, base_path)


###########################################################
# RANGE HELPERS (one per feed)
###########################################################

def get_range_data_mtlf(end_ts: pd.Timestamp, n_periods: int, base_path: str | None = None) -> List[str]:
    """Collect IM MTLF for n_periods hours ending at end_ts."""
    return get_range_data_im(end_ts, n_periods, 'h', get_process_mtlf, base_path=base_path)


def get_range_data_mtrf(end_ts: pd.Timestamp, n_periods: int, base_path: str | None = None) -> List[str]:
    """Collect IM MTRF for n_periods hours ending at end_ts."""
    return get_range_data_im(end_ts, n_periods, 'h', get_process_mtrf, base_path=base_path)


def get_range_data_5min_lmp(end_ts: pd.Timestamp, n_periods: int, base_path: str | None = None) -> List[str]:
    """Collect IM 5-minute LMP for n_periods intervals ending at end_ts."""
    return get_range_data_im(end_ts, n_periods, '5min', get_process_5min_lmp, base_path=base_path)


def get_range_data_daily_lmp(end_ts: pd.Timestamp, n_periods: int, base_path: str | None = None) -> List[str]:
    """
    Collect IM daily LMP rollups for n_periods days, lag-adjusted.

    The rollup for day D publishes at ~D+5, so the window is shifted back
    by DAILY_LMP_LAG_DAYS: pass end_ts = "now" and get the n_periods most
    recent days that have actually published.

    Args:
        end_ts: Reference time, normally the current time.
        n_periods: Number of published days to gather.
        base_path: Optional base path for output. If None, uses the
            data_im/ S3 path from AWS env vars.

    Returns:
        List of file paths for successful writes, or URLs for files that failed to download or process.
    """
    lagged_end = end_ts - pd.Timedelta(days=DAILY_LMP_LAG_DAYS)
    return get_range_data_im(lagged_end, n_periods, 'D', get_process_daily_lmp, base_path=base_path)


def get_range_data_rf_reserve_zone(end_ts: pd.Timestamp, n_periods: int, base_path: str | None = None) -> List[str]:
    """Collect RF_RESERVE_ZONE for n_periods hours ending at end_ts."""
    return get_range_data_im(end_ts, n_periods, 'h', get_process_rf_reserve_zone, base_path=base_path)


def get_range_data_da_lmp(end_ts: pd.Timestamp, n_periods: int, base_path: str | None = None) -> List[str]:
    """
    Collect DA LMP for n_periods days ending at end_ts.

    The file for operating day D publishes the prior afternoon, so
    tomorrow's file exists after ~14:00 CT — callers may pass
    end_ts = now + 1 day to pick it up.
    """
    return get_range_data_im(end_ts, n_periods, 'D', get_process_da_lmp, base_path=base_path)


###########################################################
# UPSERT DATA
###########################################################

def upsert_im(
    parquet_files: List[str],
    target: str,
    base_path: str | None = None,
) -> None:
    """
    Upsert individual IM parquet files into a consolidated data_im/ table.

    Deduplicates by the target's UPSERT_KEYS (keeping the latest row by
    file_create_time_utc), merges with the existing consolidated file if
    present, and writes the result to {base_path}{target}.parquet.

    Args:
        parquet_files: Paths of individual parquet files to upsert.
        target: One of UPSERT_KEYS: 'lmp' (5-min + daily rollup), 'mtlf',
            'mtrf', 'rf_reserve_zone', 'da_lmp'.
        base_path: Optional base path. If None, uses the data_im/ S3 path
            from AWS env vars.

    Returns:
        None - writes the consolidated parquet.
    """
    if target not in UPSERT_KEYS:
        raise ValueError(f'{target = } - expected one of {sorted(UPSERT_KEYS)}')
    key_cols = UPSERT_KEYS[target]

    if base_path is None:
        base_path = get_s3_base_path_im()
    target_path = f'{base_path}{target}.parquet'

    if target_path.startswith('s3://'):
        bucket_name, object_name = target_path.replace('s3://', '').split('/', 1)
        file_exists = check_file_exists_client(bucket_name, object_name)
    else:
        file_exists = os.path.exists(target_path)
    log.info(f'{target_path = }')
    log.info(f'{file_exists = }')
    log.info(f'number of files upserting: {len(parquet_files)}')

    storage_opts = _s3_storage_options()

    def dedup(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .sort(key_cols + ['file_create_time_utc'], descending=False)
            .unique(subset=key_cols, keep='last', maintain_order=True)
        )

    upsert_df = dedup(pl.scan_parquet(parquet_files, storage_options=storage_opts)).collect()
    log.info(f'{upsert_df.shape = }')
    update_count = upsert_df.shape[0]

    time_col = 'GMTIntervalEnd' if 'GMTIntervalEnd' in upsert_df.columns else 'GMTIntervalEnd_HE'
    min_max = upsert_df.select(
        pl.col(time_col).min().alias('min_date'),
        pl.col(time_col).max().alias('max_date'),
    )
    log.info(f'min/max update times: \n{min_max}')

    if file_exists:
        target_df = pl.read_parquet(target_path, storage_options=storage_opts)
        start_count = target_df.shape[0]
        target_df = dedup(pl.concat([target_df, upsert_df]).lazy()).collect()
    else:
        start_count = 0
        target_df = upsert_df

    num_dups = target_df.select(key_cols).is_duplicated().sum()
    if num_dups != 0:
        raise RuntimeError(f'duplicate keys after dedup: {num_dups = }')

    target_df.write_parquet(target_path, storage_options=storage_opts)

    end_count = target_df.shape[0]
    insert_count = end_count - start_count
    log.info(
        f'ROWS INSERTED: {insert_count:,} - '
        f'ROWS UPDATED: {update_count - insert_count:,} - TOTAL: {end_count:,}'
    )

"""
Data engineering module for SPP western-market (RTO West / Integrated
Marketplace) price forecasting.

This module provides functions to prepare data for model training and forecasting
using polars for data manipulation and duckdb for database operations. It handles:

- Loading data from S3 parquet files into DuckDB
- Preparing LMP, MTLF, MTRF, generation capacity, and weather data
- Feature engineering (rolling windows, ratios, differencing)
- Creating time series objects for Darts forecasting models

Dependencies:
    - polars: DataFrame operations and transformations
    - duckdb: In-memory database for data storage and querying (with httpfs for S3 access)
    - darts: Time series creation and missing value handling
"""

# base imports
import os
import sys
from typing import Optional, List

# data processing
import pandas as pd
import duckdb
import polars as pl
from darts.dataprocessing.transformers import MissingValuesFiller
from darts import TimeSeries

import warnings
import logging

warnings.filterwarnings("ignore")

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Put this module's own directory (src/) on sys.path so the bare intra-src
# imports below (parameters, node_list) resolve no matter where the app,
# a notebook, or a job is launched from. Deriving it from __file__ works on
# any machine/checkout path, unlike hardcoded HOME-relative guesses.
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import parameters  # noqa: E402  (imported after the sys.path shim above)
import node_list  # noqa: E402
from data_collection_utils import IM_PREFIX  # noqa: E402


#############################################
# parameters for column names
#############################################
FUTR_COLS = [
    'MTLF', 'Wind_Forecast_MW', 'Solar_Forecast_MW',
    're_ratio', 're_diff',
    'load_net_re',
    'load_net_re_diff',
    'load_net_re_diff_rolling_2',
    'load_net_re_diff_rolling_3',
    'load_net_re_diff_rolling_4',
    'load_net_re_diff_rolling_6',
    # 'temperature',
]

PAST_COLS = [
    'Averaged_Actual',
    'lmp_diff',
    'lmp_diff_rolling_2',
    'lmp_diff_rolling_3',
    'lmp_diff_rolling_4',
    'lmp_diff_rolling_6',
    'lmp_load_net_re',
    ]

Y = ['LMP']
IDS = ['unique_id']


#############################################
# create database
#############################################
def create_database(
    datasets: List[str]=['lmp', 'mtrf', 'mtlf'],
    target: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """
    Create an in-memory DuckDB database from S3 parquet files.

    Reads parquet files directly from S3 using DuckDB's httpfs extension and
    creates tables in an in-memory database. No local file downloads required.

    Args:
        datasets: List of dataset names to load. Each name corresponds
            to a parquet file in S3 (e.g., 'mtlf' -> 'im/mtlf.parquet').
            Defaults to ['lmp', 'mtrf', 'mtlf'].
        target: Forecast target (parameters.TARGETS); defaults to
            parameters.DEFAULT_TARGET. The price target table is always named
            'lmp' downstream, but is loaded from the target's source parquet —
            real-time ('im/lmp.parquet') or day-ahead ('im/da_lmp.parquet').
            DA is already hourly, so its Interval/GMTIntervalEnd/timestamp_mst
            columns are renamed to the *_HE (hour-ending) names the RT-shaped
            pipeline expects, making the 'lmp' table schema-identical either way.

    Returns:
        duckdb.DuckDBPyConnection: Connection to in-memory DuckDB database
            with tables created for each dataset.

    Environment Variables:
        AWS_S3_BUCKET: S3 bucket containing the parquet files.
        AWS_S3_FOLDER: Folder prefix within the bucket where data is stored.
    """
    if target is None:
        target = parameters.DEFAULT_TARGET
    source_dataset = parameters.TARGETS[target]['source_dataset']

    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_S3_FOLDER = os.environ.get('AWS_S3_FOLDER', '')
    if not AWS_S3_BUCKET:
        raise ValueError('AWS_S3_BUCKET env var is not set')
    log.info(f'{AWS_S3_BUCKET = }')
    log.info(f'{AWS_S3_FOLDER = }')
    log.info(f'target = {target!r} (price source: {source_dataset})')

    con = duckdb.connect()
    con.sql("INSTALL httpfs;")
    con.sql("LOAD httpfs;")

    endpoint = os.getenv("S3_ENDPOINT_URL", "").replace("https://", "")
    if endpoint:
        con.sql(f"SET s3_endpoint = '{endpoint}';")
        con.sql("SET s3_url_style = 'path';")

    s3_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    s3_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    s3_region = os.getenv("AWS_DEFAULT_REGION", "auto")
    if s3_key and s3_secret:
        con.sql(f"SET s3_access_key_id = '{s3_key}';")
        con.sql(f"SET s3_secret_access_key = '{s3_secret}';")
        con.sql(f"SET s3_region = '{s3_region}';")

    for ds in datasets:
        # The price target table is always named 'lmp' downstream; load it from
        # the target's source parquet. Other datasets map name -> im/<name>.parquet.
        parquet_ds = source_dataset if ds == 'lmp' else ds
        pf = f's3://{AWS_S3_BUCKET}/{AWS_S3_FOLDER}{IM_PREFIX}{parquet_ds}.parquet'
        log.info(f'loading table {ds} from {pf}')
        if ds == 'lmp' and source_dataset == 'da_lmp':
            # DA is already hourly: rename its interval columns to the *_HE names
            # (RENAME, not alias — no duplicate columns), so the 'lmp' table is
            # schema-identical to RT and every downstream step is reused as-is.
            select = (
                "SELECT * RENAME (Interval AS Interval_HE, "
                "GMTIntervalEnd AS GMTIntervalEnd_HE, "
                f"timestamp_mst AS timestamp_mst_HE) FROM read_parquet('{pf}')"
            )
        else:
            select = f"SELECT * FROM read_parquet('{pf}')"
        # ds and pf are code-controlled, not user input
        con.execute(f"CREATE TABLE {ds} AS {select}")  # noqa: S608

    return con


#############################################
# data prep
#############################################
def _default_start_time() -> pd.Timestamp:
    """Default training-window start: the last TRAIN_START, but never before
    the RTO West launch.

    WEIS-era prices are a different, much calmer regime (~half the RTO West
    volatility, only shallow negatives), so mixing them in biases the model
    toward flat forecasts. Clamping here keeps training IM-only for now and
    slides into a normal rolling year once a full year of IM data exists.
    """
    return max(
        pd.Timestamp.now() - pd.Timedelta(parameters.TRAIN_START),
        node_list.RTO_WEST_LAUNCH,
    )


def prep_lmp(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    baa: str = node_list.WEST_BAA,
    nodes: Optional[List[str]] = None,
    clip_outliers: bool = False,
) -> pl.DataFrame:
    """
    Prepare LMP (Locational Marginal Price) data from DuckDB.

    Filters, transforms, and engineers features for LMP price data including
    BAA/location filtering, time range filtering, outlier clipping, and price
    differencing calculations.

    Args:
        con: DuckDB connection with 'lmp' table loaded.
        start_time: Start of time range filter. If None, uses TRAIN_START
            parameter (default ~1.5 years ago).
        end_time: End of time range filter. If None, no upper bound.
        baa: Balancing authority area to keep. Defaults to 'SWPW' (SPP West);
            the im/ table holds both BAAs.
        nodes: Settlement locations to keep. Defaults to the modeled/app node
            list (node_list.MODEL_APP_NODES).
        clip_outliers: If True, clip LMP values to 0.25% and 99.75% quantiles.

    Returns:
        pl.DataFrame: Processed LMP data with columns including 'unique_id',
            'timestamp_mst', 'LMP', and 'lmp_diff' (price change from previous hour).
    """
    lmp = con.execute("SELECT * FROM lmp").pl()

    # filter to the West BAA and the modeled/app nodes
    if nodes is None:
        nodes = node_list.MODEL_APP_NODES
    lmp = lmp.filter(
        (pl.col("BAA") == baa)
        & pl.col("Settlement_Location_Name").is_in(nodes)
    )

    drop_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name', 'BAA', 'source',
        'MLC', 'MCC', 'MEC'
    ]

    if not start_time:
        start_time = _default_start_time()

    # TODO: handle checks for start_time < end_time
    lmp = lmp.filter(pl.col("timestamp_mst_HE") >= start_time)

    if clip_outliers:
        clipped_lwr = lmp.select(pl.col("LMP").quantile(0.0025)).item()
        clipped_upr = lmp.select(pl.col("LMP").quantile(0.9975)).item()
        lmp = lmp.with_columns(
            pl.when(pl.col("LMP") > clipped_upr).then(clipped_upr)
            .when(pl.col("LMP") < clipped_lwr).then(clipped_lwr)
            .otherwise(pl.col("LMP")).alias("LMP")
        )

    if end_time:
        lmp = lmp.filter(pl.col("timestamp_mst_HE") <= end_time)

    lmp = (
        lmp
        .drop('file_create_time_utc', 'url', strict=False)
        .with_columns(pl.col("Settlement_Location_Name").alias("unique_id"))
        .drop_nulls(subset=["unique_id"])
        .with_columns(pl.col("timestamp_mst_HE").alias("timestamp_mst"))
        .with_columns(pl.col("LMP").cast(pl.Float32))
        .drop([c for c in drop_cols if c in lmp.columns], strict=False)
        .group_by(["unique_id", "timestamp_mst"])
        .mean()
        .sort(["unique_id", "timestamp_mst"])
        .with_columns(
            (pl.col("LMP") - pl.col("LMP").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("lmp_diff")
        )
    )

    return lmp


def _prep_baa_hourly(
    df: pl.DataFrame,
    value_cols: List[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    baa: str = node_list.WEST_BAA,
) -> pl.DataFrame:
    """
    Shared prep for the per-BAA hourly forecast tables (MTLF, MTRF).

    Filters to one BAA (im/ holds both), time-windows, casts the value
    columns to Float32, and averages to one row per timestamp.

    Args:
        df: Raw table read from DuckDB (mtlf or mtrf).
        value_cols: Numeric columns to keep and cast (e.g. ['MTLF', ...]).
        start_time: Start of time range filter. If None, uses TRAIN_START.
        end_time: End of time range filter. If None, no upper bound.
        baa: Balancing authority area to keep (defaults to SPP West).

    Returns:
        pl.DataFrame with 'timestamp_mst' and the value columns.
    """
    df = df.filter(pl.col("BAA") == baa)
    drop_cols = ['Interval', 'GMTIntervalEnd', 'BAA', 'source']

    if not start_time:
        start_time = _default_start_time()
    df = df.filter(pl.col("timestamp_mst") >= start_time)
    if end_time:
        df = df.filter(pl.col("timestamp_mst") <= end_time)

    return (
        df
        .drop('file_create_time_utc', 'url', strict=False)
        .with_columns([pl.col(c).cast(pl.Float32) for c in value_cols])
        .drop([c for c in drop_cols if c in df.columns], strict=False)
        .group_by("timestamp_mst")
        .mean()
        .sort("timestamp_mst")
    )


def prep_mtrf(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    baa: str = node_list.WEST_BAA,
) -> pl.DataFrame:
    """
    Prepare MTRF (Mid-Term Resource Forecast) data from DuckDB.

    Processes renewable generation forecast data (wind and solar) for one BAA.

    Args:
        con: DuckDB connection with 'mtrf' table loaded.
        start_time: Start of time range filter. If None, uses TRAIN_START.
        end_time: End of time range filter. If None, no upper bound.
        baa: Balancing authority area to keep. Defaults to 'SWPW' (SPP West);
            the im/ table holds both BAAs, so this must be set or the
            forecast becomes a whole-RTO aggregate.

    Returns:
        pl.DataFrame: Processed MTRF data with 'timestamp_mst',
            'Wind_Forecast_MW', and 'Solar_Forecast_MW' columns.
    """
    mtrf = con.execute("SELECT * FROM mtrf").pl()
    return _prep_baa_hourly(
        mtrf, ['Wind_Forecast_MW', 'Solar_Forecast_MW'],
        start_time=start_time, end_time=end_time, baa=baa,
    )


def prep_mtlf(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    baa: str = node_list.WEST_BAA,
) -> pl.DataFrame:
    """
    Prepare MTLF (Mid-Term Load Forecast) data from DuckDB.

    Processes load forecast data (forecast and actual load) for one BAA.

    Args:
        con: DuckDB connection with 'mtlf' table loaded.
        start_time: Start of time range filter. If None, uses TRAIN_START.
        end_time: End of time range filter. If None, no upper bound.
        baa: Balancing authority area to keep. Defaults to 'SWPW' (SPP West);
            the im/ table holds both BAAs, so this must be set or the
            forecast becomes a whole-RTO aggregate.

    Returns:
        pl.DataFrame: Processed MTLF data with 'timestamp_mst', 'MTLF',
            and 'Averaged_Actual' columns.
    """
    mtlf = con.execute("SELECT * FROM mtlf").pl()
    return _prep_baa_hourly(
        mtlf, ['MTLF', 'Averaged_Actual'],
        start_time=start_time, end_time=end_time, baa=baa,
    )


def prep_gen_cap(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pl.DataFrame:
    """
    Prepare generation capacity data from DuckDB.

    Processes generation capacity data by fuel type, combining coal market
    and self-scheduled into a single Coal column.

    Args:
        con: DuckDB connection with 'gen_cap' table loaded.
        start_time: Start of time range filter. If None, uses TRAIN_START.
        end_time: End of time range filter. If None, no upper bound.

    Returns:
        pl.DataFrame: Processed generation capacity data with 'timestamp_mst',
            'Coal', 'Hydro', 'Natural_Gas', 'Nuclear', 'Solar', and 'Wind'.
    """
    gen_cap = con.execute("SELECT * FROM gen_cap").pl()
    drop_cols = ['GMTIntervalEnd', 'Coal_Market', 'Coal_Self']

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.now("UTC") - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    gen_cap = gen_cap.filter(pl.col("timestamp_mst") >= start_time)

    if end_time:
        gen_cap = gen_cap.filter(pl.col("timestamp_mst") <= end_time)

    gen_cap = (
        gen_cap
        .with_columns((pl.col("Coal_Market") + pl.col("Coal_Self")).cast(pl.Float32).alias("Coal"))
        .with_columns(pl.col("Hydro").cast(pl.Float32))
        .with_columns(pl.col("Natural_Gas").cast(pl.Float32))
        .with_columns(pl.col("Nuclear").cast(pl.Float32))
        .with_columns(pl.col("Solar").cast(pl.Float32))
        .with_columns(pl.col("Wind").cast(pl.Float32))
        .drop([c for c in drop_cols if c in gen_cap.columns], strict=False)
        .sort("timestamp_mst")
    )

    return gen_cap


def prep_weather(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pl.DataFrame:
    """
    Prepare weather data from DuckDB.

    Processes weather data including temperature readings.

    Args:
        con: DuckDB connection with 'weather' table loaded.
        start_time: Start of time range filter. If None, uses TRAIN_START.
        end_time: End of time range filter. If None, no upper bound.

    Returns:
        pl.DataFrame: Processed weather data with 'timestamp_mst' and
            'temperature' columns.
    """
    weather = con.execute("SELECT * FROM weather").pl()
    drop_cols = ['timestamp']

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.now() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    weather = weather.filter(pl.col("timestamp_mst") >= start_time)

    if end_time:
        weather = weather.filter(pl.col("timestamp_mst") <= end_time)

    weather = (
        weather
        .with_columns(pl.col("temperature").cast(pl.Float32))
        .drop([c for c in drop_cols if c in weather.columns], strict=False)
        .sort("timestamp_mst")
    )

    return weather


def prep_all_df(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    clip_outliers: bool = False,
) -> pl.DataFrame:
    """
    Prepare combined dataset with all features for modeling.

    Joins LMP, MTLF, and MTRF data and engineers additional features including:
    - Renewable energy ratios and differences
    - Load net of renewable generation
    - Rolling window aggregations for price and load differences

    Args:
        con: DuckDB connection with required tables loaded.
        start_time: Start of time range filter. If None, uses TRAIN_START.
        end_time: End of time range filter. If None, no upper bound.
        clip_outliers: If True, clip LMP values to quantile bounds.

    Returns:
        pl.DataFrame: Combined dataset with all features ready for modeling,
            including engineered features like 're_ratio', 'load_net_re',
            and rolling window aggregations.
    """
    log.info('preparing lmp')
    lmp = prep_lmp(con, start_time=start_time, end_time=end_time, clip_outliers=clip_outliers)
    log.info(f'{lmp.shape = }')
    log.info('preparing mtlf')
    mtlf = prep_mtlf(con, start_time=start_time, end_time=end_time)
    log.info(f'{mtlf.shape = }')
    log.info('preparing mtrf')
    mtrf = prep_mtrf(con, start_time=start_time, end_time=end_time)
    log.info(f'{mtrf.shape = }')
    # weather = prep_weather(con, start_time=start_time, end_time=end_time)

    # join into single dataset
    log.info('joining mtrf')
    all_df = (
        mtlf
        .join(mtrf, on="timestamp_mst", how="left")
        # .join(weather, on="timestamp_mst", how="left", suffix="_weather")
    )
    # remove duplicate columns from joins
    log.info('removing columns')
    all_df = all_df.select([c for c in all_df.columns if not c.endswith("_right")])

    # create cross join of timestamps with unique_ids
    unique_ids = lmp.select("unique_id").unique()
    timestamps = all_df.select("timestamp_mst")
    ids_df = timestamps.join(unique_ids, how="cross").unique()
    log.info(f'{ids_df.shape = }')

    log.info('joining lmps')
    all_df = (
        all_df
        .join(ids_df, on="timestamp_mst", how="left")
        .join(lmp, on=["unique_id", "timestamp_mst"], how="left", suffix="_lmp")
    )
    log.info(f'{all_df.shape = }')
    # remove duplicate columns from joins
    log.info('removing columns')
    all_df = all_df.select([c for c in all_df.columns if not c.endswith("_right") and not c.endswith("_lmp")])
    log.info(f'{all_df.shape = }')

    log.info('filter and sort')
    all_df = (
        all_df
        .filter(pl.col("timestamp_mst") >= pd.Timestamp("2023-05-15"))  # some bad data early on...
        .sort(["unique_id", "timestamp_mst"])
        .drop_nulls(subset=["unique_id"])
    )
    log.info(f'{all_df.shape = }')

    log.info('engineer features')
    all_df = (
        all_df
        .with_columns(
            (pl.col("LMP") - pl.col("LMP").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("lmp_diff")
        )
        .with_columns(
            ((pl.col("Wind_Forecast_MW") + pl.col("Solar_Forecast_MW")) / pl.col("MTLF"))
            .cast(pl.Float32).alias("re_ratio")
        )
        .with_columns(
            (pl.col("re_ratio") - pl.col("re_ratio").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("re_diff")
        )
        .with_columns(
            (pl.col("MTLF") - pl.col("MTLF").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("mtlf_diff")
        )
        .with_columns(
            (pl.col("Wind_Forecast_MW") - pl.col("Wind_Forecast_MW").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("wind_diff")
        )
        .with_columns(
            (pl.col("Solar_Forecast_MW") - pl.col("Solar_Forecast_MW").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("solar_diff")
        )
        .with_columns(
            (pl.col("MTLF") - pl.col("Wind_Forecast_MW") - pl.col("Solar_Forecast_MW"))
            .cast(pl.Float32).alias("load_net_re")
        )
        .with_columns(
            pl.when(pl.col("load_net_re").abs() < 1.0)
            .then(1.0)
            .otherwise(pl.col("load_net_re"))
            .cast(pl.Float32).alias("load_net_re")  # avoid div/0 errors
        )
        .with_columns(
            (pl.col("load_net_re") - pl.col("load_net_re").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("load_net_re_diff")
        )
        .with_columns(
            (pl.col("LMP") / pl.col("load_net_re"))
            .cast(pl.Float32).alias("lmp_load_net_re")
        )
    )

    # rolling window aggregations
    for i in [2, 3, 4, 5, 6]:
        all_df = (
            all_df
            .with_columns(
                pl.col("lmp_diff")
                .rolling_sum(window_size=i + 1)
                .over("unique_id")
                .cast(pl.Float32)
                .alias(f"lmp_diff_rolling_{i}")
            )
            .with_columns(
                pl.col("load_net_re_diff")
                .rolling_sum(window_size=i + 1)
                .over("unique_id")
                .cast(pl.Float32)
                .alias(f"load_net_re_diff_rolling_{i}")
            )
        )
    log.info(f'{all_df.shape = }')

    return all_df


def all_df_to_pandas(all_df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert polars DataFrame to pandas with proper indexing and column selection.

    Converts the combined feature DataFrame to pandas format, sets the timestamp
    index, selects relevant columns, and removes excluded node IDs.

    Args:
        all_df: Polars DataFrame from prep_all_df().

    Returns:
        pd.DataFrame: Pandas DataFrame with 'timestamp_mst' as index and
            columns for IDS, Y (target), PAST_COLS, and FUTR_COLS.
    """
    all_df_pd = all_df.to_pandas()
    all_df_pd.set_index('timestamp_mst', inplace=True)
    all_df_pd = all_df_pd[IDS + Y + PAST_COLS + FUTR_COLS]
    return all_df_pd


def get_train_test_all(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    clip_outliers: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split LMP data into train, test, and combined datasets.

    Creates temporal train/test splits based on INPUT_CHUNK_LENGTH parameter,
    with a buffer at the end for price revisions.

    Args:
        con: DuckDB connection with 'lmp' table loaded.
        start_time: Start of time range filter. If None, uses TRAIN_START.
        end_time: End of time range filter. If None, no upper bound.
        clip_outliers: If True, clip LMP values to quantile bounds.

    Returns:
        Tuple of (lmp_all, train_all, test_all, train_test_all) pandas DataFrames:
            - lmp_all: Full filtered LMP dataset
            - train_all: Training data (up to split point)
            - test_all: Test data (after split point)
            - train_test_all: Combined train and test data
    """
    lmp_all = prep_lmp(con, start_time=start_time, end_time=end_time, clip_outliers=clip_outliers)
    lmp_all = lmp_all.to_pandas()
    lmp_all.set_index('timestamp_mst', inplace=True)

    # remove last week of prices since they might get revised
    test_end_buffer = 168
    train_start = lmp_all.index.min() + pd.Timedelta(f'{2 * parameters.INPUT_CHUNK_LENGTH}h')
    test_end = lmp_all.index.max() - pd.Timedelta(f'{test_end_buffer}h')
    tr_tst_split = test_end - pd.Timedelta(f'{2 * parameters.INPUT_CHUNK_LENGTH}h')
    log.info(f'train_start: {train_start}')
    log.info(f'tr_tst_split: {tr_tst_split}')
    log.info(f'test_end: {test_end}')

    train_idx = (lmp_all.index > train_start) & (lmp_all.index < tr_tst_split)
    test_idx = (lmp_all.index > tr_tst_split) & (lmp_all.index < test_end)
    all_idx = (lmp_all.index > train_start) & (lmp_all.index < test_end)

    train_all = lmp_all[train_idx]
    test_all = lmp_all[test_idx]
    train_test_all = lmp_all[all_idx]

    return lmp_all, train_all, test_all, train_test_all


def fill_missing(series: List[TimeSeries]) -> None:
    """
    Fill missing values in a list of TimeSeries objects in-place.

    Uses Darts MissingValuesFiller transformer to interpolate missing values
    for each series in the list.

    Args:
        series: List of Darts TimeSeries objects to fill. Modified in-place.
    """
    for i in range(len(series)):
        transformer = MissingValuesFiller()
        series[i] = transformer.transform(series[i])


def get_series(lmp_all: pd.DataFrame) -> List[TimeSeries]:
    """
    Create target TimeSeries objects from LMP price data.

    Converts pandas DataFrame to list of Darts TimeSeries objects grouped
    by unique_id (price node), with missing dates filled.

    Args:
        lmp_all: Pandas DataFrame with 'unique_id' and LMP target column,
            indexed by timestamp.

    Returns:
        List[TimeSeries]: List of Darts TimeSeries objects, one per price node,
            with missing values filled.
    """
    all_series = TimeSeries.from_group_dataframe(
        lmp_all,
        group_cols=IDS,
        value_cols=Y,
        fill_missing_dates=True,
        freq='h',
    )

    fill_missing(all_series)
    return all_series


def get_futr_cov(all_df_pd: pd.DataFrame) -> List[TimeSeries]:
    """
    Create future covariate TimeSeries objects for forecasting.

    Converts pandas DataFrame to list of Darts TimeSeries objects for
    features known in the future (forecasts like MTLF, wind, solar).

    Args:
        all_df_pd: Pandas DataFrame from all_df_to_pandas() with FUTR_COLS.

    Returns:
        List[TimeSeries]: List of Darts TimeSeries for future covariates,
            one per price node, with missing values filled.
    """
    futr_cov = TimeSeries.from_group_dataframe(
        all_df_pd,
        group_cols=IDS,
        value_cols=FUTR_COLS,
        fill_missing_dates=True,
        freq='h',
    )
    fill_missing(futr_cov)
    return futr_cov


def get_past_cov(all_df_pd: pd.DataFrame) -> List[TimeSeries]:
    """
    Create past covariate TimeSeries objects for forecasting.

    Converts pandas DataFrame to list of Darts TimeSeries objects for
    features only known historically (actual prices, actuals).

    Args:
        all_df_pd: Pandas DataFrame from all_df_to_pandas() with PAST_COLS.

    Returns:
        List[TimeSeries]: List of Darts TimeSeries for past covariates,
            one per price node, with missing values filled.
    """
    past_cov = TimeSeries.from_group_dataframe(
        all_df_pd,
        group_cols=IDS,
        value_cols=PAST_COLS,
        fill_missing_dates=True,
        freq='h',
    )
    fill_missing(past_cov)
    return past_cov

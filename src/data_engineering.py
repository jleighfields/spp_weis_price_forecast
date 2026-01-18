'''
Data engineering functions to prepare data for model training and forecasting
'''

# base imports
import os
import sys
from typing import Optional, List

# data processing
import boto3
import pandas as pd
import duckdb
import polars as pl
from darts.dataprocessing.transformers import MissingValuesFiller
from darts import TimeSeries

import warnings
warnings.filterwarnings("ignore")

# logging
import logging

# define log
logging.basicConfig(level=logging.INFO)
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
        log.info('adding module path')
        sys.path.insert(0, module_path)

# from src import params
import parameters


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
    datasets: List[str]=['lmp', 'mtrf', 'mtlf', 'weather']
) -> duckdb.DuckDBPyConnection:

    # client for getting parquets
    s3 = boto3.client('s3')

    os.makedirs('data', exist_ok=True)
    # create file paths for data
    file_paths = [f'data/{ds}.parquet' for ds in datasets]

    for fp in file_paths:
        log.info(f'getting: {fp} from s3')
        s3.download_file(Bucket='spp-weis', Key=fp, Filename=fp)

    log.info(f'os.listdir(data): {os.listdir("data")}')

    con = duckdb.connect()

    for i, ds in enumerate(datasets):
        log.info(f'loading: {ds}')
        log.info(f'file_paths[i]: {file_paths[i]}')
        con.execute(f"CREATE TABLE {ds} AS SELECT * FROM '{file_paths[i]}'")

    return con


#############################################
# data prep
#############################################
def prep_lmp(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    loc_filter: str = 'PSCO_',
    clip_outliers: bool = False,
) -> pl.DataFrame:
    lmp = con.execute("SELECT * FROM lmp").pl()

    # filter by location
    lmp = lmp.filter(pl.col("Settlement_Location_Name").str.contains(loc_filter))

    drop_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name',
        'MLC', 'MCC', 'MEC'
    ]

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.now() - pd.Timedelta(parameters.TRAIN_START)

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
        .with_columns(pl.col("Settlement_Location_Name").alias("unique_id"))
        .filter(~pl.col("unique_id").str.contains("_ARPA"))
        .drop_nulls(subset=["unique_id"])
        .with_columns(pl.col("timestamp_mst_HE").alias("timestamp_mst"))
        .with_columns(pl.col("LMP").cast(pl.Float32))
        .drop([c for c in drop_cols if c in lmp.columns], strict=False)
        .sort(["unique_id", "timestamp_mst"])
        .with_columns(
            (pl.col("LMP") - pl.col("LMP").shift(1).over("unique_id"))
            .cast(pl.Float32).alias("lmp_diff")
        )
    )

    return lmp


def prep_mtrf(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pl.DataFrame:
    mtrf = con.execute("SELECT * FROM mtrf").pl()
    drop_cols = ['Interval', 'GMTIntervalEnd']

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.now() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    mtrf = mtrf.filter(pl.col("timestamp_mst") >= start_time)

    if end_time:
        mtrf = mtrf.filter(pl.col("timestamp_mst") <= end_time)

    mtrf = (
        mtrf
        .with_columns(pl.col("Wind_Forecast_MW").cast(pl.Float32))
        .with_columns(pl.col("Solar_Forecast_MW").cast(pl.Float32))
        .drop([c for c in drop_cols if c in mtrf.columns], strict=False)
        .sort("timestamp_mst")
    )

    return mtrf


def prep_mtlf(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pl.DataFrame:
    mtlf = con.execute("SELECT * FROM mtlf").pl()
    drop_cols = ['Interval', 'GMTIntervalEnd']

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.now() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    mtlf = mtlf.filter(pl.col("timestamp_mst") >= start_time)

    if end_time:
        mtlf = mtlf.filter(pl.col("timestamp_mst") <= end_time)

    mtlf = (
        mtlf
        .with_columns(pl.col("MTLF").cast(pl.Float32))
        .with_columns(pl.col("Averaged_Actual").cast(pl.Float32))
        .drop([c for c in drop_cols if c in mtlf.columns], strict=False)
        .sort("timestamp_mst")
    )

    return mtlf


def prep_gen_cap(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pl.DataFrame:
    gen_cap = con.execute("SELECT * FROM gen_cap").pl()
    drop_cols = ['GMTIntervalEnd', 'Coal_Market', 'Coal_Self']

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.utcnow() - pd.Timedelta(parameters.TRAIN_START)

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
    lmp = prep_lmp(con, start_time=start_time, end_time=end_time, clip_outliers=clip_outliers)
    mtlf = prep_mtlf(con, start_time=start_time, end_time=end_time)
    mtrf = prep_mtrf(con, start_time=start_time, end_time=end_time)
    # weather = prep_weather(con, start_time=start_time, end_time=end_time)

    # join into single dataset
    all_df = (
        mtlf
        .join(mtrf, on="timestamp_mst", how="left")
        # .join(weather, on="timestamp_mst", how="left", suffix="_weather")
    )
    # remove duplicate columns from joins
    all_df = all_df.select([c for c in all_df.columns if not c.endswith("_right")])

    # create cross join of timestamps with unique_ids
    unique_ids = lmp.select("unique_id").unique()
    timestamps = all_df.select("timestamp_mst")
    ids_df = timestamps.join(unique_ids, how="cross")

    all_df = (
        all_df
        .join(ids_df, on="timestamp_mst", how="left")
        .join(lmp, on=["unique_id", "timestamp_mst"], how="left", suffix="_lmp")
    )
    # remove duplicate columns from joins
    all_df = all_df.select([c for c in all_df.columns if not c.endswith("_right") and not c.endswith("_lmp")])

    # filter and sort
    all_df = (
        all_df
        .filter(pl.col("timestamp_mst") >= pd.Timestamp("2023-05-15"))  # some bad data early on...
        .sort(["unique_id", "timestamp_mst"])
        .drop_nulls(subset=["unique_id"])
    )

    # engineer features
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

    return all_df


def all_df_to_pandas(all_df: pl.DataFrame):
    all_df_pd = all_df.to_pandas()
    all_df_pd.set_index('timestamp_mst', inplace=True)
    all_df_pd = all_df_pd[IDS + Y + PAST_COLS + FUTR_COLS]
    all_df_pd = all_df_pd[~all_df_pd.unique_id.isin(parameters.IDS_TO_REMOVE)]
    
    return all_df_pd


def get_train_test_all(
    con: duckdb.DuckDBPyConnection,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    clip_outliers: bool = False,
):
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

    # some BAA (ids) don't have enough training data
    # so they need to be removed
    lmp_all = lmp_all[~lmp_all.unique_id.isin(parameters.IDS_TO_REMOVE)]
    train_all = train_all[~train_all.unique_id.isin(parameters.IDS_TO_REMOVE)]
    test_all = test_all[~test_all.unique_id.isin(parameters.IDS_TO_REMOVE)]
    train_test_all = train_test_all[~train_test_all.unique_id.isin(parameters.IDS_TO_REMOVE)]

    return lmp_all, train_all, test_all, train_test_all


def fill_missing(series):
    for i in range(len(series)):
        transformer = MissingValuesFiller()
        series[i] = transformer.transform(series[i])


def get_series(lmp_all):
    all_series = TimeSeries.from_group_dataframe(
        lmp_all,
        group_cols=IDS,
        value_cols=Y,
        fill_missing_dates=True,
        freq='h',
    )

    fill_missing(all_series)
    return all_series


def get_futr_cov(all_df_pd):
    futr_cov = TimeSeries.from_group_dataframe(
        all_df_pd,
        group_cols=IDS,
        value_cols=FUTR_COLS,
        fill_missing_dates=True,
        freq='h',
    )
    fill_missing(futr_cov)
    return futr_cov


def get_past_cov(all_df_pd):
    past_cov = TimeSeries.from_group_dataframe(
        all_df_pd,
        group_cols=IDS,
        value_cols=PAST_COLS,
        fill_missing_dates=True,
        freq='h',
    )
    fill_missing(past_cov)
    return past_cov

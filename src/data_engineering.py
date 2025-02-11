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
import ibis
import ibis.selectors as s
from ibis import _
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
    f'{home}/Documents/spp_weis_price_forecast/src',
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
    'temperature',
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
) -> ibis.duckdb.connect:
    
    # client for getting parquets
    s3 = boto3.client('s3')

    os.makedirs('data', exist_ok=True)
    # create file paths for data
    file_paths = [f'data/{ds}.parquet' for ds in datasets]
    
    for fp in file_paths:
        log.info(f'getting: {fp} from s3')
        s3.download_file(Bucket='spp-weis', Key=fp, Filename=fp)

    log.info(f'os.listdir(data): {os.listdir("data")}')

    con = ibis.duckdb.connect()

    for i, ds in enumerate(datasets):
        log.info(f'loading: {ds}')
        log.info(f'file_paths[i]: {file_paths[i]}')
        con.read_parquet(file_paths[i], ds)

    return con


#############################################
# data prep
#############################################
def prep_lmp(
    con: ibis.duckdb.connect,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    loc_filter: str = 'PSCO_',
    clip_outliers: bool = False,
):
    # con = ibis.duckdb.connect("data/spp.ddb", read_only=True)
    lmp = con.table('lmp')
    lmp = lmp.filter(_.Settlement_Location_Name.contains(loc_filter))
    drop_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name',
        'MLC', 'MCC', 'MEC'
    ]

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.utcnow() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    lmp = lmp.filter(_.timestamp_mst_HE >= start_time)

    
    if clip_outliers:
        clipped_lwr = lmp.LMP.quantile(0.0025)
        clipped_upr = lmp.LMP.quantile(0.9975)
        lmp = (
            lmp
            .mutate(LMP = ibis.ifelse(_.LMP < clipped_upr, _.LMP, clipped_upr))
            .mutate(LMP = ibis.ifelse(_.LMP > clipped_lwr, _.LMP, clipped_lwr))
        )

    if end_time:
        lmp = lmp.filter(_.timestamp_mst_HE <= end_time)

    lmp = (
        lmp
        .mutate(unique_id=_.Settlement_Location_Name)
        .filter(~_.unique_id.contains("_ARPA")) # is missing?
        .drop_null(['unique_id'])
        .mutate(timestamp_mst=_.timestamp_mst_HE)
        .mutate(LMP=_.LMP.cast(parameters.PRECISION))
        .drop(drop_cols)
        .group_by(['unique_id'])
        .order_by(['unique_id', 'timestamp_mst'])
        .mutate(lmp_diff = (_.LMP - _.LMP.lag(1)).cast(parameters.PRECISION))
    )

    return lmp


def prep_mtrf(
        con: ibis.duckdb.connect,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
):
    # con = ibis.duckdb.connect("data/spp.ddb", read_only=True)
    mtrf = con.table('mtrf')
    drop_cols = ['Interval', 'GMTIntervalEnd']

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.utcnow() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    mtrf = mtrf.filter(_.timestamp_mst >= start_time)

    if end_time:
        mtrf = mtrf.filter(_.timestamp_mst <= end_time)

    mtrf = (
        mtrf
        .mutate(Wind_Forecast_MW=_.Wind_Forecast_MW.cast(parameters.PRECISION))
        .mutate(Solar_Forecast_MW=_.Solar_Forecast_MW.cast(parameters.PRECISION))
        .drop(drop_cols)
        .order_by(['timestamp_mst'])
    )

    return mtrf


def prep_mtlf(
        con: ibis.duckdb.connect,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
):
    # con = ibis.duckdb.connect("data/spp.ddb", read_only=True)
    mtlf = con.table('mtlf')
    drop_cols = ['Interval', 'GMTIntervalEnd', ]

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.utcnow() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    mtlf = mtlf.filter(_.timestamp_mst >= start_time)

    if end_time:
        mtlf = mtlf.filter(_.timestamp_mst <= end_time)

    mtlf = (
        mtlf
        .mutate(MTLF=_.MTLF.cast(parameters.PRECISION))
        .mutate(Averaged_Actual=_.Averaged_Actual.cast(parameters.PRECISION))
        .drop(drop_cols)
        .order_by(['timestamp_mst'])
    )

    return mtlf


def prep_gen_cap(
        con: ibis.duckdb.connect,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
):
    # con = ibis.duckdb.connect("data/spp.ddb", read_only=True)
    gen_cap = con.table('gen_cap')
    drop_cols = ['GMTIntervalEnd', ]

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.utcnow() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    gen_cap = gen_cap.filter(_.timestamp_mst >= start_time)

    if end_time:
        gen_cap = gen_cap.filter(_.timestamp_mst <= end_time)

    gen_cap = (
        gen_cap
        .mutate(Coal=(_.Coal_Market + _.Coal_Self).cast(parameters.PRECISION))
        .mutate(Hydro=_.Hydro.cast(parameters.PRECISION))
        .mutate(Natural_Gas=_.Natural_Gas.cast(parameters.PRECISION))
        .mutate(Nuclear=_.Nuclear.cast(parameters.PRECISION))
        .mutate(Solar=_.Solar.cast(parameters.PRECISION))
        .mutate(Wind=_.Wind.cast(parameters.PRECISION))
        .drop(drop_cols + ['Coal_Market', 'Coal_Self'])
        .order_by(['timestamp_mst'])
    )

    return gen_cap


def prep_weather(
        con: ibis.duckdb.connect,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
):
    # con = ibis.duckdb.connect("data/spp.ddb", read_only=True)
    weather = con.table('weather')
    drop_cols = ['timestamp', ]

    if not start_time:
        # get last 1.5 years
        start_time = pd.Timestamp.utcnow() - pd.Timedelta(parameters.TRAIN_START)

    # TODO: handle checks for start_time < end_time
    weather = weather.filter(_.timestamp_mst >= start_time)

    if end_time:
        weather = weather.filter(_.timestamp_mst <= end_time)

    weather = (
        weather
        .mutate(temperature=_.temperature.cast(parameters.PRECISION))
        .drop(drop_cols)
        .order_by(['timestamp_mst'])
    )

    return weather


def prep_all_df(
    con: ibis.duckdb.connect,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    clip_outliers: bool = False,
):
    lmp = prep_lmp(con, start_time=start_time, end_time=end_time, clip_outliers=clip_outliers)
    mtlf = prep_mtlf(con, start_time=start_time, end_time=end_time)
    mtrf = prep_mtrf(con, start_time=start_time, end_time=end_time)
    # gen_cap = prep_gen_cap(con, start_time=start_time, end_time=end_time)
    weather = prep_weather(con, start_time=start_time, end_time=end_time)
    

    # join into single dataset
    all_df = (
        mtlf
        .left_join(mtrf, 'timestamp_mst')
        .select(~s.contains("_right"))  # remove 'dt_right'
        .left_join(weather, 'timestamp_mst')
        .select(~s.contains("_right"))  # remove 'dt_right'
    )

    uid_df = ibis.memtable({'unique_id': lmp.unique_id.to_pandas().unique()})
    ids_df = all_df[['timestamp_mst']].cross_join(uid_df)

    all_df = (
        all_df.left_join(ids_df, 'timestamp_mst')
        .select(~s.contains("_right"))
        .left_join(lmp, ['unique_id', 'timestamp_mst'])
        .select(~s.contains("_right"))
        .filter(_.timestamp_mst >= '2023-05-15')  # some bad data early on...
        .order_by(['unique_id', 'timestamp_mst'])
    )

    # engineer features
    all_df = (
        all_df
        .drop_null(['unique_id'])
        .group_by(['unique_id'])
        .order_by(['unique_id', 'timestamp_mst'])
        .mutate(lmp_diff = (_.LMP - _.LMP.lag(1)).cast(parameters.PRECISION))
        .mutate(re_ratio = ((_.Wind_Forecast_MW + _.Solar_Forecast_MW) / _.MTLF).cast(parameters.PRECISION))
        .mutate(re_diff = (_.re_ratio - _.re_ratio.lag(1)).cast(parameters.PRECISION))
        .mutate(mtlf_diff = (_.MTLF - _.MTLF.lag(1)).cast(parameters.PRECISION))
        .mutate(wind_diff = (_.Wind_Forecast_MW - _.Wind_Forecast_MW.lag(1)).cast(parameters.PRECISION))
        .mutate(solar_diff = (_.Solar_Forecast_MW - _.Solar_Forecast_MW.lag(1)).cast(parameters.PRECISION))
        .mutate(load_net_re = (_.MTLF - _.Wind_Forecast_MW - _.Solar_Forecast_MW).cast(parameters.PRECISION))
        .mutate(load_net_re = (ibis.ifelse(_.load_net_re.abs() < 1.0, 1.0, _.load_net_re)).cast(parameters.PRECISION)) # avoid div/0 errors
        .mutate(load_net_re_diff = (_.load_net_re - _.load_net_re.lag(1)).cast(parameters.PRECISION))
        .mutate(lmp_load_net_re = (_.LMP / _.load_net_re).cast(parameters.PRECISION))
    )


    for i in [2,3,4,5,6]:
        win = ibis.window(preceding=i, following=0, group_by=all_df.unique_id, order_by=all_df.timestamp_mst)
        all_df = (
            all_df
            .mutate(all_df.lmp_diff.sum().over(win).cast(parameters.PRECISION).name(f'lmp_diff_rolling_{i}'))
            .mutate(all_df.load_net_re_diff.sum().over(win).cast(parameters.PRECISION).name(f'load_net_re_diff_rolling_{i}'))
        )
    

    return all_df


def all_df_to_pandas(all_df):
    all_df_pd = all_df.to_pandas()
    all_df_pd.set_index('timestamp_mst', inplace=True)
    all_df_pd = all_df_pd[IDS + Y + PAST_COLS + FUTR_COLS]
    return all_df_pd


def get_train_test_all(
    con: ibis.duckdb.connect,
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

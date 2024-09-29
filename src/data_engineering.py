import os
import shutil
import pickle
import random
import sys
import numpy as np
import pandas as pd
import duckdb
from typing import List

import requests
from io import StringIO

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

from src import params


#############################################
# parameters for column names
#############################################
FUTR_COLS = ['MTLF', 'Wind_Forecast_MW', 'Solar_Forecast_MW', 're_ratio', 're_diff']
PAST_COLS = ['Averaged_Actual']
Y = ['LMP']
IDS = ['unique_id']


#############################################
# data prep
#############################################
def prep_lmp(loc_filter: str='PSCO_'):
    con = ibis.duckdb.connect("data/spp.ddb")
    lmp = con.table('lmp')
    lmp = lmp.filter(_.Settlement_Location_Name.contains(loc_filter))
    drop_cols = [
        'Interval_HE', 'GMTIntervalEnd_HE', 'timestamp_mst_HE',
        'Settlement_Location_Name', 'PNODE_Name', 
        'MLC', 'MCC', 'MEC'
    ]
    
    lmp = (
        lmp
        .mutate(unique_id = _.Settlement_Location_Name )
        .mutate(timestamp_mst = _.timestamp_mst_HE)
        # .mutate(y = _.LMP) 
        .drop(drop_cols) 
        .order_by(['unique_id', 'timestamp_mst'])
    )

    return lmp


def prep_mtrf():
    con = ibis.duckdb.connect("data/spp.ddb")
    mtrf = con.table('mtrf')
    drop_cols = ['Interval', 'GMTIntervalEnd']

    mtrf = (
        mtrf
        # .mutate(ds = _.timestamp_mst)
        .drop(drop_cols) 
        .order_by(['timestamp_mst'])
    )
    
    return mtrf


def prep_mtlf():
    con = ibis.duckdb.connect("data/spp.ddb")
    mtlf = con.table('mtlf')
    drop_cols = ['Interval', 'GMTIntervalEnd',]

    mtlf = (
        mtlf
        # .mutate(ds = _.timestamp_mst)
        .drop(drop_cols) 
        .order_by(['timestamp_mst'])
    )
    
    return mtlf


def prep_all_df():
    lmp = prep_lmp()
    mtlf = prep_mtlf()
    mtrf = prep_mtrf()

    # join into single dataset
    all_df = (
        mtlf
        .left_join(mtrf, 'timestamp_mst')
        .select(~s.contains("_right")) # remove 'dt_right'
        .left_join(lmp, 'timestamp_mst')
        .select(~s.contains("_right")) # remove 'dt_right'
        .order_by(['unique_id', 'timestamp_mst'])
    )

    # engineer features
    all_df = (
        all_df
        .drop_null(['unique_id'])
        .mutate(re_ratio = (_.Wind_Forecast_MW + _.Solar_Forecast_MW) / _.MTLF)
        .mutate(re_diff = _.re_ratio - _.re_ratio.lag(1))
    )

    return all_df


def all_df_to_pandas(all_df):
    all_df_pd = all_df.to_pandas()
    all_df_pd.set_index('timestamp_mst', inplace=True)
    all_df_pd = all_df_pd[IDS + Y + PAST_COLS + FUTR_COLS]
    return all_df_pd



def get_train_test_all(all_df_pd):
    train_start = all_df_pd.index.min() + pd.Timedelta(f'{2*params.INPUT_CHUNK_LENGTH}h')
    test_end = all_df_pd.index.max() - pd.Timedelta(f'{2*params.FORECAST_HORIZON}h')
    tr_tst_split =  test_end - pd.Timedelta(f'{2*params.INPUT_CHUNK_LENGTH}h')
    log.info(f'train_start: {train_start}')
    log.info(f'tr_tst_split: {tr_tst_split}')
    log.info(f'test_end: {test_end}')

    train_idx = (all_df_pd.index < tr_tst_split) & (all_df_pd.index > train_start)
    test_idx = (all_df_pd.index > tr_tst_split) & (all_df_pd.index < test_end)
    
    train_all = all_df_pd[train_idx]
    test_all = all_df_pd[test_idx]

    return train_all, test_all



def fill_missing(series):
    for i in range(len(series)):
        transformer = MissingValuesFiller()
        series[i] = transformer.transform(series[i])


def get_all_series(all_df_pd):
    all_series = TimeSeries.from_group_dataframe(
        all_df_pd,
        group_cols=IDS,
        value_cols=Y,
        fill_missing_dates=True,
        freq='h',
    )
    
    fill_missing(all_series) 
    return all_series


def get_train_series(train_all):
    train_series = TimeSeries.from_group_dataframe(
        train_all,
        group_cols=IDS,
        value_cols=Y,
        fill_missing_dates=True,
        freq='h',
    )
    fill_missing(train_series)
    return train_series


def get_test_series(test_all):
    test_series = TimeSeries.from_group_dataframe(
        test_all,
        group_cols=IDS,
        value_cols=Y,
        fill_missing_dates=True,
        freq='h',
    )
    fill_missing(test_series)
    return test_series


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

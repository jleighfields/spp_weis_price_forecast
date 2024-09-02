import os
import sys
import random
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape, smape, mae, ope, rmse
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression, GumbelLikelihood, GaussianLikelihood

from darts import TimeSeries
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    TFTModel,
    LinearRegressionModel,
    LightGBMModel,
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    VARIMA,
)


from torchmetrics import MeanAbsolutePercentageError
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

import logging

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


########################################
# Feature engineering
########################################

########################################
# fill missing values
########################################
def fill_missed_values(df):
    """
    """
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df


########################################
# lmp
########################################
def get_psco_price_df(price_df):
    """
    """
    psco_idx = price_df["PNODE_Name"].str.contains("PSCO", case=False)
    psco_price_df_long = price_df[psco_idx]
    psco_price_df = psco_price_df_long.pivot_table(
                            index='GMTIntervalEnd',
                            columns='PNODE_Name',
                            values='LMP_HOURLY',
                            margins=False,
                        ).reset_index()

    psco_price_df.columns.name=None

    # get list of node names
    ls_nodes_name = list(psco_price_df.columns[1:])
    # separate load area names
    la_names = [n for n in ls_nodes_name if '_LA' in n]
    not_la_names = [n for n in ls_nodes_name if '_LA' not in n]
    random.Random(42).shuffle(not_la_names)
    # put load area in the front, make sure we train on these
    ls_nodes_name = la_names + not_la_names

    # return psco_price_df_long to create an overall scaler for psco lmps
    return fill_missed_values(psco_price_df), ls_nodes_name, psco_price_df_long


def create_psco_price_series(psco_price_df, node_name_ls):
    """
    """
    psco_price_series = TimeSeries.from_dataframe(
            psco_price_df, 
            time_col='GMTIntervalEnd', 
            value_cols=node_name_ls, 
            fill_missing_dates=True, 
            freq='H', 
            fillna_value=0, 
            static_covariates=None, 
            hierarchy=None
        ).astype(np.float32)

    return psco_price_series


########################################
# mtlf
########################################
def remove_duplicates(df):
    """
    Removes duplicates in mtlf and mtrf dataframes.
    """
    dups = df.GMTIntervalEnd.duplicated()
    df = df[~dups]
    return df


def create_mtlf_series(mtlf_df):
    """
    """
    # remove duplicates
    mtlf_df = remove_duplicates(mtlf_df)
    # fill missing values
    mtlf_df = fill_missed_values(mtlf_df)
    # create series
    mtlf_series = TimeSeries.from_dataframe(mtlf_df, time_col='GMTIntervalEnd', value_cols='MTLF', fill_missing_dates=True, freq='H', fillna_value=0, static_covariates=None, hierarchy=None).astype(np.float32)
    avg_act_series = TimeSeries.from_dataframe(mtlf_df, time_col='GMTIntervalEnd', value_cols='Averaged_Actual', fill_missing_dates=True, freq='H', fillna_value=0, static_covariates=None, hierarchy=None).astype(np.float32)
    
    return mtlf_series, avg_act_series


def create_mtlf_lmp_series(mtlf_df, psco_lmp_df, list_nodes_name):
    """
    """
    lmp_cols = ['GMTIntervalEnd'] + list_nodes_name
    mtlf_df = mtlf_df.merge(psco_lmp_df[lmp_cols], on='GMTIntervalEnd', how='left')
    print(f'mtlf_df.columns: {mtlf_df.columns}')
    mtlf_df = fill_missed_values(mtlf_df)
    mtlf_series = TimeSeries.from_dataframe(mtlf_df, time_col='GMTIntervalEnd', value_cols='MTLF', fill_missing_dates=True, freq='H', fillna_value=0, static_covariates=None, hierarchy=None).astype(np.float32)
    avg_act_cols = ['Averaged_Actual'] + list_nodes_name
    avg_act_series = TimeSeries.from_dataframe(mtlf_df, time_col='GMTIntervalEnd', value_cols=avg_act_cols, fill_missing_dates=True, freq='H', fillna_value=0, static_covariates=None, hierarchy=None).astype(np.float32)
    
    return mtlf_series, avg_act_series
          

########################################
# mtrf
########################################  
# Add renewable/load ratio feature to mtrf dataframe
def add_enrgy_ratio_to_mtrf(mtlf_df, mtrf_df):
    """
    """
    # remove duplicates
    mtlf_df = remove_duplicates(mtlf_df)
    mtrf_df = remove_duplicates(mtrf_df)

    mtrf_df = mtrf_df[['GMTIntervalEnd', 'Wind_Forecast_MW', 'Solar_Forecast_MW']].set_index('GMTIntervalEnd').asfreq('H').sort_index()
    mtrf_df = mtrf_df.join(mtlf_df[['GMTIntervalEnd','MTLF']].set_index('GMTIntervalEnd').asfreq('H').sort_index(), on='GMTIntervalEnd', how='outer').sort_values('GMTIntervalEnd').reset_index(drop=True)
    mtrf_df['Ratio'] = (mtrf_df['Wind_Forecast_MW'] + mtrf_df['Solar_Forecast_MW']) / mtrf_df['MTLF']
    mtrf_df.drop('MTLF', axis=1, inplace=True) 

    return fill_missed_values(mtrf_df)

def add_enrgy_ratio_diff_to_mtrf(mtrf_df):
    """
    """
    from darts.dataprocessing.transformers import Diff
    # remove duplicates
    mtrf_df = remove_duplicates(mtrf_df)

    mtrf_df['Ratio_diff'] = mtrf_df['Ratio'] - mtrf_df['Ratio'].shift(1)
    return fill_missed_values(mtrf_df)

def create_mtrf_series(mtrf_ratio_df):
    """
    """
    mtrf_series= TimeSeries.from_dataframe(mtrf_ratio_df, time_col='GMTIntervalEnd', value_cols=['Wind_Forecast_MW','Solar_Forecast_MW', 'Ratio'], fill_missing_dates=True, freq='H', fillna_value=0, static_covariates=None, hierarchy=None).astype(np.float32)
    
    return mtrf_series      


########################################
# Preprocess series
########################################

def scale_series(series_train, series_val, series_all, global_fit=False):
    """
    """
    # use global fit to do a single scaling
    # this will allow us to do global forecasting
    # for lmps
    if global_fit:
        scaler = series_train.pd_dataframe().abs().max().mean()
    else:
        scaler = series_train.pd_dataframe().abs().max()

    series_train_transformed = TimeSeries.from_dataframe(series_train.pd_dataframe()/scaler)
    series_val_transformed = TimeSeries.from_dataframe(series_val.pd_dataframe()/scaler)
    series_transformed = TimeSeries.from_dataframe(series_all.pd_dataframe()/scaler)

    return [series_train_transformed, series_val_transformed, series_transformed, scaler]

def lmp_series_drop_horizon(lmp_series, start_time, forecast_horizon):
    """
    """
    start_time_lmp = start_time
    end_time_lmp = lmp_series.end_time() - pd.Timedelta(f'{forecast_horizon+1}H')
    lmp_series = lmp_series.drop_before(start_time_lmp)
    # lmp_series = lmp_series.drop_after(end_time_lmp)
    return lmp_series

def lmp_series_start_time(lmp_series, start_time):
    """
    """
    lmp_series = lmp_series.drop_before(start_time)
    return lmp_series

def get_train_cutoff(lmp_series, forecast_horizon):
    """
    """
    #keep last 30 days (720 data points + forecast_horizon) of target series for validation. 
    training_cutoff = lmp_series[:-(720 + forecast_horizon)].end_time()
    return training_cutoff

def get_lmp_train_test_series(lmp_series_drop_horizon, training_cutoff, forecast_horizon, input_chunk_length):
    """
    """
    lmp_series_train = lmp_series_drop_horizon.drop_after(training_cutoff)
    lmp_series_val = lmp_series_drop_horizon.drop_before(training_cutoff)
    lmp_series_all = lmp_series_drop_horizon

    return [lmp_series_train, lmp_series_val, lmp_series_all]


############## mtlf #####################
def get_mtlf_train_test_series(mtlf_series, start_time, training_cutoff, forecast_horizon, input_chunk_length):
    """
    """
    # drop times before the starting time
    mtlf_series = mtlf_series.drop_before(start_time)

    # Split
    mtlf_series_train = mtlf_series.drop_after(training_cutoff + pd.Timedelta(f'{forecast_horizon+1}H')) # for future covariates
    mtlf_series_val = mtlf_series.drop_before(training_cutoff - pd.Timedelta(f'{input_chunk_length+1}H'))

    return [mtlf_series_train, mtlf_series_val, mtlf_series]


# def scale_mtlf_series(mtlf_series_train, mtlf_series_val, mtlf_series):
#     """
#     """
#     transformer_mtlf = Scaler()
#     mtlf_series_train_transformed = transformer_mtlf.fit_transform(mtlf_series_train)
#     mtlf_series_val_transformed = transformer_mtlf.transform(mtlf_series_val)
#     mtlf_series_transformed = transformer_mtlf.transform(mtlf_series)

#     return [mtlf_series_train_transformed, mtlf_series_val_transformed, mtlf_series_transformed]

############# avg_actual #################
def get_avg_act_train_test_series(avg_act_series, start_time, training_cutoff):
    """
    """
    avg_act_series = avg_act_series.drop_before(start_time)

    # Split
    avg_act_series_train = avg_act_series.drop_after(training_cutoff)
    avg_act_series_val = avg_act_series.drop_before(training_cutoff)

    return [avg_act_series_train, avg_act_series_val, avg_act_series]


############## mtrf #####################
def get_mtrf_train_test_series(mtrf_series, start_time, training_cutoff, forecast_horizon, input_chunk_length):
    """
    """
    mtrf_series = mtrf_series.drop_before(start_time)

    # Split
    mtrf_series_train = mtrf_series.drop_after(training_cutoff + pd.Timedelta(f'{forecast_horizon+1}H')) # for future covariates
    mtrf_series_val = mtrf_series.drop_before(training_cutoff - pd.Timedelta(f'{input_chunk_length+1}H'))

    return [mtrf_series_train, mtrf_series_val, mtrf_series]




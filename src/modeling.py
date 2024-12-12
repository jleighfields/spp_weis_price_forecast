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
ibis.options.interactive = True

from sklearn.preprocessing import RobustScaler

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
    Mapper,
    InvertibleMapper,
)
from darts.dataprocessing import Pipeline
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
    TiDEModel,
    DLinearModel,
    NLinearModel,
    TSMixerModel
)


from torchmetrics import (
    SymmetricMeanAbsolutePercentageError, 
    MeanAbsoluteError, 
    MeanSquaredError,
)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
]
for module_path in module_paths:
    if os.path.isdir(module_path):
        log.info('adding module path')
        sys.path.insert(0, module_path)

import parameters


import pprint
# set up pretty printer
pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)


def log_pretty(obj):
    pretty_out = f"{pp.pformat(obj)}"

    return f'{pretty_out}\n'
    

def build_fit_tsmixerx(
    series: List[TimeSeries],
    val_series: List[TimeSeries],
    future_covariates: List[TimeSeries],
    past_covariates: List[TimeSeries],
    hidden_size: int=10,
    ff_size: int=27,
    num_blocks: int=3,
    forecast_horizon: int=parameters.FORECAST_HORIZON,
    input_chunk_length: int=parameters.INPUT_CHUNK_LENGTH,
    lr: float=8e-4,
    batch_size: int=64,
    n_epochs: int=8,
    dropout: float=0.45,
    activation: str='ELU', #  “ReLU”, “RReLU”, “PReLU”, “ELU”, “Softplus”, “Tanh”, “SELU”, “LeakyReLU”, “Sigmoid”, “GELU”.
    force_reset: bool=True, # reset model if already exists
    callbacks=None,
    model_id: str='ts_mixer',
):

    MODEL_TYPE = "ts_mixer_model"
    work_dir = os.getcwd() + f'/model_checkpoints/{MODEL_TYPE}'
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...
    
    encoders = {
        # "datetime_attribute": {
        #     "future": ["month"], # "dayofweek", "month", "hour"
        #     "past": ["month"], # "dayofweek", "month", "hour"
        # },
        "position": {
            "past": ["relative"], 
            "future": ["relative"]
        },
        "transformer": Scaler(RobustScaler(), global_fit=True)
    }

    # common parameters across models
    model_params = {
        'hidden_size': hidden_size,
        'ff_size': ff_size,
        'num_blocks': num_blocks,
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': forecast_horizon,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'dropout': dropout,
        'activation': activation,
        'add_encoders': encoders,
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': model_id, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': True,
    }

    # throughout training we'll monitor the validation loss for early stopping
    # early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
    # if callbacks is None:
    #     callbacks = [early_stopper]
    # else:
    #     callbacks = [early_stopper] + callbacks

    # pl_trainer_kwargs = {"callbacks": callbacks}
    # model_params['pl_trainer_kwargs'] = pl_trainer_kwargs
    log.info(f'model_params: \n{log_pretty(model_params)}')
    
    model = TSMixerModel(**model_params)

    # train the model
    fit_params = {
        'series': series,
        'val_series': val_series,
        'future_covariates': future_covariates,
        'past_covariates': past_covariates,
        'val_future_covariates': future_covariates,
        'val_past_covariates': past_covariates,
    }
    model.fit(**fit_params)

    # reload best model over course of training
    # model = TSMixerModel.load_from_checkpoint(
    #     work_dir=work_dir,
    #     model_name=MODEL_TYPE,
    #     best=False,
    # )
    
    model.MODEL_TYPE = MODEL_TYPE
    model.TRAIN_TIMESTAMP = pd.Timestamp.utcnow()

    return model


def build_fit_tide(
    series: List[TimeSeries],
    val_series: List[TimeSeries],
    future_covariates: List[TimeSeries],
    past_covariates: List[TimeSeries],
    num_encoder_layers: int=1,
    num_decoder_layers: int=1,
    decoder_output_dim: int=17,
    hidden_size: int=32,
    temporal_width_past: int=1,
    temporal_width_future: int=0,
    temporal_decoder_hidden: int=20,
    temporal_hidden_size_past: int=4,
    temporal_hidden_size_future: int=24,
    forecast_horizon: int=parameters.FORECAST_HORIZON,
    input_chunk_length: int=parameters.INPUT_CHUNK_LENGTH,
    lr: float=3e-4,
    batch_size: int=64,
    n_epochs: int=10,
    dropout: float=0.43,
    force_reset: bool=True, # reset model if already exists
    callbacks=None,
    model_id: str='tide',
):

    MODEL_TYPE = "tide_model"
    work_dir = os.getcwd() + f'/model_checkpoints/{MODEL_TYPE}'
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...
    
    encoders = {
        # "datetime_attribute": {
        #     "future": ["month"], # "dayofweek", "month", "hour"
        #     "past": ["month"], # "dayofweek", "month", "hour"
        # },
        "position": {
            "past": ["relative"], 
            "future": ["relative"]
        },
        "transformer": Scaler(RobustScaler(), global_fit=True)
    }

    # common parameters across models
    model_params = {
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers,
        'decoder_output_dim': decoder_output_dim,
        'hidden_size': hidden_size,
        'temporal_width_past': temporal_width_past,
        'temporal_width_future': temporal_width_future,
        'temporal_decoder_hidden': temporal_decoder_hidden,
        'temporal_hidden_size_past': temporal_hidden_size_past,
        'temporal_hidden_size_future': temporal_hidden_size_future,
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': forecast_horizon,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'dropout': dropout,
        'add_encoders': encoders,
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': model_id, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': True,
    }

    # throughout training we'll monitor the validation loss for early stopping
    # early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
    # if callbacks is None:
    #     callbacks = [early_stopper]
    # else:
    #     callbacks = [early_stopper] + callbacks

    # pl_trainer_kwargs = {"callbacks": callbacks}
    # model_params['pl_trainer_kwargs'] = pl_trainer_kwargs
    log.info(f'model_params: \n{log_pretty(model_params)}')
    
    model = TiDEModel(**model_params)

    # train the model
    fit_params = {
        'series': series,
        'val_series': val_series,
        'future_covariates': future_covariates,
        'past_covariates': past_covariates,
        'val_future_covariates': future_covariates,
        'val_past_covariates': past_covariates,
    }
    model.fit(**fit_params)

    # reload best model over course of training
    # model = TiDEModel.load_from_checkpoint(
    #     work_dir=work_dir,
    #     model_name=MODEL_TYPE,
    #     best=False,
    # )
    
    model.MODEL_TYPE = MODEL_TYPE
    model.TRAIN_TIMESTAMP = pd.Timestamp.utcnow()

    return model


def build_fit_tft(
    series: List[TimeSeries],
    val_series: List[TimeSeries],
    future_covariates: List[TimeSeries],
    past_covariates: List[TimeSeries],
    hidden_size: int=9, # Hidden state size of the TFT. It is the main hyper-parameter and common across the internal TFT architecture.
    lstm_layers: int=4, # Number of layers for the Long Short Term Memory (LSTM) Encoder and Decoder (1 is a good default).
    num_attention_heads: int=1, # Number of attention heads (4 is a good default)
    dropout: float=0.49,
    full_attention: bool=True,
    forecast_horizon: int=parameters.FORECAST_HORIZON,
    input_chunk_length: int=parameters.INPUT_CHUNK_LENGTH,
    lr: float=1.0e-3,
    batch_size: int=64,
    n_epochs: int=3,
    force_reset: bool=True, # reset model if already exists
    callbacks=None,
    model_id: str='tft',
):
    MODEL_TYPE = "tft_model"
    work_dir = os.getcwd() + f'/model_checkpoints/{MODEL_TYPE}'
    
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...
    
    encoders = {
        # "datetime_attribute": {
        #     "future": ["month"], # "dayofweek", "month", "hour"
        #     "past": ["month"], # "dayofweek", "month", "hour"
        # },
        "position": {
            "past": ["relative"], 
            "future": ["relative"]
        },
        "transformer": Scaler(RobustScaler(), global_fit=True)
    }

    # common parameters across models
    model_params = {
        'hidden_size': hidden_size,
        'lstm_layers': lstm_layers,
        'num_attention_heads': num_attention_heads,
        'dropout': dropout,
        'full_attention': full_attention,
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': forecast_horizon,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'add_encoders': encoders,
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': model_id, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': True,
    }
    

    # throughout training we'll monitor the validation loss for early stopping
    # early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
    # if callbacks is None:
    #     callbacks = [early_stopper]
    # else:
    #     callbacks = [early_stopper] + callbacks

    # pl_trainer_kwargs = {"callbacks": callbacks}
    # model_params['pl_trainer_kwargs'] = pl_trainer_kwargs
    log.info(f'model_params: \n{log_pretty(model_params)}')

    model = TFTModel(**model_params)

    # train the model
    fit_params = {
        'series': series,
        'val_series': val_series,
        'future_covariates': future_covariates,
        'past_covariates': past_covariates,
        'val_future_covariates': future_covariates,
        'val_past_covariates': past_covariates,
    }
    model.fit(**fit_params)

    # reload best model over course of training
    # model = TFTModel.load_from_checkpoint(
    #     work_dir=work_dir,
    #     model_name=MODEL_TYPE,
    #     best=False,
    # )
    
    model.MODEL_TYPE = MODEL_TYPE
    model.TRAIN_TIMESTAMP = pd.Timestamp.utcnow()

    return model


def build_fit_dlinear(
    series: List[TimeSeries],
    val_series: List[TimeSeries],
    future_covariates: List[TimeSeries],
    past_covariates: List[TimeSeries],
    kernel_size: int=25, # The size of the kernel for the moving average (default=25)
    forecast_horizon: int=parameters.FORECAST_HORIZON,
    input_chunk_length: int=parameters.INPUT_CHUNK_LENGTH,
    lr: float=1.0e-3,
    batch_size: int=64,
    n_epochs: int=3,
    force_reset: bool=True, # reset model if already exists
    callbacks=None,
):
    work_dir = os.getcwd() + '/model_checkpoints'
    MODEL_TYPE = "tft_model"
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...
    
    encoders = {
        # "datetime_attribute": {
        #     "future": ["month"], # "dayofweek", "month", "hour"
        #     "past": ["month"], # "dayofweek", "month", "hour"
        # },
        "position": {
            "past": ["relative"], 
            "future": ["relative"]
        },
        "transformer": Scaler(RobustScaler(), global_fit=True)
    }

    # common parameters across models
    model_params = {
        'kernel_size': kernel_size,
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': forecast_horizon,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'add_encoders': encoders,
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': MODEL_TYPE, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': True,
    }
    

    # throughout training we'll monitor the validation loss for early stopping
    # early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
    # if callbacks is None:
    #     callbacks = [early_stopper]
    # else:
    #     callbacks = [early_stopper] + callbacks

    # pl_trainer_kwargs = {"callbacks": callbacks}
    # model_params['pl_trainer_kwargs'] = pl_trainer_kwargs
    log.info(f'model_params: \n{log_pretty(model_params)}')

    model = DLinearModel(**model_params)

    # train the model
    fit_params = {
        'series': series,
        'val_series': val_series,
        'future_covariates': future_covariates,
        'past_covariates': past_covariates,
        'val_future_covariates': future_covariates,
        'val_past_covariates': past_covariates,
    }
    model.fit(**fit_params)

    # reload best model over course of training
    # model = TFTModel.load_from_checkpoint(
    #     work_dir=work_dir,
    #     model_name=MODEL_TYPE,
    #     best=False,
    # )
    
    model.MODEL_TYPE = MODEL_TYPE
    model.TRAIN_TIMESTAMP = pd.Timestamp.utcnow()

    return model


def build_fit_nlinear(
    series: List[TimeSeries],
    val_series: List[TimeSeries],
    future_covariates: List[TimeSeries],
    past_covariates: List[TimeSeries],
    # kernel_size: int=25, # The size of the kernel for the moving average (default=25)
    forecast_horizon: int=parameters.FORECAST_HORIZON,
    input_chunk_length: int=parameters.INPUT_CHUNK_LENGTH,
    lr: float=1.0e-3,
    batch_size: int=64,
    n_epochs: int=3,
    force_reset: bool=True, # reset model if already exists
    callbacks=None,
):
    work_dir = os.getcwd() + '/model_checkpoints'
    MODEL_TYPE = "tft_model"
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...
    
    encoders = {
        # "datetime_attribute": {
        #     "future": ["month"], # "dayofweek", "month", "hour"
        #     "past": ["month"], # "dayofweek", "month", "hour"
        # },
        "position": {
            "past": ["relative"], 
            "future": ["relative"]
        },
        "transformer": Scaler(RobustScaler(), global_fit=True)
    }

    # common parameters across models
    model_params = {
        # 'kernel_size': kernel_size,
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': forecast_horizon,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'add_encoders': encoders,
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': MODEL_TYPE, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': True,
    }
    

    # throughout training we'll monitor the validation loss for early stopping
    # early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
    # if callbacks is None:
    #     callbacks = [early_stopper]
    # else:
    #     callbacks = [early_stopper] + callbacks

    # pl_trainer_kwargs = {"callbacks": callbacks}
    # model_params['pl_trainer_kwargs'] = pl_trainer_kwargs
    log.info(f'model_params: \n{log_pretty(model_params)}')

    model = NLinearModel(**model_params)

    # train the model
    fit_params = {
        'series': series,
        'val_series': val_series,
        'future_covariates': future_covariates,
        'past_covariates': past_covariates,
        'val_future_covariates': future_covariates,
        'val_past_covariates': past_covariates,
    }
    model.fit(**fit_params)

    # reload best model over course of training
    # model = TFTModel.load_from_checkpoint(
    #     work_dir=work_dir,
    #     model_name=MODEL_TYPE,
    #     best=False,
    # )
    
    model.MODEL_TYPE = MODEL_TYPE
    model.TRAIN_TIMESTAMP = pd.Timestamp.utcnow()

    return model
    


def get_ci_err(actual_series, pred_series, n_jobs=1, verbose=False):
    ci_cover_err = []
    for i, pred in enumerate(pred_series):
        
        series_qs = pred.quantiles_df((0.1, 0.9))
        val_y = actual_series[i].pd_dataframe()
        
        eval_df = series_qs.merge(
            val_y,
            how='inner',
            left_index=True,
            right_index=True,
        )
    
    
        cover = (
            (eval_df['LMP_0.9'] > eval_df['LMP']) &
            (eval_df['LMP_0.1'] < eval_df['LMP'])
        ).mean() # should be about 80%

        ci_cover_err += [100 * np.abs(cover - 0.8)]

    return ci_cover_err
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TODO:
#    1. early stopping not working optuna experiment


# In[2]:


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

import torch

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

import mlflow

import warnings
warnings.filterwarnings("ignore")

# logging
import logging

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
# from optuna.visualization import (
#     plot_optimization_history,
#     plot_contour,
#     plot_param_importances,
#     plot_pareto_front,
# )


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

log.info(f'os.getcwd(): {os.getcwd()}')
log.info(f'os.listdir(): {os.listdir()}')

# from module path
import data_engineering as de
import params
from modeling import get_ci_err, build_fit_tsmixerx, log_pretty

## will be loaded from root when deployed
from darts_wrapper import DartsGlobalModel

log.info(f'FORECAST_HORIZON: {params.FORECAST_HORIZON}')
log.info(f'INPUT_CHUNK_LENGTH: {params.INPUT_CHUNK_LENGTH}')

# Data prep
log.info('preparing data')

lmp = de.prep_lmp()
all_df = de.prep_all_df()
all_df_pd = de.all_df_to_pandas(de.prep_all_df())
all_df_pd.info()

lmp_all, train_all, test_all = de.get_train_test_all()
all_series = de.get_all_series(lmp_all)
all_series[0].plot()

train_series = de.get_train_series(train_all)
test_series = de.get_test_series(test_all)

futr_cov = de.get_futr_cov(all_df_pd)
past_cov = de.get_past_cov(all_df_pd)

# MLFlow setup
log.info(f'mlflow.get_tracking_uri(): {mlflow.get_tracking_uri()}')
exp_name = 'spp_weis'

if mlflow.get_experiment_by_name(exp_name) is None:
    exp = mlflow.create_experiment(exp_name)
    
exp = mlflow.get_experiment_by_name(exp_name)

# Get model signature
node_series = train_series[0]
future_cov_series = futr_cov[0]
past_cov_series = past_cov[0]

data = {
    'series': [node_series.to_json()],
    'past_covariates': [past_cov_series.to_json()],
    'future_covariates': [future_cov_series.to_json()],
    'n': params.FORECAST_HORIZON,
    'num_samples': 200
}

df = pd.DataFrame(data)

ouput_example = 'the endpoint return json as a string'

from mlflow.models import infer_signature
darts_signature = infer_signature(df, ouput_example)

# Refit and log model with best params
log.info('refit model')
with mlflow.start_run(experiment_id=exp.experiment_id) as run:
    model = build_fit_tsmixerx(
        series=train_series,
        val_series=test_series,
        future_covariates=futr_cov,
        past_covariates=past_cov,
    )
    
    log.info(f'run.info: \n{run.info}')
    artifact_path = "model_artifacts"
    metrics = {}
    model_params = model.model_params
    
    # back test on validation data
    acc = model.backtest(
        series=test_series,
        # series=all_series,
        past_covariates=past_cov,
        future_covariates=futr_cov,
        retrain=False,
        forecast_horizon=model_params['output_chunk_length'],
        stride=25,
        metric=[mae, rmse, get_ci_err],
        verbose=False,
        num_samples=200,
    )

    log.info(f'BACKTEST: np.mean(acc, axis=0): {np.mean(acc, axis=0)}')
    acc_df = pd.DataFrame(
        np.mean(acc, axis=0).reshape(1,-1),
        columns=['mae', 'rmse', 'ci_error']
    )

    # add metrics
    metrics['test_mae'] = acc_df.mae[0]
    metrics['test_rmse'] = acc_df.rmse[0]
    metrics['test_ci_error'] = acc_df.ci_error[0]

    # final training
    final_train_series = test_series
    log.info('final training')
    model.fit(
            series=test_series,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            verbose=True,
            )
    
    # final model back test on validation data
    acc = model.backtest(
            series=test_series,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            retrain=False,
            forecast_horizon=model_params['output_chunk_length'],
            stride=25,
            metric=[mae, rmse, get_ci_err],
            verbose=False,
            num_samples=200,
        )

    log.info(f'FINAL ACC: np.mean(acc, axis=0): {np.mean(acc, axis=0)}')
    acc_df = pd.DataFrame(
        np.mean(acc, axis=0).reshape(1,-1),
        columns=['mae', 'rmse', 'ci_error']
    )

    # add and log metrics
    metrics['final_mae'] = acc_df.mae[0]
    metrics['final_rmse'] = acc_df.rmse[0]
    metrics['final_ci_error'] = acc_df.ci_error[0]
    mlflow.log_metrics(metrics)

    # set up path to save model
    model_path = '/'.join([artifact_path, model.MODEL_TYPE])

    shutil.rmtree(artifact_path, ignore_errors=True)
    os.makedirs(artifact_path)

    # log params
    mlflow.log_params(model_params)

    # save model files (model, model.ckpt) 
    # and load them to artifacts when logging the model
    model.save(model_path)

    # save MODEL_TYPE to artifacts
    # this will be used to load the model from the artifacts
    model_type_path = '/'.join([artifact_path, 'MODEL_TYPE.pkl'])
    with open(model_type_path, 'wb') as handle:
        pickle.dump(model.MODEL_TYPE, handle)
    
    # map model artifacts in dictionary
    artifacts = {
        'model': model_path,
        'model.ckpt': model_path+'.ckpt',
        'MODEL_TYPE': model_type_path,
    }
    
    # log model
    # https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html#pip-requirements-example
    mlflow.pyfunc.log_model(
        artifact_path='GlobalForecasting',
        code_path=['src/darts_wrapper.py'],
        signature=darts_signature,
        artifacts=artifacts,
        python_model=DartsGlobalModel(), 
        pip_requirements=["-r notebooks/model_training/requirements.txt"],
    )

log.info('loading model from mlflow for testing')
# test predictions on latest run
runs = mlflow.search_runs(
    experiment_ids = exp.experiment_id,
    # order_by=['metrics.test_mae']
    order_by=['end_time']
    )

runs.sort_values('end_time', ascending=False, inplace=True)
best_run_id = runs.run_id.iloc[0]
model_path = runs['artifact_uri'].iloc[0] + '/GlobalForecasting'
loaded_model = mlflow.pyfunc.load_model(model_path)

log.info('test getting predictions')
plot_ind = 3
plot_series = all_series[plot_ind]

plot_end_time = plot_series.end_time() - pd.Timedelta(f'{params.INPUT_CHUNK_LENGTH+1}h')
log.info(f'plot_end_time: {plot_end_time}')

plot_node_name = plot_series.static_covariates.unique_id.LMP
node_series = plot_series.drop_after(plot_end_time)
log.info(f'plot_end_time: {plot_end_time}')
log.info(f'node_series.end_time(): {node_series.end_time()}')
future_cov_series = futr_cov[0]
past_cov_series = past_cov[0]

data = {
        'series': [node_series.to_json()],
        'past_covariates': [past_cov_series.to_json()],
        'future_covariates': [future_cov_series.to_json()],
        'n': 5,
        'num_samples': 2
    }
df = pd.DataFrame(data)

df['num_samples'] = 2
pred = loaded_model.predict(df)

log.info(f'pred: {pred}')


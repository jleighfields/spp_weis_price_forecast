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
    TSMixerModel,
    NaiveEnsembleModel,
)


from torchmetrics import (
    SymmetricMeanAbsolutePercentageError, 
    MeanAbsoluteError, 
    MeanSquaredError,
)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import mlflow
from mlflow.models import infer_signature

import warnings
warnings.filterwarnings("ignore")

# logging
import logging

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


MODEL_NAME = 'spp_weis'

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
from modeling import get_ci_err, build_fit_tsmixerx, build_fit_tft, build_fit_tide, log_pretty

## will be loaded from root when deployed
from darts_wrapper import DartsGlobalModel

log.info(f'FORECAST_HORIZON: {params.FORECAST_HORIZON}')
log.info(f'INPUT_CHUNK_LENGTH: {params.INPUT_CHUNK_LENGTH}')


# connect to database and prepare data
print('\n' + '*'*40)
log.info('preparing data')
# con = ibis.duckdb.connect("/teamspace/studios/data-collection/spp_weis_price_forecast/data/spp.ddb", read_only=True)
con = ibis.duckdb.connect("data/spp.ddb", read_only=True)

lmp = de.prep_lmp(con)
lmp_df = lmp.to_pandas().rename(
    columns={
        'LMP': 'LMP_HOURLY',
        'unique_id':'node',
        'timestamp_mst':'time'
    }
)

all_df = de.prep_all_df(con)
all_df_pd = de.all_df_to_pandas(de.prep_all_df(con))
all_df_pd.info()

lmp_all, train_all, test_all, train_test_all = de.get_train_test_all(con)
con.disconnect()

all_series = de.get_series(lmp_all)
train_test_all_series = de.get_series(train_test_all)
train_series = de.get_series(train_all)
test_series = de.get_series(test_all)

futr_cov = de.get_futr_cov(all_df_pd)
past_cov = de.get_past_cov(all_df_pd)


# MLFlow setup
print('\n' + '*'*40)
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlruns.db'
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
darts_signature = infer_signature(df, ouput_example)


## build pretrained models   
model_tsmixer = build_fit_tsmixerx(
    series=train_test_all_series,
    val_series=test_series,
    future_covariates=futr_cov,
    past_covariates=past_cov,
)

model_tide = build_fit_tide(
    series=train_test_all_series,
    val_series=test_series,
    future_covariates=futr_cov,
    past_covariates=past_cov,
)


# Refit and log model with best params
log.info('log ensemble model')
with mlflow.start_run(experiment_id=exp.experiment_id) as run:

    MODEL_TYPE = 'naive_ens'
    
    # fit model with best params from study
    model = NaiveEnsembleModel(
        forecasting_models=[model_tsmixer, model_tide], 
        train_forecasting_models=False
    )

    model.MODEL_TYPE = MODEL_TYPE
    model.TRAIN_TIMESTAMP = pd.Timestamp.utcnow()
    
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
        forecast_horizon=params.FORECAST_HORIZON,
        stride=25,
        metric=[mae, rmse, get_ci_err],
        verbose=False,
        num_samples=200,
    )

    # log.info(f'BACKTEST: acc: {acc}')
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
    # final_train_series = test_series
    # log.info('final training')
    # model.fit(
    #         series=test_series,
    #         past_covariates=past_cov,
    #         future_covariates=futr_cov,
    #         verbose=True,
    #         )
    
    # final model back test on validation data
    acc = model.backtest(
            series=test_series,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            retrain=False,
            forecast_horizon=params.FORECAST_HORIZON,
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

    model_timestamp = '/'.join([artifact_path, 'TRAIN_TIMESTAMP.pkl'])
    with open(model_timestamp, 'wb') as handle:
        pickle.dump(model.TRAIN_TIMESTAMP, handle)
    
    # map model artififacts in dictionary
    # artifacts = {
    #     'model': model_path,
    #     'model.ckpt': model_path+'.ckpt',
    #     'MODEL_TYPE': model_type_path,
    #     'TRAIN_TIMESTAMP': model_timestamp,
    # }
    artifacts = {f:f'{artifact_path}/{f}' for f in os.listdir('model_artifacts')}
    artifacts['model'] = model_path
    
    # log model
    # https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html#pip-requirements-example
    mlflow.pyfunc.log_model(
        artifact_path='GlobalForecasting',
        code_path=['src/darts_wrapper.py'],
        signature=darts_signature,
        artifacts=artifacts,
        python_model=DartsGlobalModel(), 
        pip_requirements=["-r notebooks/model_training/requirements.txt"],
        registered_model_name=MODEL_NAME,
    )


# test predictions on latest run
print('\n' + '*'*40)
log.info('loading model from mlflow for testing')
# runs = mlflow.search_runs(
#     experiment_ids = exp.experiment_id,
#     # order_by=['metrics.test_mae']
#     order_by=['end_time']
#     )

# runs.sort_values('end_time', ascending=False, inplace=True)
# best_run_id = runs.run_id.iloc[0]
# model_path = runs['artifact_uri'].iloc[0] + '/GlobalForecasting'
# loaded_model = mlflow.pyfunc.load_model(model_path)
from mlflow import MlflowClient
client = MlflowClient()
def get_latest_registered_model_version(model_name = MODEL_NAME):
    filter_string = f"name='{model_name}'"
    results = client.search_registered_models(filter_string=filter_string)
    return results[0].latest_versions[0].version


client.set_registered_model_alias(MODEL_NAME, "champion", get_latest_registered_model_version())

# model uri for the above model
model_uri = "models:/spp_weis@champion"

# Load the model and access the custom metadata
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)


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

print('\n' + '*'*40)
log.info(f'pred: {pred}')

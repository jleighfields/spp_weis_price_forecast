import os
import shutil
import pickle
import sys
import numpy as np
import pandas as pd
import ibis
import boto3
import torch

ibis.options.interactive = True

from darts.metrics import mae, rmse
from darts.models import (
    TFTModel,
    TiDEModel,
    TSMixerModel,
    NaiveEnsembleModel,
)

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

import warnings

warnings.filterwarnings("ignore")

# logging
import logging

from dotenv import load_dotenv
load_dotenv()

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

log.info(f'os.getcwd(): {os.getcwd()}')
log.info(f'os.listdir(): {os.listdir()}')

# from module path
import data_engineering as de
import parameters
from modeling import get_ci_err, build_fit_tsmixerx, build_fit_tft, build_fit_tide, log_pretty

# will be loaded from root when deployed
from darts_wrapper import DartsGlobalModel

# client for uploading model weights
s3 = boto3.client('s3')

# check parameters
log.info(f'FORECAST_HORIZON: {parameters.FORECAST_HORIZON}')
log.info(f'INPUT_CHUNK_LENGTH: {parameters.INPUT_CHUNK_LENGTH}')
log.info(f'MODEL_NAME: {parameters.MODEL_NAME}')

# connect to database and prepare data
print('\n' + '*' * 40)
# con = ibis.duckdb.connect("data/spp.ddb", read_only=True)
con = ibis.duckdb.connect()
log.info('getting lmp data from s3')
con.read_parquet('s3://spp-weis/data/lmp.parquet', 'lmp')
log.info('getting mtrf data from s3')
con.read_parquet('s3://spp-weis/data/mtrf.parquet', 'mtrf')
log.info('getting mtlf data from s3')
con.read_parquet('s3://spp-weis/data/mtlf.parquet', 'mtlf')
log.info('getting weather data from s3')
con.read_parquet('s3://spp-weis/data/weather.parquet', 'weather')
log.info('finished getting data from s3')


log.info('preparing lmp data')
lmp = de.prep_lmp(con)
lmp_df = lmp.to_pandas().rename(
    columns={
        'LMP': 'LMP_HOURLY',
        'unique_id': 'node',
        'timestamp_mst': 'time'
    }
)


log.info('preparing covariate data')
# all_df = de.prep_all_df(con)
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

print('\n' + '*' * 40)


# build pretrained models
models_tsmixer = []
if parameters.USE_TSMIXER:
    for i, param in enumerate(parameters.TSMIXER_PARAMS[:parameters.TOP_N]):
        print(f'\ni: {i} \t' + '*' * 25, flush=True)
        model_tsmixer = build_fit_tsmixerx(
            series=train_test_all_series,
            val_series=test_series,
            future_covariates=futr_cov,
            past_covariates=past_cov,
            **param
        )
        models_tsmixer += [model_tsmixer]


models_tide = []
if parameters.USE_TIDE:
    for i, param in enumerate(parameters.TIDE_PARAMS[:parameters.TOP_N]):
        print(f'\ni: {i} \t' + '*' * 25, flush=True)
        model_tide = build_fit_tide(
            series=train_test_all_series,
            val_series=test_series,
            future_covariates=futr_cov,
            past_covariates=past_cov,
            **param
        )
        models_tide += [model_tide]

models_tft = []
if parameters.USE_TFT:
    for i, param in enumerate(parameters.TFT_PARAMS[:parameters.TOP_N]):
        print(f'\ni: {i} \t' + '*' * 25, flush=True)
        model_tft = build_fit_tft(
            series=train_test_all_series,
            val_series=test_series,
            future_covariates=futr_cov,
            past_covariates=past_cov,
            **param
        )
        models_tft += [model_tft]


# create directory to store artifacts
if os.path.isdir('saved_models'):
    shutil.rmtree('saved_models')
os.mkdir('saved_models')


model_timestamp = '/'.join(['saved_models', 'TRAIN_TIMESTAMP.pkl'])
with open(model_timestamp, 'wb') as handle:
    pickle.dump(pd.Timestamp.utcnow(), handle)

for i, m in enumerate(models_tide):
    m.save(f'saved_models/tide_{i}.pt')

for i, m in enumerate(models_tsmixer):
    m.save(f'saved_models/tsmixer_{i}.pt')

for i, m in enumerate(models_tft):
    m.save(f'saved_models/tft_{i}.pt')


# test models
forecasting_models = models_tide + models_tsmixer + models_tft
print('\n' + '*' * 40)
log.info('loading model from checkpoints')
loaded_model = NaiveEnsembleModel(
        forecasting_models=forecasting_models, 
        train_forecasting_models=False
    )

log.info('test getting predictions')
plot_ind = 3
plot_series = all_series[plot_ind]

plot_end_time = plot_series.end_time() - pd.Timedelta(f'{parameters.INPUT_CHUNK_LENGTH + 1}h')
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
pred=loaded_model.predict(
    series=node_series,
    past_covariates=past_cov_series,
    future_covariates=future_cov_series,
    n=5,
    num_samples=2,
)

print('\n' + '*' * 40)
log.info(f'pred: {pred}')


# upload models
ckpt_uploads = [f for f in os.listdir('saved_models') if '.pt' in f or '.ckpt' in f or 'TRAIN_TIMESTAMP.pkl' in f]
log.info(f'ckpt_uploads: {ckpt_uploads}')


# upload artifacts
for ckpt in ckpt_uploads:
    log.info(f'uploading: {ckpt}')
    s3.upload_file(f'saved_models/{ckpt}', 'spp-weis', f's3_models/{ckpt}')


loaded_models = [d['Key'] for d in s3.list_objects(Bucket='spp-weis')['Contents'] if 's3_models/' in d['Key']]
log.info(f'loaded_models: {loaded_models}')


models_to_delete = [l for l in loaded_models if l.split('/')[-1] not in ckpt_uploads]
log.info(f'models_to_delete: {models_to_delete}')


for del_model in models_to_delete:
    log.info(f'removing: {del_model}')
    s3.delete_object(Bucket='spp-weis', Key=del_model)


log.info('finished retraining')


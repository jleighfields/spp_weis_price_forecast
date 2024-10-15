# see for custom model set up
# https://github.com/mlflow/mlflow/blob/74d75109aaf2975f5026104d6125bb30f4e3f744/mlflow/pytorch.py

# for custom sktime model
# https://mlflow.org/docs/latest/models.html#example-creating-a-custom-sktime-flavor
# https://changhsinlee.com/mlflow-custom-flavor/

import os
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import time
import torch
from darts import TimeSeries


# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DartsGlobalModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from darts.models import (
            TFTModel,
            TiDEModel,
            DLinearModel,
            NLinearModel,
            TSMixerModel,
        )
        import pickle
        # print(f'context.artifacts: {context.artifacts}')
        # device = 0 if torch.cuda.is_available() else -1

        # load model type
        with open(context.artifacts["MODEL_TYPE"], 'rb') as handle:
            self.MODEL_TYPE = pickle.load(handle)
        log.info(f'MODEL_TYPE: {self.MODEL_TYPE}')

        # load scalers
        # with open(context.artifacts["scalers"], 'rb') as handle:
        #     self.scalers = pickle.load(handle)

        # load model based on MODEL_TYPE
        if self.MODEL_TYPE == "tft_model":
            self.model = TFTModel.load(context.artifacts["model"], map_location=torch.device('cpu'))
            
        elif self.MODEL_TYPE == "tide_model":
            self.model = TiDEModel.load(context.artifacts["model"], map_location=torch.device('cpu'))
            
        elif self.MODEL_TYPE == "dlinear_model":
            self.model = DLinearModel.load(context.artifacts["model"], map_location=torch.device('cpu'))
            
        elif self.MODEL_TYPE == "nlinear_model":
            self.model = NLinearModel.load(context.artifacts["model"], map_location=torch.device('cpu'))

        elif self.MODEL_TYPE == "ts_mixer_model":
            self.model = TSMixerModel.load(context.artifacts["model"], map_location=torch.device('cpu'))

        # TODO: add ensemble models
            
        else:
            raise ValueError(f'Unsuported MODEL_TYPE: {self.MODEL_TYPE}')

        # load model train time
        with open(context.artifacts["TRAIN_TIMESTAMP"], 'rb') as handle:
            self.TRAIN_TIMESTAMP = pickle.load(handle)
        log.info(f'TRAIN_TIMESTAMP: {self.TRAIN_TIMESTAMP}')

    def get_raw_model(self, context):
        with open(context.artifacts["TRAIN_TIMESTAMP"], 'rb') as handle:
            TRAIN_TIMESTAMP = pickle.load(handle)
        return TRAIN_TIMESTAMP

    def __repr__(self):
        return self.model.__repr__()
        
    def __str__(self):
        return self.model.__str__()

    def predict(self, context, model_input):
        """
        Custom predict function for Darts forecasting model.
        Args:
            model_input: pd.DataFrame. Containes the unscaled series to make prediction for,
                         future covariate series, and past covariate series as columns of a dataframe.
        Returns:
            prediction: json-formatted time series in original scale.
        """
        # ".from_json() returns a float64 dtype"
        log.info('READING INPUTS...')
        log.info(f'model_input.columns: {model_input.columns}')
        log.info(f"model_input['n']: {model_input['n'].item()}")
        series = TimeSeries.from_json(model_input['series'][0]).astype(np.float32) 
        past_covariates = TimeSeries.from_json(model_input['past_covariates'][0]).astype(np.float32)
        future_covariates = TimeSeries.from_json(model_input['future_covariates'][0]).astype(np.float32)
        forecast_horizon = model_input['n'].item()
        num_samples = model_input['num_samples'].item()

        
        # scale time series
        # log.info('SCALING INPUTS...')
        # series_scaled = TimeSeries.from_dataframe(
        #     series.pd_dataframe()/self.scalers['series']
        #     )

        # past_covariates_scaled = TimeSeries.from_dataframe(
        #     past_covariates.pd_dataframe()/self.scalers['pc']
        #     )
        
        # future_covariates_scaled = TimeSeries.from_dataframe(
        #     future_covariates.pd_dataframe()/self.scalers['fc']
        #     )

        log.info('RUNNING PREDICT...')
        pred_series = self.model.predict(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                n=forecast_horizon,
                num_samples=num_samples
            )
        
        pred_series = TimeSeries.from_dataframe(
            pred_series.pd_dataframe()
            )

        return TimeSeries.to_json(pred_series)
    

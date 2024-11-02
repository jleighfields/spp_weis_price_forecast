'''
Module for custom Darts model serving using mlflow pyfunc
'''

import logging
import numpy as np
import mlflow.pyfunc
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
            TSMixerModel,
            NaiveEnsembleModel,
        )
        import pickle
        # print(f'context.artifacts: {context.artifacts}')

        # load model type
        with open(context.artifacts["MODEL_TYPE.pkl"], 'rb') as handle:
            self.MODEL_TYPE = pickle.load(handle)
        log.info(f'MODEL_TYPE: {self.MODEL_TYPE}')

        # load model based on MODEL_TYPE
        if self.MODEL_TYPE == "tft_model":
            self.model = TFTModel.load(context.artifacts["model"], map_location=torch.device('cpu'))
            
        elif self.MODEL_TYPE == "tide_model":
            self.model = TiDEModel.load(context.artifacts["model"], map_location=torch.device('cpu'))

        elif self.MODEL_TYPE == "ts_mixer_model":
            self.model = TSMixerModel.load(context.artifacts["model"], map_location=torch.device('cpu'))

        elif self.MODEL_TYPE == 'naive_ens':
            # TODO: subclass ensemble model to allow map_location kwarg
            # otherwise loading model fails if ensembled models are trained on a gpu
            # because the models are pickled when saved
            # https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu
            self.model = NaiveEnsembleModel.load(context.artifacts["model"])
            
        else:
            raise ValueError(f'Unsuported MODEL_TYPE: {self.MODEL_TYPE}')

        # load model train time
        with open(context.artifacts["TRAIN_TIMESTAMP.pkl"], 'rb') as handle:
            self.model.TRAIN_TIMESTAMP = pickle.load(handle)
        log.info(f'TRAIN_TIMESTAMP: {self.model.TRAIN_TIMESTAMP}')


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


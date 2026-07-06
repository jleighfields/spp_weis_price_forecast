'''
Module for custom Darts model serving using mlflow pyfunc
'''

import logging
import numpy as np
import pandas as pd
import mlflow.pyfunc
import torch
from darts import TimeSeries

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DartsGlobalModel(mlflow.pyfunc.PythonModel):
    """mlflow PyFunc wrapper serving a global Darts forecasting model.

    Wraps a single Darts model (TiDE/TFT/TSMixer) or a NaiveEnsembleModel
    so it can be logged and served via mlflow. The concrete model type is
    read from the ``MODEL_TYPE.pkl`` artifact at load time.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the Darts model from mlflow artifacts by MODEL_TYPE.

        Args:
            context: mlflow context whose ``artifacts`` map holds the
                serialized model, its MODEL_TYPE, and train timestamp.

        Raises:
            ValueError: If MODEL_TYPE is not a supported model kind.
        """
        from darts.models import (
            TFTModel,
            TiDEModel,
            TSMixerModel,
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
            from src.modeling import load_ensemble_from_dir

            log.info(f'context.artifacts["ens_models"]: {context.artifacts["ens_models"]}')
            model_path = context.artifacts["ens_models"]
            self.model, self.model.TRAIN_TIMESTAMP = load_ensemble_from_dir(model_path)
            log.info(f'TRAIN_TIMESTAMP: {self.model.TRAIN_TIMESTAMP}')
            return

        else:
            raise ValueError(f'Unsupported MODEL_TYPE: {self.MODEL_TYPE}')

        # load model train time (single-model branches only; ensemble handles it above)
        with open(context.artifacts["TRAIN_TIMESTAMP.pkl"], 'rb') as handle:
            self.model.TRAIN_TIMESTAMP = pickle.load(handle)
        log.info(f'TRAIN_TIMESTAMP: {self.model.TRAIN_TIMESTAMP}')


    def __repr__(self) -> str:
        return self.model.__repr__()

    def __str__(self) -> str:
        return self.model.__str__()

    def predict(self, context: mlflow.pyfunc.PythonModelContext,
                model_input: pd.DataFrame) -> str:
        """Forecast from json-serialized series and covariates.

        Args:
            context: mlflow context (unused; required by the PyFunc API).
            model_input: One-row DataFrame with json-serialized 'series',
                'past_covariates', and 'future_covariates' columns plus the
                'n' (horizon) and 'num_samples' scalars.

        Returns:
            The forecast TimeSeries as a json string, in the original scale.
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
            pred_series.to_dataframe()
            )

        return TimeSeries.to_json(pred_series)


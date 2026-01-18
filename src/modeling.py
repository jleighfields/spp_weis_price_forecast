"""
Modeling module for SPP WEIS LMP price forecasting.

This module provides functions to build, configure, and train deep learning
time series forecasting models using the Darts library. Supported models include:

- TSMixerModel: Mixer-based architecture for multivariate time series
- TiDEModel: Time-series Dense Encoder model
- TFTModel: Temporal Fusion Transformer

All models are configured for probabilistic forecasting using quantile regression,
enabling prediction intervals for uncertainty quantification.

Dependencies:
    - darts: Time series forecasting framework
    - pytorch-lightning: Training infrastructure
    - torchmetrics: Model evaluation metrics
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Optional, Any



from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression

from darts.models import (
    TFTModel,
    TiDEModel,
    TSMixerModel
)


from torchmetrics import (
    MeanSquaredError,
)


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
    f'{home}/Documents/github/spp_weis_price_forecast/src',
]
for module_path in module_paths:
    if os.path.isdir(module_path):
        log.info('adding module path')
        sys.path.insert(0, module_path)

import parameters


import pprint
# set up pretty printer
pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)


def log_pretty(obj: Any) -> str:
    """
    Format an object for pretty-printed logging output.

    Args:
        obj: Any Python object to format.

    Returns:
        str: Formatted string representation with trailing newline.
    """
    pretty_out = f"{pp.pformat(obj)}"

    return f'{pretty_out}\n'
    

def build_fit_tsmixerx(
    series: List[TimeSeries],
    val_series: List[TimeSeries],
    future_covariates: List[TimeSeries],
    past_covariates: List[TimeSeries],
    hidden_size: int = 10,
    ff_size: int = 27,
    num_blocks: int = 3,
    forecast_horizon: int = parameters.FORECAST_HORIZON,
    input_chunk_length: int = parameters.INPUT_CHUNK_LENGTH,
    lr: float = 8e-4,
    batch_size: int = 64,
    use_reversible_instance_norm: bool = True,
    n_epochs: int = 8,
    dropout: float = 0.45,
    encoder_key: str = 'rel',
    activation: str = 'ELU',
    force_reset: bool = True,
    callbacks: Optional[List[Any]] = None,
    model_id: str = 'ts_mixer',
    log_tensorboard: bool = False,
) -> TSMixerModel:
    """
    Build and train a TSMixer model for time series forecasting.

    TSMixer is a mixer-based architecture that applies MLP-Mixer concepts
    to multivariate time series forecasting with probabilistic outputs.

    Args:
        series: List of target TimeSeries for training.
        val_series: List of target TimeSeries for validation.
        future_covariates: List of future covariate TimeSeries.
        past_covariates: List of past covariate TimeSeries.
        hidden_size: Hidden layer size in mixer blocks.
        ff_size: Feed-forward layer size.
        num_blocks: Number of mixer blocks.
        forecast_horizon: Number of time steps to forecast.
        input_chunk_length: Number of historical time steps as input.
        lr: Learning rate for Adam optimizer.
        batch_size: Training batch size.
        use_reversible_instance_norm: Apply reversible instance normalization.
        n_epochs: Number of training epochs.
        dropout: Dropout probability.
        encoder_key: Key for time encoders from parameters.ENCODERS.
        activation: Activation function name (e.g., 'ELU', 'ReLU', 'GELU').
        force_reset: Reset model checkpoint if exists.
        callbacks: Optional list of PyTorch Lightning callbacks.
        model_id: Model identifier for checkpointing.
        log_tensorboard: Enable TensorBoard logging.

    Returns:
        TSMixerModel: Trained model with MODEL_TYPE and TRAIN_TIMESTAMP attributes.
    """
    MODEL_TYPE = "ts_mixer_model"
    work_dir = os.getcwd() + f'/model_checkpoints/{MODEL_TYPE}'
    os.makedirs(work_dir, exist_ok=True)
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...


    # common parameters across models
    model_params = {
        'hidden_size': hidden_size,
        'ff_size': ff_size,
        'num_blocks': num_blocks,
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': forecast_horizon,
        'batch_size': batch_size,
        # 'use_layer_norm': use_layer_norm,
        'use_reversible_instance_norm': use_reversible_instance_norm,
        'n_epochs': n_epochs,
        'dropout': dropout,
        'activation': activation,
        'add_encoders': parameters.ENCODERS[encoder_key],
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': model_id, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': log_tensorboard,
    }

    # throughout training we'll monitor the validation loss for early stopping
    # early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
    # if callbacks is None:
    #     callbacks = [early_stopper]
    # else:
    #     callbacks = [early_stopper] + callbacks

    # pl_trainer_kwargs = {"callbacks": callbacks}
    # if pl_trainer_kwargs:
    #     model_params['pl_trainer_kwargs'] = pl_trainer_kwargs
        
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
    num_encoder_decoder_layers: int = 1,
    decoder_output_dim: int = 17,
    hidden_size: int = 32,
    temporal_width_past: int = 1,
    temporal_width_future: int = 1,
    temporal_decoder_hidden: int = 20,
    temporal_hidden_size_past: int = 24,
    temporal_hidden_size_future: int = 24,
    forecast_horizon: int = parameters.FORECAST_HORIZON,
    input_chunk_length: int = parameters.INPUT_CHUNK_LENGTH,
    lr: float = 3e-4,
    batch_size: int = 64,
    use_layer_norm: bool = False,
    use_reversible_instance_norm: bool = True,
    n_epochs: int = 10,
    dropout: float = 0.43,
    encoder_key: str = 'rel',
    force_reset: bool = True,
    callbacks: Optional[List[Any]] = None,
    model_id: str = 'tide',
    log_tensorboard: bool = False,
) -> TiDEModel:
    """
    Build and train a TiDE model for time series forecasting.

    TiDE (Time-series Dense Encoder) uses dense encoder-decoder architecture
    with temporal processing for multivariate time series forecasting.

    Args:
        series: List of target TimeSeries for training.
        val_series: List of target TimeSeries for validation.
        future_covariates: List of future covariate TimeSeries.
        past_covariates: List of past covariate TimeSeries.
        num_encoder_decoder_layers: Number of encoder and decoder layers.
        decoder_output_dim: Output dimension of decoder.
        hidden_size: Hidden layer size.
        temporal_width_past: Temporal convolution width for past.
        temporal_width_future: Temporal convolution width for future.
        temporal_decoder_hidden: Hidden size for temporal decoder.
        temporal_hidden_size_past: Hidden size for past temporal processing.
        temporal_hidden_size_future: Hidden size for future temporal processing.
        forecast_horizon: Number of time steps to forecast.
        input_chunk_length: Number of historical time steps as input.
        lr: Learning rate for Adam optimizer.
        batch_size: Training batch size.
        use_layer_norm: Apply layer normalization.
        use_reversible_instance_norm: Apply reversible instance normalization.
        n_epochs: Number of training epochs.
        dropout: Dropout probability.
        encoder_key: Key for time encoders from parameters.ENCODERS.
        force_reset: Reset model checkpoint if exists.
        callbacks: Optional list of PyTorch Lightning callbacks.
        model_id: Model identifier for checkpointing.
        log_tensorboard: Enable TensorBoard logging.

    Returns:
        TiDEModel: Trained model with MODEL_TYPE and TRAIN_TIMESTAMP attributes.
    """
    MODEL_TYPE = "tide_model"
    work_dir = os.getcwd() + f'/model_checkpoints/{MODEL_TYPE}'
    os.makedirs(work_dir, exist_ok=True)
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...


    # common parameters across models
    model_params = {
        'num_encoder_layers': num_encoder_decoder_layers,
        'num_decoder_layers': num_encoder_decoder_layers,
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
        'use_layer_norm': use_layer_norm,
        'use_reversible_instance_norm': use_reversible_instance_norm,
        'n_epochs': n_epochs,
        'dropout': dropout,
        'add_encoders': parameters.ENCODERS[encoder_key],
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': model_id, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': log_tensorboard,
    }

    # throughout training we'll monitor the validation loss for early stopping
    # early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
    # if callbacks is None:
    #     callbacks = [early_stopper]
    # else:
    #     callbacks = [early_stopper] + callbacks

    # pl_trainer_kwargs = {"callbacks": callbacks}
    # if pl_trainer_kwargs:
    #     model_params['pl_trainer_kwargs'] = pl_trainer_kwargs
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
    hidden_size: int = 9,
    lstm_layers: int = 4,
    num_attention_heads: int = 1,
    dropout: float = 0.49,
    full_attention: bool = True,
    forecast_horizon: int = parameters.FORECAST_HORIZON,
    input_chunk_length: int = parameters.INPUT_CHUNK_LENGTH,
    lr: float = 1.0e-3,
    batch_size: int = 64,
    use_reversible_instance_norm: bool = True,
    n_epochs: int = 3,
    encoder_key: str = 'rel',
    force_reset: bool = True,
    callbacks: Optional[List[Any]] = None,
    model_id: str = 'tft',
    log_tensorboard: bool = False,
) -> TFTModel:
    """
    Build and train a Temporal Fusion Transformer model for time series forecasting.

    TFT combines LSTM encoders with multi-head attention for interpretable
    and accurate multivariate time series forecasting.

    Args:
        series: List of target TimeSeries for training.
        val_series: List of target TimeSeries for validation.
        future_covariates: List of future covariate TimeSeries.
        past_covariates: List of past covariate TimeSeries.
        hidden_size: Hidden state size, main hyper-parameter for TFT architecture.
        lstm_layers: Number of LSTM encoder/decoder layers.
        num_attention_heads: Number of attention heads.
        dropout: Dropout probability.
        full_attention: Use full attention mechanism.
        forecast_horizon: Number of time steps to forecast.
        input_chunk_length: Number of historical time steps as input.
        lr: Learning rate for Adam optimizer.
        batch_size: Training batch size.
        use_reversible_instance_norm: Apply reversible instance normalization.
        n_epochs: Number of training epochs.
        encoder_key: Key for time encoders from parameters.ENCODERS.
        force_reset: Reset model checkpoint if exists.
        callbacks: Optional list of PyTorch Lightning callbacks.
        model_id: Model identifier for checkpointing.
        log_tensorboard: Enable TensorBoard logging.

    Returns:
        TFTModel: Trained model with MODEL_TYPE and TRAIN_TIMESTAMP attributes.
    """
    MODEL_TYPE = "tft_model"
    work_dir = os.getcwd() + f'/model_checkpoints/{MODEL_TYPE}'
    os.makedirs(work_dir, exist_ok=True)
    quantiles = [0.01]+np.arange(0.05, 1, 0.05).tolist()+[0.99]
    
    #TODO: pick a metric...
    # torch_metrics = MeanAbsoluteError()
    torch_metrics = MeanSquaredError(squared=False)
    # torch_metrics = SymmetricMeanAbsolutePercentageError() # don't use...


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
        'use_reversible_instance_norm': use_reversible_instance_norm,
        'n_epochs': n_epochs,
        'add_encoders': parameters.ENCODERS[encoder_key],
        'likelihood': QuantileRegression(quantiles=quantiles),  # QuantileRegression is set per default
        'optimizer_kwargs': {"lr": lr},
        'random_state': 42,
        'torch_metrics': torch_metrics,
        'use_static_covariates': False,
        'save_checkpoints': True,
        'work_dir': work_dir,
        'model_name': model_id, # used for checkpoint saves
        'force_reset': force_reset, # reset model if already exists
        'log_tensorboard': log_tensorboard,
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

    


def get_ci_err(
    actual_series: List[TimeSeries],
    pred_series: List[TimeSeries],
    n_jobs: int = 1,
    verbose: bool = False,
) -> List[float]:
    """
    Calculate confidence interval coverage error for predictions.

    Computes how far the 80% prediction interval coverage deviates from
    the expected 80% for each series.

    Args:
        actual_series: List of actual target TimeSeries.
        pred_series: List of predicted TimeSeries with quantiles.
        n_jobs: Number of parallel jobs (currently unused).
        verbose: Enable verbose output (currently unused).

    Returns:
        List[float]: Coverage error percentage for each series, where 0%
            means perfect 80% coverage and higher values indicate worse
            calibration.
    """
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
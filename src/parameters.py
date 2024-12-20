'''
set up global parameters
'''

from sklearn.preprocessing import RobustScaler
from darts.dataprocessing.transformers import Scaler


TRAIN_START = '365D'

FORECAST_HORIZON = 24*5
INPUT_CHUNK_LENGTH = 24*7
PRECISION = 'float32'
MODEL_NAME = 'spp_weis'

USE_TSMIXER = False
USE_TIDE = True
USE_TFT = False

TOP_N = 5


## set of encoders for experiment
ENCODERS = {}

ENCODERS['rel'] = {
    "position": {
        "past": ["relative"], 
        "future": ["relative"]
    },
    "transformer": Scaler(RobustScaler(), global_fit=True)
    }

ENCODERS['rel_mon'] = {
            "datetime_attribute": {
                "future": ["month"], 
                "past": ["month"], 
            },
            "position": {
                "past": ["relative"], 
                "future": ["relative"]
            },
            "transformer": Scaler(RobustScaler(), global_fit=True)
        }

ENCODERS['rel_mon_day'] = {
            "datetime_attribute": {
                "future": ["month", "dayofweek"], 
                "past": ["month", "dayofweek"], 
            },
            "position": {
                "past": ["relative"], 
                "future": ["relative"]
            },
            "transformer": Scaler(RobustScaler(), global_fit=True)
        }

ENCODERS['rel_mon_day_hour'] = {
            "datetime_attribute": {
                "future": ["month", "dayofweek", "hour"], 
                "past": ["month", "dayofweek", "hour"], 
            },
            "position": {
                "past": ["relative"], 
                "future": ["relative"]
            },
            "transformer": Scaler(RobustScaler(), global_fit=True)
        }




# best tsmixer model params from optuna experiment
TSMIXER_PARAMS = [{'hidden_size': 104,
  'ff_size': 18,
  'num_blocks': 8,
  'lr': 8.9e-05,
  'n_epochs': 7,
  'dropout': 0.49,
  'activation': 'ELU'},
 {'hidden_size': 66,
  'ff_size': 18,
  'num_blocks': 8,
  'lr': 5.9000000000000004e-05,
  'n_epochs': 8,
  'dropout': 0.43,
  'activation': 'ELU'},
 {'hidden_size': 112,
  'ff_size': 18,
  'num_blocks': 8,
  'lr': 5.9000000000000004e-05,
  'n_epochs': 8,
  'dropout': 0.43,
  'activation': 'ELU'},
 {'hidden_size': 124,
  'ff_size': 56,
  'num_blocks': 6,
  'lr': 7.1e-05,
  'n_epochs': 7,
  'dropout': 0.44999999999999996,
  'activation': 'ELU'},
 {'hidden_size': 124,
  'ff_size': 88,
  'num_blocks': 6,
  'lr': 9.2e-05,
  'n_epochs': 5,
  'dropout': 0.44999999999999996,
  'activation': 'ELU'}]


# best tide model params from optuna experiment
TIDE_PARAMS = [{'num_encoder_decoder_layers': 1,
  'decoder_output_dim': 17,
  'hidden_size': 20,
  'temporal_width': 0,
  'temporal_decoder_hidden': 16,
  'temporal_hidden_size': 11,
  'lr': 4.9e-05,
  'n_epochs': 18,
  'dropout': 0.43999999999999995,
  'encoder_key': 'rel_mon_day'},
 {'num_encoder_decoder_layers': 1,
  'decoder_output_dim': 13,
  'hidden_size': 14,
  'temporal_width': 0,
  'temporal_decoder_hidden': 6,
  'temporal_hidden_size': 9,
  'lr': 7e-05,
  'n_epochs': 19,
  'dropout': 0.49,
  'encoder_key': 'rel_mon_day'},
 {'num_encoder_decoder_layers': 1,
  'decoder_output_dim': 31,
  'hidden_size': 25,
  'temporal_width': 0,
  'temporal_decoder_hidden': 30,
  'temporal_hidden_size': 30,
  'lr': 5.1e-05,
  'n_epochs': 19,
  'dropout': 0.44999999999999996,
  'encoder_key': 'rel_mon'},
 {'num_encoder_decoder_layers': 3,
  'decoder_output_dim': 23,
  'hidden_size': 14,
  'temporal_width': 0,
  'temporal_decoder_hidden': 7,
  'temporal_hidden_size': 8,
  'lr': 4.4e-05,
  'n_epochs': 15,
  'dropout': 0.5,
  'encoder_key': 'rel'},
 {'num_encoder_decoder_layers': 3,
  'decoder_output_dim': 21,
  'hidden_size': 24,
  'temporal_width': 1,
  'temporal_decoder_hidden': 19,
  'temporal_hidden_size': 16,
  'lr': 4.7999999999999994e-05,
  'n_epochs': 14,
  'dropout': 0.43,
  'encoder_key': 'rel'}]


# best tide model params from optuna experiment
TFT_PARAMS = [{'hidden_size': 43,
  'lstm_layers': 1,
  'num_attention_heads': 1,
  'lr': 0.00015000000000000001,
  'n_epochs': 2,
  'dropout': 0.31},
 {'hidden_size': 35,
  'lstm_layers': 2,
  'num_attention_heads': 1,
  'lr': 0.00048,
  'n_epochs': 1,
  'dropout': 0.36},
 {'hidden_size': 26,
  'lstm_layers': 4,
  'num_attention_heads': 3,
  'lr': 0.00025,
  'n_epochs': 2,
  'dropout': 0.48},
 {'hidden_size': 16,
  'lstm_layers': 4,
  'num_attention_heads': 3,
  'lr': 0.0007700000000000001,
  'n_epochs': 3,
  'dropout': 0.43},
 {'hidden_size': 63,
  'lstm_layers': 1,
  'num_attention_heads': 2,
  'lr': 0.0004,
  'n_epochs': 1,
  'dropout': 0.35}]

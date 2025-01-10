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

USE_TSMIXER = True
USE_TIDE = False
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

# ENCODERS['rel_mon_day_hour'] = {
#             "datetime_attribute": {
#                 "future": ["month", "dayofweek", "hour"], 
#                 "past": ["month", "dayofweek", "hour"], 
#             },
#             "position": {
#                 "past": ["relative"], 
#                 "future": ["relative"]
#             },
#             "transformer": Scaler(RobustScaler(), global_fit=True)
#         }




# best tsmixer model params from optuna experiment
TSMIXER_PARAMS = [{'hidden_size': 42,
  'ff_size': 204,
  'num_blocks': 12,
  'lr': 6.7e-05,
  'n_epochs': 10,
  'dropout': 0.41000000000000003,
  'activation': 'ELU',
  'encoder_key': 'rel_mon'},
 {'hidden_size': 212,
  'ff_size': 182,
  'num_blocks': 9,
  'lr': 3.9999999999999996e-05,
  'n_epochs': 10,
  'dropout': 0.46,
  'activation': 'ELU',
  'encoder_key': 'rel_mon'},
 {'hidden_size': 50,
  'ff_size': 208,
  'num_blocks': 12,
  'lr': 5.6e-05,
  'n_epochs': 9,
  'dropout': 0.5,
  'activation': 'SELU',
  'encoder_key': 'rel_mon'},
 {'hidden_size': 166,
  'ff_size': 232,
  'num_blocks': 9,
  'lr': 8.8e-05,
  'n_epochs': 4,
  'dropout': 0.49,
  'activation': 'ELU',
  'encoder_key': 'rel_mon_day'},
 {'hidden_size': 170,
  'ff_size': 232,
  'num_blocks': 9,
  'lr': 3.9999999999999996e-05,
  'n_epochs': 7,
  'dropout': 0.46,
  'activation': 'ELU',
  'encoder_key': 'rel_mon_day'}]


# best tide model params from optuna experiment
TIDE_PARAMS = [{'num_encoder_decoder_layers': 3,
  'decoder_output_dim': 18,
  'hidden_size': 30,
  'temporal_width': 0,
  'temporal_decoder_hidden': 14,
  'temporal_hidden_size': 30,
  'lr': 4.7e-05,
  'n_epochs': 18,
  'dropout': 0.35,
  'encoder_key': 'rel_mon'},
 {'num_encoder_decoder_layers': 3,
  'decoder_output_dim': 28,
  'hidden_size': 30,
  'temporal_width': 2,
  'temporal_decoder_hidden': 30,
  'temporal_hidden_size': 30,
  'lr': 2.6000000000000002e-05,
  'n_epochs': 18,
  'dropout': 0.39999999999999997,
  'encoder_key': 'rel_mon'},
 {'num_encoder_decoder_layers': 1,
  'decoder_output_dim': 15,
  'hidden_size': 25,
  'temporal_width': 2,
  'temporal_decoder_hidden': 13,
  'temporal_hidden_size': 10,
  'lr': 2.9999999999999997e-05,
  'n_epochs': 18,
  'dropout': 0.37,
  'encoder_key': 'rel'},
 {'num_encoder_decoder_layers': 2,
  'decoder_output_dim': 20,
  'hidden_size': 28,
  'temporal_width': 1,
  'temporal_decoder_hidden': 30,
  'temporal_hidden_size': 25,
  'lr': 6.8e-05,
  'n_epochs': 17,
  'dropout': 0.39999999999999997,
  'encoder_key': 'rel'},
 {'num_encoder_decoder_layers': 2,
  'decoder_output_dim': 32,
  'hidden_size': 16,
  'temporal_width': 0,
  'temporal_decoder_hidden': 15,
  'temporal_hidden_size': 28,
  'lr': 3.2e-05,
  'n_epochs': 12,
  'dropout': 0.35,
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

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
TSMIXER_PARAMS = [{'hidden_size': 62,
  'ff_size': 38,
  'num_blocks': 7,
  'lr': 5.2999999999999994e-05,
  'n_epochs': 7,
  'dropout': 0.47000000000000003,
  'activation': 'ELU',
  'encoder_key': 'rel'},
 {'hidden_size': 126,
  'ff_size': 252,
  'num_blocks': 4,
  'lr': 5.4e-05,
  'n_epochs': 12,
  'dropout': 0.43000000000000005,
  'activation': 'SELU',
  'encoder_key': 'rel_mon'},
 {'hidden_size': 116,
  'ff_size': 136,
  'num_blocks': 4,
  'lr': 4.7999999999999994e-05,
  'n_epochs': 7,
  'dropout': 0.46,
  'activation': 'ELU',
  'encoder_key': 'rel_mon_day'},
 {'hidden_size': 76,
  'ff_size': 68,
  'num_blocks': 5,
  'lr': 8.499999999999999e-05,
  'n_epochs': 6,
  'dropout': 0.5,
  'activation': 'ELU',
  'encoder_key': 'rel'},
 {'hidden_size': 126,
  'ff_size': 86,
  'num_blocks': 4,
  'lr': 6.4e-05,
  'n_epochs': 8,
  'dropout': 0.45,
  'activation': 'ELU',
  'encoder_key': 'rel_mon_day'}]


# best tide model params from optuna experiment
TIDE_PARAMS = [{'num_encoder_decoder_layers': 1,
  'decoder_output_dim': 8,
  'hidden_size': 4,
  'temporal_width': 1,
  'temporal_decoder_hidden': 16,
  'temporal_hidden_size': 9,
  'lr': 3.8e-05,
  'n_epochs': 9,
  'dropout': 0.49,
  'encoder_key': 'rel_mon_day'},
 {'num_encoder_decoder_layers': 5,
  'decoder_output_dim': 8,
  'hidden_size': 9,
  'temporal_width': 1,
  'temporal_decoder_hidden': 12,
  'temporal_hidden_size': 14,
  'lr': 4.7e-05,
  'n_epochs': 8,
  'dropout': 0.38999999999999996,
  'encoder_key': 'rel_mon'},
 {'num_encoder_decoder_layers': 3,
  'decoder_output_dim': 18,
  'hidden_size': 15,
  'temporal_width': 1,
  'temporal_decoder_hidden': 32,
  'temporal_hidden_size': 13,
  'lr': 1.5e-05,
  'n_epochs': 18,
  'dropout': 0.38999999999999996,
  'encoder_key': 'rel_mon_day'},
 {'num_encoder_decoder_layers': 7,
  'decoder_output_dim': 26,
  'hidden_size': 12,
  'temporal_width': 1,
  'temporal_decoder_hidden': 26,
  'temporal_hidden_size': 4,
  'lr': 4.4999999999999996e-05,
  'n_epochs': 10,
  'dropout': 0.42,
  'encoder_key': 'rel'},
 {'num_encoder_decoder_layers': 6,
  'decoder_output_dim': 31,
  'hidden_size': 10,
  'temporal_width': 2,
  'temporal_decoder_hidden': 10,
  'temporal_hidden_size': 15,
  'lr': 4.9999999999999996e-05,
  'n_epochs': 8,
  'dropout': 0.47,
  'encoder_key': 'rel_mon_day'}]


# best tide model params from optuna experiment
TFT_PARAMS = [{'hidden_size': 20,
  'lstm_layers': 1,
  'num_attention_heads': 3,
  'lr': 0.000749,
  'n_epochs': 5,
  'dropout': 0.49,
  'full_attention': True,
  'encoder_key': 'rel_mon'},
 {'hidden_size': 9,
  'lstm_layers': 2,
  'num_attention_heads': 4,
  'lr': 0.000412,
  'n_epochs': 6,
  'dropout': 0.44,
  'full_attention': False,
  'encoder_key': 'rel_mon'},
 {'hidden_size': 8,
  'lstm_layers': 2,
  'num_attention_heads': 3,
  'lr': 0.000371,
  'n_epochs': 4,
  'dropout': 0.44999999999999996,
  'full_attention': False,
  'encoder_key': 'rel'},
 {'hidden_size': 35,
  'lstm_layers': 2,
  'num_attention_heads': 1,
  'lr': 0.0007769999999999999,
  'n_epochs': 2,
  'dropout': 0.44,
  'full_attention': False,
  'encoder_key': 'rel'},
 {'hidden_size': 47,
  'lstm_layers': 1,
  'num_attention_heads': 4,
  'lr': 0.00030900000000000003,
  'n_epochs': 6,
  'dropout': 0.39,
  'full_attention': True,
  'encoder_key': 'rel_mon'}]



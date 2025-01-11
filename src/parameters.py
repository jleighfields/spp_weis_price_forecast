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
USE_TFT = True

TOP_N = 2


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
TIDE_PARAMS = [{'num_encoder_decoder_layers': 4,
  'decoder_output_dim': 27,
  'hidden_size': 13,
  'temporal_width': 2,
  'temporal_decoder_hidden': 4,
  'temporal_hidden_size': 7,
  'lr': 4.9e-05,
  'n_epochs': 18,
  'dropout': 0.48,
  'encoder_key': 'rel_mon'},
 {'num_encoder_decoder_layers': 2,
  'decoder_output_dim': 17,
  'hidden_size': 10,
  'temporal_width': 2,
  'temporal_decoder_hidden': 24,
  'temporal_hidden_size': 15,
  'lr': 8.3e-05,
  'n_epochs': 11,
  'dropout': 0.44999999999999996,
  'encoder_key': 'rel_mon'},
 {'num_encoder_decoder_layers': 4,
  'decoder_output_dim': 12,
  'hidden_size': 16,
  'temporal_width': 1,
  'temporal_decoder_hidden': 20,
  'temporal_hidden_size': 15,
  'lr': 6.9e-05,
  'n_epochs': 17,
  'dropout': 0.48,
  'encoder_key': 'rel_mon'},
 {'num_encoder_decoder_layers': 1,
  'decoder_output_dim': 5,
  'hidden_size': 4,
  'temporal_width': 1,
  'temporal_decoder_hidden': 23,
  'temporal_hidden_size': 14,
  'lr': 8.6e-05,
  'n_epochs': 16,
  'dropout': 0.43999999999999995,
  'encoder_key': 'rel_mon_day'},
 {'num_encoder_decoder_layers': 1,
  'decoder_output_dim': 16,
  'hidden_size': 12,
  'temporal_width': 1,
  'temporal_decoder_hidden': 7,
  'temporal_hidden_size': 4,
  'lr': 4.2000000000000004e-05,
  'n_epochs': 16,
  'dropout': 0.44999999999999996,
  'encoder_key': 'rel_mon_day'}]


# best tide model params from optuna experiment
TFT_PARAMS = [{'hidden_size': 60,
  'lstm_layers': 3,
  'num_attention_heads': 1,
  'lr': 0.00026,
  'n_epochs': 2,
  'dropout': 0.42,
  'full_attention': True,
  'encoder_key': 'rel'},
 {'hidden_size': 40,
  'lstm_layers': 1,
  'num_attention_heads': 4,
  'lr': 0.000352,
  'n_epochs': 5,
  'dropout': 0.5,
  'full_attention': True,
  'encoder_key': 'rel'},
 {'hidden_size': 30,
  'lstm_layers': 1,
  'num_attention_heads': 2,
  'lr': 0.000234,
  'n_epochs': 5,
  'dropout': 0.5,
  'full_attention': False,
  'encoder_key': 'rel'},
 {'hidden_size': 20,
  'lstm_layers': 1,
  'num_attention_heads': 4,
  'lr': 0.000458,
  'n_epochs': 5,
  'dropout': 0.44999999999999996,
  'full_attention': True,
  'encoder_key': 'rel_mon'},
 {'hidden_size': 26,
  'lstm_layers': 4,
  'num_attention_heads': 2,
  'lr': 0.00029800000000000003,
  'n_epochs': 6,
  'dropout': 0.5,
  'full_attention': False,
  'encoder_key': 'rel'}]



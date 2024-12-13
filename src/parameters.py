'''
set up global parameters
'''

TRAIN_START = '365D'

FORECAST_HORIZON = 24*5
INPUT_CHUNK_LENGTH = 24*7
PRECISION = 'float32'
MODEL_NAME = 'spp_weis'

USE_TSMIXER = False
USE_TIDE = True
USE_TFT = False

TOP_N = 5


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
TIDE_PARAMS = [{'num_encoder_layers': 1,
  'decoder_output_dim': 11,
  'hidden_size': 32,
  'temporal_width': 2,
  'temporal_decoder_hidden': 12,
  'temporal_hidden_size': 6,
  'lr': 9.400000000000001e-05,
  'n_epochs': 11,
  'dropout': 0.4},
 {'num_encoder_layers': 2,
  'decoder_output_dim': 10,
  'hidden_size': 14,
  'temporal_width': 1,
  'temporal_decoder_hidden': 32,
  'temporal_hidden_size': 18,
  'lr': 9.3e-05,
  'n_epochs': 11,
  'dropout': 0.49},
 {'num_encoder_layers': 2,
  'decoder_output_dim': 24,
  'hidden_size': 26,
  'temporal_width': 0,
  'temporal_decoder_hidden': 6,
  'temporal_hidden_size': 30,
  'lr': 5.6000000000000006e-05,
  'n_epochs': 7,
  'dropout': 0.41000000000000003},
 {'num_encoder_layers': 1,
  'decoder_output_dim': 16,
  'hidden_size': 18,
  'temporal_width': 2,
  'temporal_decoder_hidden': 8,
  'temporal_hidden_size': 22,
  'lr': 6.3e-05,
  'n_epochs': 11,
  'dropout': 0.48000000000000004},
 {'num_encoder_layers': 2,
  'decoder_output_dim': 10,
  'hidden_size': 24,
  'temporal_width': 3,
  'temporal_decoder_hidden': 12,
  'temporal_hidden_size': 8,
  'lr': 5.3e-05,
  'n_epochs': 11,
  'dropout': 0.47000000000000003}]


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

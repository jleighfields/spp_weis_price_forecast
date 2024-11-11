'''
set up global parameters
'''

FORECAST_HORIZON = 24*5
INPUT_CHUNK_LENGTH = 2*FORECAST_HORIZON
PRECISION = 'float32'
MODEL_NAME = 'spp_weis'

# best tsmixer model params from optuna experiment
TSMIXER_PARAMS = [{'hidden_size': 19,
  'ff_size': 29,
  'num_blocks': 4,
  'lr': 0.00021999999999999998,
  'n_epochs': 6,
  'dropout': 0.43000000000000005},
 {'hidden_size': 37,
  'ff_size': 27,
  'num_blocks': 6,
  'lr': 6.500000000000001e-05,
  'n_epochs': 5,
  'dropout': 0.4},
 {'hidden_size': 26,
  'ff_size': 24,
  'num_blocks': 4,
  'lr': 0.000373,
  'n_epochs': 10,
  'dropout': 0.45}]

# best tide model params from optuna experiment
TIDE_PARAMS = [{'num_encoder_layers': 1,
  'decoder_output_dim': 23,
  'hidden_size': 76,
  'temporal_width_past': 9,
  'temporal_width_future': 4,
  'temporal_decoder_hidden': 28,
  'temporal_hidden_size_past': 6,
  'temporal_hidden_size_future': 16,
  'lr': 5.8e-05,
  'n_epochs': 7,
  'dropout': 0.48000000000000004},
 {'num_encoder_layers': 1,
  'decoder_output_dim': 23,
  'hidden_size': 76,
  'temporal_width_past': 5,
  'temporal_width_future': 3,
  'temporal_decoder_hidden': 10,
  'temporal_hidden_size_past': 6,
  'temporal_hidden_size_future': 16,
  'lr': 5.8e-05,
  'n_epochs': 5,
  'dropout': 0.42},
 {'num_encoder_layers': 2,
  'decoder_output_dim': 15,
  'hidden_size': 76,
  'temporal_width_past': 5,
  'temporal_width_future': 1,
  'temporal_decoder_hidden': 14,
  'temporal_hidden_size_past': 8,
  'temporal_hidden_size_future': 28,
  'lr': 5.8e-05,
  'n_epochs': 4,
  'dropout': 0.43000000000000005}]

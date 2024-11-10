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
  'decoder_output_dim': 21,
  'hidden_size': 48,
  'temporal_width_past': 5,
  'temporal_width_future': 1,
  'temporal_decoder_hidden': 28,
  'temporal_hidden_size_past': 20,
  'temporal_hidden_size_future': 22,
  'lr': 9.400000000000001e-05,
  'n_epochs': 4,
  'dropout': 0.41},
 {'num_encoder_layers': 1,
  'decoder_output_dim': 11,
  'hidden_size': 60,
  'temporal_width_past': 5,
  'temporal_width_future': 4,
  'temporal_decoder_hidden': 20,
  'temporal_hidden_size_past': 10,
  'temporal_hidden_size_future': 26,
  'lr': 0.00011200000000000001,
  'n_epochs': 4,
  'dropout': 0.39999999999999997},
 {'num_encoder_layers': 2,
  'decoder_output_dim': 10,
  'hidden_size': 124,
  'temporal_width_past': 4,
  'temporal_width_future': 2,
  'temporal_decoder_hidden': 16,
  'temporal_hidden_size_past': 28,
  'temporal_hidden_size_future': 20,
  'lr': 0.000115,
  'n_epochs': 4,
  'dropout': 0.49}]
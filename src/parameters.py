'''
set up global parameters
'''

FORECAST_HORIZON = 24*5
INPUT_CHUNK_LENGTH = 2*FORECAST_HORIZON
PRECISION = 'float32'
MODEL_NAME = 'spp_weis'


# best tsmixer model params from optuna experiment
TSMIXER_PARAMS = [{'hidden_size': 13,
  'ff_size': 39,
  'num_blocks': 2,
  'lr': 0.00038,
  'n_epochs': 5,
  'dropout': 0.38},
 {'hidden_size': 11,
  'ff_size': 10,
  'num_blocks': 7,
  'lr': 0.00015000000000000001,
  'n_epochs': 8,
  'dropout': 0.37},
 {'hidden_size': 13,
  'ff_size': 39,
  'num_blocks': 2,
  'lr': 0.00038,
  'n_epochs': 5,
  'dropout': 0.35},
 {'hidden_size': 5,
  'ff_size': 60,
  'num_blocks': 5,
  'lr': 0.00024000000000000003,
  'n_epochs': 8,
  'dropout': 0.3},
 {'hidden_size': 62,
  'ff_size': 39,
  'num_blocks': 1,
  'lr': 0.0008700000000000001,
  'n_epochs': 3,
  'dropout': 0.48}]


# best tide model params from optuna experiment
TIDE_PARAMS = [{'num_encoder_layers': 2,
  'decoder_output_dim': 11,
  'hidden_size': 60,
  'temporal_width_past': 4,
  'temporal_width_future': 9,
  'temporal_decoder_hidden': 20,
  'temporal_hidden_size_past': 18,
  'temporal_hidden_size_future': 24,
  'lr': 5.3e-05,
  'n_epochs': 7,
  'dropout': 0.46},
 {'num_encoder_layers': 1,
  'decoder_output_dim': 30,
  'hidden_size': 60,
  'temporal_width_past': 6,
  'temporal_width_future': 9,
  'temporal_decoder_hidden': 4,
  'temporal_hidden_size_past': 12,
  'temporal_hidden_size_future': 6,
  'lr': 5.3e-05,
  'n_epochs': 4,
  'dropout': 0.5},
 {'num_encoder_layers': 1,
  'decoder_output_dim': 15,
  'hidden_size': 36,
  'temporal_width_past': 2,
  'temporal_width_future': 3,
  'temporal_decoder_hidden': 14,
  'temporal_hidden_size_past': 24,
  'temporal_hidden_size_future': 28,
  'lr': 0.000144,
  'n_epochs': 6,
  'dropout': 0.45}]


# best tide model params from optuna experiment
TFT_PARAMS = [{'hidden_size': 10,
  'lstm_layers': 2,
  'num_attention_heads': 1,
  'lr': 0.00054,
  'n_epochs': 2,
  'dropout': 0.38},
 {'hidden_size': 6,
  'lstm_layers': 3,
  'num_attention_heads': 1,
  'lr': 0.00042,
  'n_epochs': 4,
  'dropout': 0.33999999999999997},
 {'hidden_size': 9,
  'lstm_layers': 3,
  'num_attention_heads': 1,
  'lr': 0.0009400000000000001,
  'n_epochs': 2,
  'dropout': 0.35}]


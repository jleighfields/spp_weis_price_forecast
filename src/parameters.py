'''
set up global parameters
'''

FORECAST_HORIZON = 24*5
INPUT_CHUNK_LENGTH = 24*7
PRECISION = 'float32'
MODEL_NAME = 'spp_weis'


# best tsmixer model params from optuna experiment
TSMIXER_PARAMS = [{'hidden_size': 120,
  'ff_size': 34,
  'num_blocks': 5,
  'lr': 0.000934,
  'n_epochs': 9,
  'dropout': 0.41000000000000003},
 {'hidden_size': 52,
  'ff_size': 54,
  'num_blocks': 8,
  'lr': 0.0008,
  'n_epochs': 5,
  'dropout': 0.4},
 {'hidden_size': 108,
  'ff_size': 116,
  'num_blocks': 3,
  'lr': 0.000735,
  'n_epochs': 7,
  'dropout': 0.48000000000000004},
 {'hidden_size': 38,
  'ff_size': 92,
  'num_blocks': 5,
  'lr': 0.00043299999999999995,
  'n_epochs': 10,
  'dropout': 0.41000000000000003},
 {'hidden_size': 98,
  'ff_size': 38,
  'num_blocks': 6,
  'lr': 0.000805,
  'n_epochs': 9,
  'dropout': 0.49}]

# best tide model params from optuna experiment
TIDE_PARAMS = [{'num_encoder_layers': 2,
  'decoder_output_dim': 24,
  'hidden_size': 64,
  'temporal_width_past': 5,
  'temporal_width_future': 6,
  'temporal_decoder_hidden': 16,
  'temporal_hidden_size_past': 28,
  'temporal_hidden_size_future': 14,
  'lr': 9.8e-05,
  'n_epochs': 4,
  'dropout': 0.42000000000000004},
 {'num_encoder_layers': 1,
  'decoder_output_dim': 23,
  'hidden_size': 76,
  'temporal_width_past': 2,
  'temporal_width_future': 3,
  'temporal_decoder_hidden': 28,
  'temporal_hidden_size_past': 10,
  'temporal_hidden_size_future': 18,
  'lr': 9.6e-05,
  'n_epochs': 5,
  'dropout': 0.45},
 {'num_encoder_layers': 2,
  'decoder_output_dim': 27,
  'hidden_size': 92,
  'temporal_width_past': 8,
  'temporal_width_future': 0,
  'temporal_decoder_hidden': 6,
  'temporal_hidden_size_past': 30,
  'temporal_hidden_size_future': 10,
  'lr': 6.7e-05,
  'n_epochs': 4,
  'dropout': 0.42000000000000004}]


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

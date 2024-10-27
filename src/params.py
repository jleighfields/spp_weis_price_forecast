FORECAST_HORIZON = 24*5
INPUT_CHUNK_LENGTH = 2*FORECAST_HORIZON
PRECISION = 'float32'

# best tsmixer model params from optuna experiment
TSMIXER_PARAMS = [
    {
        'hidden_size': 10,
        'ff_size': 27,
        'num_blocks': 3,
        'lr': 0.0007700000000000001,
        'n_epochs': 8,
        'dropout': 0.45
    },
    {
        'hidden_size': 53,
        'ff_size': 19,
        'num_blocks': 4,
        'lr': 0.00161,
        'n_epochs': 4,
        'dropout': 0.45
    }
]

# best tide model params from optuna experiment
TIDE_PARAMS = [
    {
        'num_encoder_layers': 1,
        'decoder_output_dim': 17,
        'hidden_size': 32,
        'temporal_width_past': 1,
        'temporal_width_future': 0,
        'temporal_decoder_hidden': 20,
        'temporal_hidden_size_past': 4,
        'temporal_hidden_size_future': 24,
        'lr': 0.00029,
        'n_epochs': 10,
        'dropout': 0.43
    },
    {
        'num_encoder_layers': 1,
        'decoder_output_dim': 9,
        'hidden_size': 32,
        'temporal_width_past': 1,
        'temporal_width_future': 6,
        'temporal_decoder_hidden': 24,
        'temporal_hidden_size_past': 4,
        'temporal_hidden_size_future': 28,
        'lr': 8.8e-05,
        'n_epochs': 10,
        'dropout': 0.44999999999999996
    }
]
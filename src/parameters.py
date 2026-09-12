"""
set up global parameters
"""

import os
import sys

from sklearn.preprocessing import RobustScaler
from darts.dataprocessing.transformers import Scaler

# Put src/ on sys.path so the bare `import targets` resolves regardless of how
# this module is imported (matches the other src/ modules).
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Forecast targets live in the darts-free leaf module `targets`; re-export them
# here so existing `parameters.TARGETS` / `parameters.DEFAULT_TARGET` callers
# keep working (single source of truth: src/targets.py).
from targets import DEFAULT_TARGET, TARGETS  # noqa: E402

# Model-selection objectives live in the darts-free leaf module `selection` (so
# the scripts/tune_parameters.py CLI can import them without darts); re-export
# here for darts-side callers (single source of truth: src/selection.py).
from selection import (  # noqa: E402, F401  (re-exported for callers)
    DEFAULT_OBJECTIVE,
    DIAGNOSTIC_BANDS,
    OBJECTIVES,
    TOP_N,
)


TRAIN_START = "365D"

FORECAST_HORIZON = 24 * 5
INPUT_CHUNK_LENGTH = 24 * 7
PRECISION = "float32"

# Default (primary target) model name, kept for callers that predate the target
# dimension.
MODEL_NAME = TARGETS[DEFAULT_TARGET]["model_name"]

# Outlier clipping for TRAINING data — the single home for both the switch and
# the bounds. The hyperparameter study and the retrain must agree: params chosen
# against a tail-suppressed distribution and then trained on the raw one are
# params selected for a dataset that was never served. Both read these.
#
# Deliberately NOT the default of `data_engineering.prep_lmp`: the app calls it
# for the actuals it PLOTS, and clipping those would hide real spikes from
# users. Training opts in explicitly; display paths stay raw.
#
# Tuned on WEIS; re-validate on the spikier IM distribution (run the study once
# True and once False and compare MAE/CRPS and coverage/tail error).
CLIP_OUTLIERS = True
CLIP_QUANTILES = (0.0025, 0.9975)

USE_TSMIXER = False
USE_TIDE = True
USE_TFT = False

# Single home for the QuantileRegression quantile set the models are trained
# on (the three build_fit_* functions in src/modeling.py read this). Wider
# than the old 0.01..0.99 set: the extra 0.001/0.005/0.025 and 0.975/0.995/
# 0.999 levels let the model represent the spike/negative tails, which gives
# a small CRPS gain and honest far-tail (99%) bands for the app (validated
# out-of-sample against the old set).
QUANTILES = [
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
]


## set of encoders for experiment
ENCODERS = {}

ENCODERS["rel"] = {
    "position": {"past": ["relative"], "future": ["relative"]},
    "transformer": Scaler(RobustScaler(), global_fit=True),
}

ENCODERS["rel_mon"] = {
    "datetime_attribute": {
        "future": ["month"],
        "past": ["month"],
    },
    "position": {"past": ["relative"], "future": ["relative"]},
    "transformer": Scaler(RobustScaler(), global_fit=True),
}

ENCODERS["rel_mon_day"] = {
    "datetime_attribute": {
        "future": ["month", "dayofweek"],
        "past": ["month", "dayofweek"],
    },
    "position": {"past": ["relative"], "future": ["relative"]},
    "transformer": Scaler(RobustScaler(), global_fit=True),
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
TSMIXER_PARAMS = [
    {
        "hidden_size": 62,
        "ff_size": 38,
        "num_blocks": 7,
        "lr": 5.2999999999999994e-05,
        "n_epochs": 7,
        "dropout": 0.47000000000000003,
        "activation": "ELU",
        "encoder_key": "rel",
    },
    {
        "hidden_size": 126,
        "ff_size": 252,
        "num_blocks": 4,
        "lr": 5.4e-05,
        "n_epochs": 12,
        "dropout": 0.43000000000000005,
        "activation": "SELU",
        "encoder_key": "rel_mon",
    },
    {
        "hidden_size": 116,
        "ff_size": 136,
        "num_blocks": 4,
        "lr": 4.7999999999999994e-05,
        "n_epochs": 7,
        "dropout": 0.46,
        "activation": "ELU",
        "encoder_key": "rel_mon_day",
    },
    {
        "hidden_size": 76,
        "ff_size": 68,
        "num_blocks": 5,
        "lr": 8.499999999999999e-05,
        "n_epochs": 6,
        "dropout": 0.5,
        "activation": "ELU",
        "encoder_key": "rel",
    },
    {
        "hidden_size": 126,
        "ff_size": 86,
        "num_blocks": 4,
        "lr": 6.4e-05,
        "n_epochs": 8,
        "dropout": 0.45,
        "activation": "ELU",
        "encoder_key": "rel_mon_day",
    },
]


# ── Tuned TiDE params, per target ────────────────────────────────────────
# The TIDE_PARAMS_<TARGET> blocks below (between the >>> / <<< markers) are
# managed by scripts/tune_parameters.py, which replaces a block with the
# top-TOP_N trials from that target's Optuna study. Edit via the script.
# >>> TIDE_PARAMS_RT >>>
# TIDE_PARAMS_RT — top 5 trials by 'mae_ci_rt' score from study 'spp_west_tide_mae_ci_rt' (score 51.138-52.314; metrics mae+ci_err). Managed by scripts/tune_parameters.py.
TIDE_PARAMS_RT = [
    # trial #19  MAE 47.5381  CI_ERR 3.6000  score 51.1381
    {'num_encoder_decoder_layers': 4, 'decoder_output_dim': 21, 'hidden_size': 14, 'temporal_width_past': 5, 'temporal_width_future': 10, 'temporal_decoder_hidden': 53, 'temporal_hidden_size_past': 17, 'temporal_hidden_size_future': 11, 'lr': 0.00022526445927397446, 'n_epochs': 26, 'dropout': 0.5, 'encoder_key': 'rel_mon_day'},
    # trial #88  MAE 47.5381  CI_ERR 3.6000  score 51.1381
    {'num_encoder_decoder_layers': 4, 'decoder_output_dim': 21, 'hidden_size': 14, 'temporal_width_past': 5, 'temporal_width_future': 10, 'temporal_decoder_hidden': 53, 'temporal_hidden_size_past': 17, 'temporal_hidden_size_future': 11, 'lr': 0.00022526445927397446, 'n_epochs': 26, 'dropout': 0.5, 'encoder_key': 'rel_mon_day'},
    # trial #12  MAE 48.6953  CI_ERR 2.6667  score 51.3620
    {'num_encoder_decoder_layers': 4, 'decoder_output_dim': 8, 'hidden_size': 25, 'temporal_width_past': 5, 'temporal_width_future': 0, 'temporal_decoder_hidden': 41, 'temporal_hidden_size_past': 24, 'temporal_hidden_size_future': 18, 'lr': 0.0005479266995583818, 'n_epochs': 23, 'dropout': 0.35, 'encoder_key': 'rel'},
    # trial #32  MAE 49.0050  CI_ERR 2.7167  score 51.7216
    {'num_encoder_decoder_layers': 3, 'decoder_output_dim': 32, 'hidden_size': 14, 'temporal_width_past': 0, 'temporal_width_future': 3, 'temporal_decoder_hidden': 22, 'temporal_hidden_size_past': 23, 'temporal_hidden_size_future': 32, 'lr': 0.00023863763526189405, 'n_epochs': 25, 'dropout': 0.1, 'encoder_key': 'rel'},
    # trial #82  MAE 51.0303  CI_ERR 1.2833  score 52.3136
    {'num_encoder_decoder_layers': 8, 'decoder_output_dim': 31, 'hidden_size': 63, 'temporal_width_past': 4, 'temporal_width_future': 1, 'temporal_decoder_hidden': 48, 'temporal_hidden_size_past': 21, 'temporal_hidden_size_future': 28, 'lr': 2.4867027385576873e-05, 'n_epochs': 15, 'dropout': 0.25, 'encoder_key': 'rel_mon'},
]
# <<< TIDE_PARAMS_RT <<<

# >>> TIDE_PARAMS_DA >>>
# TIDE_PARAMS_DA — top 5 trials by 'mae_ci_da' score from study 'spp_west_da_tide_mae_ci_da' (score 8.015-10.320; metrics mae+ci_err). Managed by scripts/tune_parameters.py.
TIDE_PARAMS_DA = [
    # trial #63  MAE 7.6503  CI_ERR 0.3646  score 8.0149
    {'num_encoder_decoder_layers': 7, 'decoder_output_dim': 27, 'hidden_size': 54, 'temporal_width_past': 5, 'temporal_width_future': 2, 'temporal_decoder_hidden': 30, 'temporal_hidden_size_past': 25, 'temporal_hidden_size_future': 19, 'lr': 0.00011561955456882098, 'n_epochs': 22, 'dropout': 0.35, 'encoder_key': 'rel'},
    # trial #28  MAE 8.3964  CI_ERR 0.3021  score 8.6985
    {'num_encoder_decoder_layers': 7, 'decoder_output_dim': 9, 'hidden_size': 42, 'temporal_width_past': 2, 'temporal_width_future': 3, 'temporal_decoder_hidden': 7, 'temporal_hidden_size_past': 27, 'temporal_hidden_size_future': 12, 'lr': 3.767875471900317e-05, 'n_epochs': 29, 'dropout': 0.25, 'encoder_key': 'rel'},
    # trial #57  MAE 7.3585  CI_ERR 1.3438  score 8.7022
    {'num_encoder_decoder_layers': 7, 'decoder_output_dim': 30, 'hidden_size': 33, 'temporal_width_past': 4, 'temporal_width_future': 9, 'temporal_decoder_hidden': 44, 'temporal_hidden_size_past': 20, 'temporal_hidden_size_future': 32, 'lr': 0.0005669716262493834, 'n_epochs': 13, 'dropout': 0.1, 'encoder_key': 'rel'},
    # trial #97  MAE 6.7224  CI_ERR 2.4062  score 9.1286
    {'num_encoder_decoder_layers': 4, 'decoder_output_dim': 27, 'hidden_size': 60, 'temporal_width_past': 0, 'temporal_width_future': 9, 'temporal_decoder_hidden': 15, 'temporal_hidden_size_past': 12, 'temporal_hidden_size_future': 24, 'lr': 4.631920567277456e-05, 'n_epochs': 16, 'dropout': 0.5, 'encoder_key': 'rel'},
    # trial #36  MAE 7.7259  CI_ERR 2.5937  score 10.3197
    {'num_encoder_decoder_layers': 4, 'decoder_output_dim': 29, 'hidden_size': 45, 'temporal_width_past': 3, 'temporal_width_future': 2, 'temporal_decoder_hidden': 4, 'temporal_hidden_size_past': 11, 'temporal_hidden_size_future': 32, 'lr': 0.0001706879500768014, 'n_epochs': 7, 'dropout': 0.35, 'encoder_key': 'rel'},
]
# <<< TIDE_PARAMS_DA <<<

# Per-target tuned TiDE params. The retrain (and the ensemble dev notebook)
# read the active target's list; the Optuna study writes each target's slot.
TIDE_PARAMS_BY_TARGET = {"rt": TIDE_PARAMS_RT, "da": TIDE_PARAMS_DA}

# Back-compat single-target params = the default target's.
TIDE_PARAMS = TIDE_PARAMS_BY_TARGET[DEFAULT_TARGET]


# best tide model params from optuna experiment
TFT_PARAMS = [
    {
        "hidden_size": 20,
        "lstm_layers": 1,
        "num_attention_heads": 3,
        "lr": 0.000749,
        "n_epochs": 5,
        "dropout": 0.49,
        "full_attention": True,
        "encoder_key": "rel_mon",
    },
    {
        "hidden_size": 9,
        "lstm_layers": 2,
        "num_attention_heads": 4,
        "lr": 0.000412,
        "n_epochs": 6,
        "dropout": 0.44,
        "full_attention": False,
        "encoder_key": "rel_mon",
    },
    {
        "hidden_size": 8,
        "lstm_layers": 2,
        "num_attention_heads": 3,
        "lr": 0.000371,
        "n_epochs": 4,
        "dropout": 0.44999999999999996,
        "full_attention": False,
        "encoder_key": "rel",
    },
    {
        "hidden_size": 35,
        "lstm_layers": 2,
        "num_attention_heads": 1,
        "lr": 0.0007769999999999999,
        "n_epochs": 2,
        "dropout": 0.44,
        "full_attention": False,
        "encoder_key": "rel",
    },
    {
        "hidden_size": 47,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "lr": 0.00030900000000000003,
        "n_epochs": 6,
        "dropout": 0.39,
        "full_attention": True,
        "encoder_key": "rel_mon",
    },
]

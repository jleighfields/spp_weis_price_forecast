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
# Real-time (RT) tide params from the IM-only CRPS Optuna study (2026-07-06,
# study 'spp_west_tide', 100 trials): the top 5 trials by CRPS on the West
# holdout (14.05-14.13), one per TOP_N ensemble member.
TIDE_PARAMS_RT = [
    {
        "num_encoder_decoder_layers": 4,  # trial #85  CRPS 14.054
        "decoder_output_dim": 20,
        "hidden_size": 20,
        "temporal_width_past": 5,
        "temporal_width_future": 6,
        "temporal_decoder_hidden": 21,
        "temporal_hidden_size_past": 29,
        "temporal_hidden_size_future": 19,
        "lr": 0.0003269562607836474,
        "n_epochs": 17,
        "dropout": 0.45000000000000007,
        "encoder_key": "rel_mon",
    },
    {
        "num_encoder_decoder_layers": 4,  # trial #99  CRPS 14.071
        "decoder_output_dim": 18,
        "hidden_size": 53,
        "temporal_width_past": 5,
        "temporal_width_future": 8,
        "temporal_decoder_hidden": 23,
        "temporal_hidden_size_past": 25,
        "temporal_hidden_size_future": 25,
        "lr": 0.00031352894876833086,
        "n_epochs": 20,
        "dropout": 0.4,
        "encoder_key": "rel_mon_day",
    },
    {
        "num_encoder_decoder_layers": 6,  # trial #6  CRPS 14.079
        "decoder_output_dim": 17,
        "hidden_size": 20,
        "temporal_width_past": 1,
        "temporal_width_future": 7,
        "temporal_decoder_hidden": 21,
        "temporal_hidden_size_past": 22,
        "temporal_hidden_size_future": 25,
        "lr": 0.0006605081917306321,
        "n_epochs": 16,
        "dropout": 0.45000000000000007,
        "encoder_key": "rel",
    },
    {
        "num_encoder_decoder_layers": 5,  # trial #74  CRPS 14.106
        "decoder_output_dim": 19,
        "hidden_size": 9,
        "temporal_width_past": 6,
        "temporal_width_future": 7,
        "temporal_decoder_hidden": 27,
        "temporal_hidden_size_past": 30,
        "temporal_hidden_size_future": 22,
        "lr": 0.00026451904225915446,
        "n_epochs": 19,
        "dropout": 0.35,
        "encoder_key": "rel_mon",
    },
    {
        "num_encoder_decoder_layers": 6,  # trial #20  CRPS 14.129
        "decoder_output_dim": 18,
        "hidden_size": 25,
        "temporal_width_past": 3,
        "temporal_width_future": 1,
        "temporal_decoder_hidden": 21,
        "temporal_hidden_size_past": 16,
        "temporal_hidden_size_future": 19,
        "lr": 0.00019831794710668533,
        "n_epochs": 17,
        "dropout": 0.35,
        "encoder_key": "rel",
    },
]
# <<< TIDE_PARAMS_RT <<<

# >>> TIDE_PARAMS_DA >>>
# TIDE_PARAMS_DA — top 5 trials by CRPS from study 'spp_west_da_tide' (CRPS 3.810-4.066). Managed by scripts/tune_parameters.py.
TIDE_PARAMS_DA = [
    # trial #86  CRPS 3.8101
    {
        "num_encoder_decoder_layers": 2,
        "decoder_output_dim": 29,
        "hidden_size": 40,
        "temporal_width_past": 6,
        "temporal_width_future": 9,
        "temporal_decoder_hidden": 22,
        "temporal_hidden_size_past": 10,
        "temporal_hidden_size_future": 22,
        "lr": 0.0005532723190569155,
        "n_epochs": 24,
        "dropout": 0.5,
        "encoder_key": "rel_mon",
    },
    # trial #71  CRPS 3.9204
    {
        "num_encoder_decoder_layers": 1,
        "decoder_output_dim": 30,
        "hidden_size": 38,
        "temporal_width_past": 6,
        "temporal_width_future": 8,
        "temporal_decoder_hidden": 20,
        "temporal_hidden_size_past": 9,
        "temporal_hidden_size_future": 25,
        "lr": 0.00028061572090196113,
        "n_epochs": 28,
        "dropout": 0.5,
        "encoder_key": "rel_mon",
    },
    # trial #49  CRPS 3.9490
    {
        "num_encoder_decoder_layers": 1,
        "decoder_output_dim": 30,
        "hidden_size": 31,
        "temporal_width_past": 6,
        "temporal_width_future": 7,
        "temporal_decoder_hidden": 20,
        "temporal_hidden_size_past": 9,
        "temporal_hidden_size_future": 26,
        "lr": 0.00024763161328855314,
        "n_epochs": 25,
        "dropout": 0.5,
        "encoder_key": "rel_mon",
    },
    # trial #30  CRPS 4.0285
    {
        "num_encoder_decoder_layers": 1,
        "decoder_output_dim": 28,
        "hidden_size": 55,
        "temporal_width_past": 2,
        "temporal_width_future": 5,
        "temporal_decoder_hidden": 24,
        "temporal_hidden_size_past": 8,
        "temporal_hidden_size_future": 23,
        "lr": 0.00033338161882574656,
        "n_epochs": 19,
        "dropout": 0.45000000000000007,
        "encoder_key": "rel_mon",
    },
    # trial #85  CRPS 4.0657
    {
        "num_encoder_decoder_layers": 1,
        "decoder_output_dim": 30,
        "hidden_size": 40,
        "temporal_width_past": 6,
        "temporal_width_future": 9,
        "temporal_decoder_hidden": 23,
        "temporal_hidden_size_past": 10,
        "temporal_hidden_size_future": 22,
        "lr": 0.00013540542936833213,
        "n_epochs": 26,
        "dropout": 0.5,
        "encoder_key": "rel_mon",
    },
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

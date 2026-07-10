# Optuna hyperparameter tuning for SPP West (RTO West / Integrated
# Marketplace) nodal price forecast models.
#
# Supports TiDE, TSMixer, and TFT model types. Runs single-objective
# optimization on CRPS (a proper score that captures point accuracy and
# interval calibration/sharpness at once), matching the primary metric of the
# evaluation harness in src/evaluation.py; MAE is logged per trial as a
# diagnostic user_attr.
#
# Usage:
#   Interactive: marimo edit notebooks/model_training/model.py

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    import pathlib
    import shutil
    import optuna

    return optuna, os, shutil


@app.cell
def _():
    # MODEL_TYPE = 'ts_mixer'
    MODEL_TYPE = "tide"
    # MODEL_TYPE = 'tft'

    RUN_EXP = True
    NUM_TRIALS = 100

    # Clip LMP to the 0.25% / 99.75% quantiles before training/scoring.
    # A deliberate, tested win on WEIS; re-validate on the spikier IM
    # distribution by running the study once True and once False and
    # comparing MAE/CRPS and CI coverage/tail error on the harness.
    CLIP_OUTLIERS = True

    REMOVE_PRIOR_MODELS = True
    TEST_BUILD_BACKTEST = False
    return (
        CLIP_OUTLIERS,
        MODEL_TYPE,
        NUM_TRIALS,
        REMOVE_PRIOR_MODELS,
        RUN_EXP,
        TEST_BUILD_BACKTEST,
    )


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import polars as pl
    import torch
    import pathlib as _pathlib

    from darts.metrics import mae, mcrps
    from darts.models import TFTModel, TiDEModel, TSMixerModel, NaiveEnsembleModel

    import warnings

    warnings.filterwarnings("ignore")

    from dotenv import load_dotenv

    load_dotenv(override=True)

    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Add project root and src/ to sys.path — src/ so modules like
    # data_engineering can `import parameters` directly (matches model_retrain.py).
    import sys as _sys

    _root = _pathlib.Path(__file__).resolve().parent.parent.parent
    for _p in [str(_root), str(_root / "src")]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    return (
        NaiveEnsembleModel,
        TFTModel,
        TSMixerModel,
        TiDEModel,
        log,
        mae,
        mcrps,
        np,
        pd,
        pl,
        torch,
    )


@app.cell
def _():
    from optuna.integration import PyTorchLightningPruningCallback
    from optuna.visualization import (
        plot_optimization_history,
        plot_contour,
        plot_param_importances,
    )

    return (
        PyTorchLightningPruningCallback,
        plot_contour,
        plot_optimization_history,
        plot_param_importances,
    )


@app.cell
def _():
    import src.data_engineering as de
    from src import parameters
    from src import plotting
    from src.modeling import (
        get_ci_err,
        build_fit_tsmixerx,
        build_fit_tide,
        build_fit_tft,
        log_pretty,
    )

    return (
        build_fit_tft,
        build_fit_tide,
        build_fit_tsmixerx,
        de,
        get_ci_err,
        log_pretty,
        parameters,
        plotting,
    )


@app.cell
def _(log, os, parameters):
    # Forecast target to tune (parameters.TARGETS); DA is the default. Set the
    # TARGET env var to tune a different target (e.g. TARGET=rt).
    TARGET = os.environ.get("TARGET", parameters.DEFAULT_TARGET)
    MODEL_NAME = parameters.TARGETS[TARGET]["model_name"]
    log.info(f"TARGET: {TARGET}  MODEL_NAME: {MODEL_NAME}")
    log.info(f"FORECAST_HORIZON: {parameters.FORECAST_HORIZON}")
    log.info(f"INPUT_CHUNK_LENGTH: {parameters.INPUT_CHUNK_LENGTH}")
    return MODEL_NAME, TARGET


@app.cell
def _(torch):
    torch.set_float32_matmul_precision("medium")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data prep""")
    return


@app.cell
def _(TARGET, de):
    con = de.create_database(target=TARGET)
    return (con,)


@app.cell
def _(con):
    con.execute("SHOW TABLES").fetchall()
    return


@app.cell
def _(con):
    con.execute("SELECT * FROM lmp LIMIT 5").pl()
    return


@app.cell
def _(con, de):
    lmp = de.prep_lmp(con)
    lmp
    return (lmp,)


@app.cell
def _(lmp, pl):
    lmp.select(pl.col("LMP").min()), lmp.select(pl.col("LMP").max())
    return


@app.cell
def _(lmp):
    lmp_df = lmp.to_pandas().rename(
        columns={"LMP": "LMP_HOURLY", "unique_id": "node", "timestamp_mst": "time"}
    )
    return (lmp_df,)


@app.cell
def _(con, de):
    mtrf = de.prep_mtrf(con)
    mtrf
    return


@app.cell
def _(con, de):
    mtlf = de.prep_mtlf(con)
    mtlf
    return


@app.cell
def _(CLIP_OUTLIERS, con, de):
    all_df = de.prep_all_df(con, clip_outliers=CLIP_OUTLIERS)
    all_df
    return (all_df,)


@app.cell
def _(all_df, pl):
    all_df.select(pl.col("LMP").min()), all_df.select(pl.col("LMP").max())
    return


@app.cell
def _(all_df, de):
    all_df_pd = de.all_df_to_pandas(all_df)
    all_df_pd
    return (all_df_pd,)


@app.cell
def _(all_df_pd):
    all_df_pd.info()
    return


@app.cell
def _(all_df_pd):
    all_df_pd.reset_index()[["unique_id", "timestamp_mst"]].duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prep model training data""")
    return


@app.cell
def _(CLIP_OUTLIERS, con, de):
    lmp_all, train_all, test_all, train_test_all = de.get_train_test_all(
        con, clip_outliers=CLIP_OUTLIERS
    )
    return lmp_all, test_all, train_all, train_test_all


@app.cell
def _(de, lmp_all):
    all_series = de.get_series(lmp_all)
    all_series[0].plot()
    return (all_series,)


@app.cell
def _(de, train_all):
    train_series = de.get_series(train_all)
    train_series[0].plot()
    return (train_series,)


@app.cell
def _(de, test_all):
    test_series = de.get_series(test_all)
    test_series[0].plot()
    return (test_series,)


@app.cell
def _(all_df_pd, de):
    futr_cov = de.get_futr_cov(all_df_pd)
    futr_cov[0].plot()
    return (futr_cov,)


@app.cell
def _(all_df_pd, de):
    past_cov = de.get_past_cov(all_df_pd)
    past_cov[0].plot()
    return (past_cov,)


@app.cell
def _(con):
    con.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Set up hyperparameter tuning study

    https://unit8co.github.io/darts/examples/17-hyperparameter-optimization.html?highlight=optuna
    """
    )
    return


@app.cell
def _(
    TEST_BUILD_BACKTEST,
    build_fit_tide,
    futr_cov,
    get_ci_err,
    log,
    mae,
    np,
    parameters,
    past_cov,
    test_series,
    train_series,
):
    if TEST_BUILD_BACKTEST:
        _model = build_fit_tide(
            series=train_series,
            val_series=test_series,
            future_covariates=futr_cov,
            past_covariates=past_cov,
            n_epochs=1,
        )
        log.info(f"model.MODEL_TYPE: {_model.MODEL_TYPE}")
        _err = _model.backtest(
            series=test_series,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            retrain=False,
            forecast_horizon=parameters.FORECAST_HORIZON,
            stride=25,
            metric=[mae],
            verbose=False,
        )
        log.info(f"err_metric: {np.mean(_err)}")
        _preds = _model.predict(
            series=train_series,
            n=parameters.FORECAST_HORIZON,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            num_samples=200,
        )
        _errs = np.mean(mae(test_series, _preds, n_jobs=-1, verbose=True))
        log.info(f"errs: {_errs}")
        _ci_err = get_ci_err(test_series, _preds)
        log.info(f"test_ci_err mean: {np.mean(_ci_err)}")
        _val = _model.backtest(
            series=test_series,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            retrain=False,
            forecast_horizon=parameters.FORECAST_HORIZON,
            stride=25,
            metric=[mae, get_ci_err],
            verbose=False,
            num_samples=200,
        )
        log.info(f"val_backtest: {_val}")
    return


@app.cell
def _(mae, mcrps, np, parameters):
    def score_trial_crps(model, trial, test_series, past_cov, futr_cov):
        """Backtest one trial's model on the West holdout and return CRPS.

        The single study objective, shared by all model types. ``test_series``
        is a list of nodes, so ``backtest`` returns one ``[crps, mae]`` row per
        node (each reduced over that node's windows); average across nodes. MAE
        is stored as a diagnostic ``user_attr``. ``num_samples`` makes the
        forecast stochastic so CRPS is meaningful (it degenerates to MAE on a
        point forecast).
        """
        val_backtest = model.backtest(
            series=test_series,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            retrain=False,
            forecast_horizon=parameters.FORECAST_HORIZON,
            stride=24,  # daily origins over the hourly series
            metric=[mcrps, mae],
            verbose=False,
            num_samples=200,
            last_points_only=False,
        )
        crps = np.mean([e[0] for e in val_backtest])
        trial.set_user_attr("mae", float(np.mean([e[1] for e in val_backtest])))
        return float(crps) if np.isfinite(crps) else float("inf")

    return (score_trial_crps,)


@app.cell
def _(
    PyTorchLightningPruningCallback,
    TRIAL_MODEL_DIR,
    build_fit_tsmixerx,
    futr_cov,
    past_cov,
    score_trial_crps,
    test_series,
    train_series,
):
    def objective_tsmixer(trial):
        callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=2)
        ff_size = trial.suggest_int("ff_size", 16, 256, step=2)
        num_blocks = trial.suggest_int("num_blocks", 4, 12)
        lr = trial.suggest_float("lr", 1e-5, 1e-4, step=1e-6)
        n_epochs = trial.suggest_int("n_epochs", 4, 12)
        dropout = trial.suggest_float("dropout", 0.4, 0.50, step=0.01)
        activation = trial.suggest_categorical("activation", ["ELU", "SELU"])
        encoder_key = trial.suggest_categorical(
            "encoder_key", ["rel", "rel_mon", "rel_mon_day"]
        )

        model = build_fit_tsmixerx(
            series=train_series,
            val_series=test_series,
            future_covariates=futr_cov,
            past_covariates=past_cov,
            hidden_size=hidden_size,
            ff_size=ff_size,
            num_blocks=num_blocks,
            lr=lr,
            n_epochs=n_epochs,
            dropout=dropout,
            encoder_key=encoder_key,
            activation=activation,
            callbacks=callback,
            model_id=f"{trial.number:03}",
            log_tensorboard=False,
        )

        model_path = f"{TRIAL_MODEL_DIR}/model_{trial.number}"
        trial.set_user_attr("model_path", model_path)
        model.save(model_path)

        return score_trial_crps(model, trial, test_series, past_cov, futr_cov)

    return (objective_tsmixer,)


@app.cell
def _(futr_cov, past_cov):
    n_futr = futr_cov[0].shape[1]
    n_past = past_cov[0].shape[1]
    n_futr, n_past
    return n_futr, n_past


@app.cell
def _(
    PyTorchLightningPruningCallback,
    TRIAL_MODEL_DIR,
    build_fit_tide,
    futr_cov,
    n_futr,
    n_past,
    past_cov,
    score_trial_crps,
    test_series,
    train_series,
):
    def objective_tide(trial):
        callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        num_encoder_decoder_layers = trial.suggest_int(
            "num_encoder_decoder_layers", 1, 8
        )
        decoder_output_dim = trial.suggest_int("decoder_output_dim", 8, 32)
        hidden_size = trial.suggest_int("hidden_size", 8, 64, 1)
        temporal_width_past = trial.suggest_int("temporal_width_past", 0, n_past)
        temporal_width_future = trial.suggest_int("temporal_width_future", 0, n_futr)
        temporal_decoder_hidden = trial.suggest_int("temporal_decoder_hidden", 4, 64, 1)
        temporal_hidden_size_past = trial.suggest_int(
            "temporal_hidden_size_past", 8, 32, 1
        )
        temporal_hidden_size_future = trial.suggest_int(
            "temporal_hidden_size_future", 8, 32, 1
        )
        # lr and dropout widened from the WEIS-era ranges (lr 1e-5..5e-5,
        # dropout 0.35..0.5) for the spikier, shorter IM series: a faster
        # learning rate (log scale) and lighter regularization. n_epochs stays
        # 6..20 — the tuned value is baked into TIDE_PARAMS and drives every
        # production retrain, so we don't want a 60-epoch champion.
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        n_epochs = trial.suggest_int("n_epochs", 6, 20)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        encoder_key = trial.suggest_categorical(
            "encoder_key", ["rel", "rel_mon", "rel_mon_day"]
        )

        model = build_fit_tide(
            series=train_series,
            val_series=test_series,
            future_covariates=futr_cov,
            past_covariates=past_cov,
            num_encoder_decoder_layers=num_encoder_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_width_past=temporal_width_past,
            temporal_width_future=temporal_width_future,
            temporal_decoder_hidden=temporal_decoder_hidden,
            temporal_hidden_size_past=temporal_hidden_size_past,
            temporal_hidden_size_future=temporal_hidden_size_future,
            lr=lr,
            n_epochs=n_epochs,
            dropout=dropout,
            encoder_key=encoder_key,
            callbacks=callback,
            model_id=f"{trial.number:03}",
            log_tensorboard=False,
        )

        model_path = f"{TRIAL_MODEL_DIR}/model_{trial.number}"
        trial.set_user_attr("model_path", model_path)
        model.save(model_path)

        return score_trial_crps(model, trial, test_series, past_cov, futr_cov)

    return (objective_tide,)


@app.cell
def _(
    PyTorchLightningPruningCallback,
    TRIAL_MODEL_DIR,
    build_fit_tft,
    futr_cov,
    past_cov,
    score_trial_crps,
    test_series,
    train_series,
):
    def objective_tft(trial):
        callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        hidden_size = trial.suggest_int("hidden_size", 8, 32)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
        num_attention_heads = trial.suggest_int("num_attention_heads", 1, 2)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, step=1e-6)
        n_epochs = trial.suggest_int("n_epochs", 2, 6)
        dropout = trial.suggest_float("dropout", 0.3, 0.5, step=0.01)
        full_attention = trial.suggest_categorical("full_attention", [False, True])
        encoder_key = trial.suggest_categorical(
            "encoder_key", ["rel", "rel_mon", "rel_mon_day"]
        )

        model = build_fit_tft(
            series=train_series,
            val_series=test_series,
            future_covariates=futr_cov,
            past_covariates=past_cov,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            lr=lr,
            n_epochs=n_epochs,
            dropout=dropout,
            encoder_key=encoder_key,
            full_attention=full_attention,
            batch_size=64,
            callbacks=callback,
            model_id=f"{trial.number:03}",
            log_tensorboard=False,
        )

        model_path = f"{TRIAL_MODEL_DIR}/model_{trial.number}"
        trial.set_user_attr("model_path", model_path)
        model.save(model_path)

        return score_trial_crps(model, trial, test_series, past_cov, futr_cov)

    return (objective_tft,)


@app.cell
def _(MODEL_TYPE, os):
    os.makedirs(f"study_csv/{MODEL_TYPE}", exist_ok=True)
    return


@app.cell
def _(MODEL_TYPE, log, log_pretty, target_names):
    def print_callback(study, trial):
        best = study.best_trial
        print("\n" + "*" * 30, flush=True)
        log.info(f"\nTrial: {trial.number} Current {target_names[0]}: {trial.value}")
        log.info(f"Current params: \n{log_pretty(trial.params)}")
        log.info(
            f"Best {target_names[0]}: Num: {best.number}, {best.value}, "
            f"Best params: \n{log_pretty(best.params)}"
        )
        study.trials_dataframe().to_csv(
            f"study_csv/{MODEL_TYPE}/{trial.number:03}.csv"
        )

    return (print_callback,)


@app.cell
def _():
    target_names = ["CRPS"]
    return (target_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Start Experiment""")
    return


@app.cell
def _(MODEL_NAME, MODEL_TYPE, REMOVE_PRIOR_MODELS, optuna, os, shutil):
    TRIAL_MODEL_DIR = f"optuna/{MODEL_TYPE}"
    MODEL_CHECKPOINT_DIR = f"model_checkpoints/{MODEL_TYPE}_model"

    if REMOVE_PRIOR_MODELS:
        try:
            optuna.delete_study(
                study_name=f"{MODEL_NAME}_{MODEL_TYPE}",
                storage="sqlite:///spp_trials.db",
            )
            shutil.rmtree(TRIAL_MODEL_DIR)
            shutil.rmtree(MODEL_CHECKPOINT_DIR)
        except Exception:
            pass

    os.makedirs(TRIAL_MODEL_DIR, exist_ok=True)
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    return (TRIAL_MODEL_DIR,)


@app.cell
def _(MODEL_TYPE, objective_tft, objective_tide, objective_tsmixer):
    if MODEL_TYPE == "tft":
        objective_func = objective_tft
    elif MODEL_TYPE == "tide":
        objective_func = objective_tide
    elif MODEL_TYPE == "ts_mixer":
        objective_func = objective_tsmixer
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
    return (objective_func,)


@app.cell
def _(MODEL_NAME, MODEL_TYPE):
    study_name = f"{MODEL_NAME}_{MODEL_TYPE}"
    return (study_name,)


@app.cell
def _(optuna, study_name):
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///spp_trials.db",
        study_name=study_name,
        load_if_exists=True,
    )
    return (study,)


@app.cell
def _(NUM_TRIALS, RUN_EXP, objective_func, print_callback, study):
    if RUN_EXP:
        study.optimize(
            objective_func, n_trials=NUM_TRIALS, callbacks=[print_callback]
        )
    return


@app.cell
def _(plot_optimization_history, study):
    plot_optimization_history(study).show()
    return


@app.cell
def _(plot_contour, study):
    plot_contour(study, params=["lr", "n_epochs"]).show()
    return


@app.cell
def _(plot_param_importances, study):
    plot_param_importances(study)
    return


@app.cell
def _(log, log_pretty, study):
    _best = study.best_trial
    log.info(f"Best number: {_best.number}")
    log.info(f"Best value: {_best.value}")
    log.info(f"Best params: \n{log_pretty(_best.params)}")
    return


@app.cell
def _(study):
    study.trials_dataframe().to_csv("study_csv/test.csv")
    return


@app.cell
def _(np, optuna, pd):
    def get_best_trials(
        study_name: str,
        storage: str = "sqlite:///spp_trials.db",
        n_results: int = 5,
    ) -> pd.DataFrame:
        _study = optuna.load_study(study_name=study_name, storage=storage)
        trials = pd.DataFrame(
            [
                {
                    "number": s.number,
                    "value": s.value if s.value is not None else np.nan,
                    "params": s.params,
                    "model_path": s.user_attrs.get("model_path"),
                }
                for s in _study.trials
            ]
        )
        trials = trials[~trials.params.duplicated()]
        return trials.sort_values("value").head(n_results)

    return (get_best_trials,)


@app.cell
def _(get_best_trials, study_name):
    best_trials = get_best_trials(study_name, n_results=5)
    best_trials
    return (best_trials,)


@app.cell
def _(best_trials):
    [p for p in best_trials.params]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Create ensemble from best models""")
    return


@app.cell
def _(TFTModel, TSMixerModel, TiDEModel, best_trials, torch):
    # weights_only=False: torch 2.6+ defaults torch.load to weights_only=True,
    # which refuses to unpickle Darts' QuantileRegression likelihood in the
    # Lightning checkpoint. These are our own trial checkpoints written this
    # run to a local dir (trusted source), so full unpickling is safe.
    forecasting_models = []
    for _m in best_trials.model_path:
        if "ts_mixer" in _m.lower():
            forecasting_models += [
                TSMixerModel.load(_m, map_location=torch.device("cpu"), weights_only=False)
            ]
        elif "tide" in _m.lower():
            forecasting_models += [
                TiDEModel.load(_m, map_location=torch.device("cpu"), weights_only=False)
            ]
        elif "tft" in _m.lower():
            forecasting_models += [
                TFTModel.load(_m, map_location=torch.device("cpu"), weights_only=False)
            ]
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {_m}")
    return (forecasting_models,)


@app.cell
def _(NaiveEnsembleModel, forecasting_models):
    loaded_model = NaiveEnsembleModel(
        forecasting_models=forecasting_models, train_forecasting_models=False
    )
    return (loaded_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plot test predictions""")
    return


@app.cell
def _(all_series):
    plot_ind = 3
    plot_series = all_series[plot_ind]
    return plot_ind, plot_series


@app.cell
def _(plot_series):
    plot_series.plot()
    return


@app.cell
def _(pd, plot_ind, test_series):
    plot_end_times = pd.date_range(
        end=test_series[plot_ind].end_time(), periods=10, freq="d"
    )
    plot_end_times
    return (plot_end_times,)


@app.cell
def _(
    futr_cov,
    lmp_df,
    loaded_model,
    log,
    parameters,
    past_cov,
    plot_end_times,
    plot_series,
    plotting,
):
    for _pet in plot_end_times:
        log.info(f"plot_end_time: {_pet}")
        _node_name = plot_series.static_covariates.unique_id.LMP
        _ns = plot_series.drop_after(_pet)
        log.info(f"node_series.end_time(): {_ns.end_time()}")
        _fc = futr_cov[0]
        _pc = past_cov[0]

        _preds = loaded_model.predict(
            series=_ns,
            past_covariates=_pc,
            future_covariates=_fc,
            n=parameters.FORECAST_HORIZON,
            num_samples=500,
        )

        _cov_df = (
            _fc.to_dataframe()
            .reset_index()
            .rename(columns={"timestamp_mst": "time", "re_ratio": "Ratio"})
        )
        _plot_df = plotting.get_plot_df(_preds, _cov_df, lmp_df, _node_name)
        _plot_df.rename(columns={"mean": "mean_fcast"}, inplace=True)
        plotting.plotly_forecast(_plot_df, _node_name, show_fig=True)
    return


@app.cell
def _(MODEL_NAME, all_series, de, log, parameters, torch):
    # Record the training config for this study run (the same schema the retrain
    # writes to R2), documenting exactly which covariates / nodes / quantiles
    # were tuned against. The study does not save a servable model, so this is
    # logged for provenance rather than uploaded — the winning params get
    # retrained + saved (with this config) by model_retrain.py.
    import darts as _darts
    import node_list as _node_list
    import utils as _utils

    _study_model_types = _utils.active_model_types(
        parameters.USE_TIDE, parameters.USE_TSMIXER, parameters.USE_TFT
    )
    _study_cfg = _utils.build_training_config(
        train_timestamp=str(all_series[0].end_time()),
        future_covariates=de.FUTR_COLS,
        past_covariates=de.PAST_COLS,
        nodes=_node_list.MODEL_APP_NODES,
        quantiles=parameters.QUANTILES,
        model_name=MODEL_NAME,
        model_types=_study_model_types,
        forecast_horizon=parameters.FORECAST_HORIZON,
        input_chunk_length=parameters.INPUT_CHUNK_LENGTH,
        train_start=str(all_series[0].start_time()),
        train_end=str(all_series[0].end_time()),
        darts_version=_darts.__version__,
        torch_version=torch.__version__,
    )
    log.info(f"study training_config: {_study_cfg}")
    return


if __name__ == "__main__":
    app.run()

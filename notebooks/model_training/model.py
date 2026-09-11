# Optuna hyperparameter tuning for SPP West (RTO West / Integrated
# Marketplace) nodal price forecast models.
#
# Supports TiDE, TSMixer, and TFT model types. The objective is chosen by the
# OBJECTIVE_MODE env var against the table in src/selection.py (default
# 'mae_ci': a two-objective study on MAE and weighted prediction-interval
# coverage error). Whichever metrics the active mode does not optimize are
# recorded per trial as user_attrs, so every trial carries MAE, CRPS and
# coverage error at all diagnostic bands and can be re-ranked under another
# mode without re-running the study.
#
# Usage:
#   Interactive: marimo edit notebooks/model_training/model.py
#   Another objective: OBJECTIVE_MODE=crps marimo edit notebooks/model_training/model.py

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
def _(os):
    # MODEL_TYPE = 'ts_mixer'
    MODEL_TYPE = os.environ.get("MODEL_TYPE", "tide")
    # MODEL_TYPE = 'tft'

    RUN_EXP = True
    # Env-overridable so a smoke run (NUM_TRIALS=2) needs no edit to tracked
    # code — same pattern as TARGET / OBJECTIVE_MODE.
    NUM_TRIALS = int(os.environ.get("NUM_TRIALS", 100))

    # Clip LMP to the 0.25% / 99.75% quantiles before training/scoring.
    # A deliberate, tested win on WEIS; re-validate on the spikier IM
    # distribution by running the study once True and once False and
    # comparing MAE/CRPS and CI coverage/tail error on the harness.
    CLIP_OUTLIERS = True

    REMOVE_PRIOR_MODELS = True
    return (
        CLIP_OUTLIERS,
        MODEL_TYPE,
        NUM_TRIALS,
        REMOVE_PRIOR_MODELS,
        RUN_EXP,
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
        plot_pareto_front,
    )

    return (
        PyTorchLightningPruningCallback,
        plot_contour,
        plot_optimization_history,
        plot_param_importances,
        plot_pareto_front,
    )


@app.cell
def _():
    import src.data_engineering as de
    from src import parameters
    from src import plotting
    from src import selection
    from src.modeling import (
        coverage_metric,
        build_fit_tsmixerx,
        build_fit_tide,
        build_fit_tft,
        log_pretty,
    )

    return (
        build_fit_tft,
        build_fit_tide,
        build_fit_tsmixerx,
        coverage_metric,
        de,
        log_pretty,
        parameters,
        plotting,
        selection,
    )


@app.cell
def _(log, os, parameters, selection):
    # Forecast target to tune (parameters.TARGETS); DA is the default. Set the
    # TARGET env var to tune a different target (e.g. TARGET=rt).
    TARGET = os.environ.get("TARGET", parameters.DEFAULT_TARGET)
    MODEL_NAME = parameters.TARGETS[TARGET]["model_name"]

    # How trials are ranked (selection.OBJECTIVES). resolve_mode rejects an
    # unknown name rather than falling back, so a typo'd OBJECTIVE_MODE cannot
    # quietly tune against an objective nobody chose.
    OBJECTIVE_MODE = selection.mode_name_from_env()
    MODE = selection.resolve_mode(OBJECTIVE_MODE)

    log.info(f"TARGET: {TARGET}  MODEL_NAME: {MODEL_NAME}")
    log.info(f"OBJECTIVE_MODE: {OBJECTIVE_MODE}  metrics: {MODE['metrics']}")
    log.info(f"  bands: {MODE['intervals']}  weights: {MODE['scalers']}")
    log.info(f"FORECAST_HORIZON: {parameters.FORECAST_HORIZON}")
    log.info(f"INPUT_CHUNK_LENGTH: {parameters.INPUT_CHUNK_LENGTH}")
    return MODE, MODEL_NAME, OBJECTIVE_MODE, TARGET


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
def _(MODE, coverage_metric, mae, mcrps, np, parameters, selection):
    def score_trial(model, trial, test_series, past_cov, futr_cov):
        """Backtest one trial and return the active mode's objective values.

        One backtest produces every metric: MAE, CRPS, and realized coverage
        at each band in ``selection.DIAGNOSTIC_BANDS``, from which each band's
        coverage error is derived. The mode picks which of those become Optuna
        objectives; the rest are stored as ``user_attrs``, so a study run under
        one mode can be re-ranked under another without re-running it. The
        extra bands cost arithmetic, not forecasts — Darts scores every metric
        against the same flattened forecasts.

        Both the raw coverage and the error are recorded per band. The error is
        unsigned, so on its own it cannot say whether a band over- or
        under-covers; the raw coverage is what makes a miscalibrated sweep
        diagnosable while it runs.

        ``test_series`` is a list of nodes, so ``backtest`` returns one row of
        metrics per node (each reduced over that node's windows); average
        across nodes. ``num_samples`` makes the forecast stochastic so the
        probabilistic metrics are meaningful (CRPS degenerates to MAE on a
        point forecast, and every band's coverage would be 0 or 1).

        Args:
            model: The fitted candidate model for this trial.
            trial: The Optuna trial, used to record the non-optimized metrics
                as ``user_attrs``.
            test_series: Per-node holdout target series to backtest on.
            past_cov: Per-node past covariates, aligned with ``test_series``.
            futr_cov: Per-node future covariates, aligned with ``test_series``.

        Returns:
            The mode's objective values in ``MODE['metrics']`` order — a bare
            float for a single-objective mode, a tuple otherwise. A non-finite
            value becomes ``inf`` so the trial loses rather than poisoning the
            study.
        """
        band_metrics = [coverage_metric(b) for b in selection.DIAGNOSTIC_BANDS]
        val_backtest = model.backtest(
            series=test_series,
            past_covariates=past_cov,
            future_covariates=futr_cov,
            retrain=False,
            forecast_horizon=parameters.FORECAST_HORIZON,
            stride=24,  # daily origins over the hourly series
            metric=[mae, mcrps, *band_metrics],
            verbose=False,
            num_samples=200,
            last_points_only=False,
        )
        # Metric columns come back in the order they were passed above.
        names = ["mae", "crps"] + [
            selection.coverage_label(b) for b in selection.DIAGNOSTIC_BANDS
        ]
        scored = {
            name: float(np.mean([row[i] for row in val_backtest]))
            for i, name in enumerate(names)
        }
        # Derive each band's (unsigned) error from its realized coverage, then
        # weight the mode's own bands with selection.weight_ci_errs — the same
        # step the promote gate uses, so the study cannot optimize a
        # differently-weighted number than the gate decides on.
        band_errs = {
            b: selection.coverage_error(scored[selection.coverage_label(b)], b)
            for b in selection.DIAGNOSTIC_BANDS
        }
        scored.update(
            {selection.band_label(b): err for b, err in band_errs.items()}
        )
        scored["ci_err"] = selection.weight_ci_errs(band_errs, MODE)

        # Whatever the mode does not optimize is recorded as a diagnostic.
        for name, value in scored.items():
            if name not in MODE["metrics"]:
                trial.set_user_attr(name, value)

        values = [
            scored[name] if np.isfinite(scored[name]) else float("inf")
            for name in MODE["metrics"]
        ]
        return values[0] if len(values) == 1 else tuple(values)

    return (score_trial,)


@app.cell
def _(
    PyTorchLightningPruningCallback,
    TRIAL_MODEL_DIR,
    build_fit_tsmixerx,
    futr_cov,
    past_cov,
    score_trial,
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

        return score_trial(model, trial, test_series, past_cov, futr_cov)

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
    score_trial,
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
        # learning rate (log scale) and lighter regularization. n_epochs 6..30 —
        # widened from 6..20 because the DA study's best trials pinned at the 20
        # ceiling, so give them headroom (still well short of a 60-epoch champion).
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        n_epochs = trial.suggest_int("n_epochs", 6, 30)
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

        return score_trial(model, trial, test_series, past_cov, futr_cov)

    return (objective_tide,)


@app.cell
def _(
    PyTorchLightningPruningCallback,
    TRIAL_MODEL_DIR,
    build_fit_tft,
    futr_cov,
    past_cov,
    score_trial,
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

        return score_trial(model, trial, test_series, past_cov, futr_cov)

    return (objective_tft,)


@app.cell
def _(MODEL_TYPE, os):
    os.makedirs(f"study_csv/{MODEL_TYPE}", exist_ok=True)
    return


@app.cell
def _(MODE, MODEL_TYPE, log, log_pretty, selection, target_names):
    def print_callback(study, trial):
        print("\n" + "*" * 30, flush=True)
        log.info(f"\nTrial: {trial.number} Current {target_names}: {trial.values}")
        log.info(f"Current params: \n{log_pretty(trial.params)}")

        if len(MODE["metrics"]) == 1:
            # Single-objective: study.best_trial is well defined.
            _best = study.best_trial
            log.info(
                f"Best {target_names[0]}: Num: {_best.number}, {_best.values}, "
                f"Best params: \n{log_pretty(_best.params)}"
            )
        else:
            # Multi-objective: there is no single best trial, so report the
            # best of each objective plus the best composite — the composite
            # is what actually picks the ensemble, the other two show which
            # term is driving it.
            _front = study.best_trials
            for _i, _name in enumerate(target_names):
                _b = min(_front, key=lambda t, i=_i: t.values[i])
                log.info(f"Best {_name}: Num: {_b.number}, {_b.values}")
            _b = min(_front, key=lambda t: selection.selection_score(t.values, MODE))
            log.info(
                f"Best composite: Num: {_b.number}, {_b.values}, "
                f"score {selection.selection_score(_b.values, MODE):.4f}, "
                f"Best params: \n{log_pretty(_b.params)}"
            )

        study.trials_dataframe().to_csv(
            f"study_csv/{MODEL_TYPE}/{trial.number:03}.csv"
        )

    return (print_callback,)


@app.cell
def _(MODE):
    target_names = [m.upper() for m in MODE["metrics"]]
    return (target_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Start Experiment""")
    return


@app.cell
def _(MODEL_TYPE, REMOVE_PRIOR_MODELS, os, shutil):
    # Scratch dirs for trial checkpoints. Cleared best-effort — they are absent
    # on a first run. Resetting the Optuna *study* is deliberately NOT done
    # here: it has to happen immediately before create_study, in that cell.
    TRIAL_MODEL_DIR = f"optuna/{MODEL_TYPE}"
    MODEL_CHECKPOINT_DIR = f"model_checkpoints/{MODEL_TYPE}_model"

    if REMOVE_PRIOR_MODELS:
        for _dir in (TRIAL_MODEL_DIR, MODEL_CHECKPOINT_DIR):
            shutil.rmtree(_dir, ignore_errors=True)

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
def _(MODEL_NAME, MODEL_TYPE, OBJECTIVE_MODE, selection):
    # One home for this format (selection.study_name): the bake CLI builds the
    # same name, and if the two drift it silently reads a different study.
    study_name = selection.study_name(MODEL_NAME, MODEL_TYPE, OBJECTIVE_MODE)
    return (study_name,)


@app.cell
def _(MODE, REMOVE_PRIOR_MODELS, log, optuna, study_name):
    # Reset lives here, not in the scratch-dir cell, so it cannot be reordered
    # to run AFTER create_study and silently delete the study just created —
    # marimo is a DAG, and these two cells would otherwise have no dependency
    # between them. Delete by `study_name` (selection.study_name): deleting by
    # any other name is a silent no-op that leaves the prior trials in place,
    # so the sweep appends to trials scored by different code or data.
    if REMOVE_PRIOR_MODELS:
        try:
            optuna.delete_study(
                study_name=study_name, storage="sqlite:///spp_trials.db"
            )
            log.info(f"removed prior study {study_name!r}")
        except KeyError:
            log.info(f"no prior study {study_name!r} to remove")

    study = optuna.create_study(
        directions=["minimize"] * len(MODE["metrics"]),
        storage="sqlite:///spp_trials.db",
        study_name=study_name,
        load_if_exists=True,
    )

    # create_study(load_if_exists=True) SILENTLY ignores `directions` when the
    # stored study disagrees — it returns the study with its original objective
    # count. Every trial then fails with "The number of the values N did not
    # match the number of the objectives M", and study.optimize does not
    # propagate that: it logs a warning per trial and marks them FAIL. The
    # sweep would run for hours and exit cleanly with nothing usable. The
    # mode-suffixed study name should make this unreachable; the guard stays
    # because the failure it catches is silent.
    if len(study.directions) != len(MODE["metrics"]):
        raise ValueError(
            f"study {study_name!r} is stored with {len(study.directions)} "
            f"objective(s) but mode expects {len(MODE['metrics'])} "
            f"{MODE['metrics']}. Set REMOVE_PRIOR_MODELS=True to start it "
            "fresh, or pick a different OBJECTIVE_MODE."
        )
    log.info(f"study {study_name!r}: {len(study.trials)} existing trial(s)")
    return (study,)


@app.cell
def _(NUM_TRIALS, RUN_EXP, objective_func, print_callback, study):
    if RUN_EXP:
        study.optimize(
            objective_func, n_trials=NUM_TRIALS, callbacks=[print_callback]
        )
    return


@app.cell
def _(MODE, study, target_names):
    def show_per_objective(plot_fn, **kwargs):
        """Render an Optuna plot once per objective.

        Optuna's visualizations take a single scalar target, so under a
        multi-objective study each objective needs its own figure with an
        explicit ``target``; a single-objective study needs none. The default
        ``idx`` in the lambda binds the loop variable at definition time —
        without it every figure would plot the last objective.
        """
        if len(MODE["metrics"]) == 1:
            plot_fn(study, **kwargs).show()
            return
        for i, name in enumerate(target_names):
            plot_fn(
                study,
                target=lambda t, idx=i: t.values[idx],
                target_name=name,
                **kwargs,
            ).show()

    return (show_per_objective,)


@app.cell
def _(plot_optimization_history, show_per_objective):
    show_per_objective(plot_optimization_history)
    return


@app.cell
def _(plot_contour, show_per_objective):
    show_per_objective(plot_contour, params=["lr", "n_epochs"])
    return


@app.cell
def _(plot_param_importances, show_per_objective):
    show_per_objective(plot_param_importances)
    return


@app.cell
def _(MODE, plot_pareto_front, study, target_names):
    # Pareto front — only meaningful for a multi-objective study. Shows the
    # accuracy/calibration tradeoff the composite weights collapse.
    if len(MODE["metrics"]) > 1:
        plot_pareto_front(study, target_names=target_names).show()
    return


@app.cell
def _(MODE, plot_pareto_front, study, target_names):
    if len(MODE["metrics"]) > 1:
        plot_pareto_front(
            study, target_names=target_names, include_dominated_trials=False
        ).show()
    return


@app.cell
def _(MODE, log, log_pretty, selection, study):
    # study.best_trial raises under multi-objective, so rank the Pareto front
    # by the same composite the ensemble is built from.
    if len(MODE["metrics"]) == 1:
        _best = study.best_trial
    else:
        _best = min(
            study.best_trials,
            key=lambda t: selection.selection_score(t.values, MODE),
        )
    log.info(f"Best number: {_best.number}")
    log.info(f"Best values: {_best.values}")
    log.info(f"Best score: {selection.selection_score(_best.values, MODE):.4f}")
    log.info(f"Best params: \n{log_pretty(_best.params)}")
    return


@app.cell
def _(study):
    study.trials_dataframe().to_csv("study_csv/test.csv")
    return


@app.cell
def _(MODE, np, optuna, pd, selection):
    def get_best_trials(
        study_name: str,
        n_results: int,
        storage: str = "sqlite:///spp_trials.db",
    ) -> pd.DataFrame:
        """Top-N trials by the active mode's composite score, best first.

        Ranks on ``selection.selection_score`` rather than a raw objective, so
        the ensemble members are chosen by the same formula the promote gate
        uses. Note ``trial.value`` raises under a multi-objective study — read
        ``trial.values`` only.

        Args:
            study_name: Optuna study to load (includes the objective mode).
            storage: Optuna storage URL holding the study.
            n_results: How many trials to return.

        Returns:
            One row per distinct param set, with ``number``, ``values`` (the
            raw objectives), ``score`` (the composite, NaN for a trial that
            did not complete), ``params``, and ``model_path``, sorted best
            score first and truncated to ``n_results``.
        """
        _study = optuna.load_study(study_name=study_name, storage=storage)
        trials = pd.DataFrame(
            [
                {
                    "number": s.number,
                    "values": s.values,
                    "score": (
                        selection.selection_score(s.values, MODE)
                        if s.values is not None
                        else np.nan
                    ),
                    "params": s.params,
                    "model_path": s.user_attrs.get("model_path"),
                }
                for s in _study.trials
            ]
        )
        trials = trials[~trials.params.duplicated()]
        return trials.sort_values("score").head(n_results)

    return (get_best_trials,)


@app.cell
def _(get_best_trials, parameters, study_name):
    best_trials = get_best_trials(study_name, parameters.TOP_N)
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

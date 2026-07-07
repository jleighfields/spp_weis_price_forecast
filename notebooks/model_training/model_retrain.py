# Model retraining notebook for SPP West (RTO West / Integrated Marketplace)
# nodal price forecasting.
#
# Workflow:
#   1. Connect to S3-backed database and prepare LMP + covariate data
#   2. Train TSMixer, TiDE, and TFT models using top-N hyperparameter sets
#   3. Save trained models to S3 using Darts' native serialization (.pt + .pt.ckpt)
#   4. Reload models from S3 and verify predictions via a NaiveEnsembleModel
#   5. Upload champion.json so the Shiny app knows which model folder to load —
#      unless PROMOTE_CHAMPION=false, which stages the checkpoints in their
#      timestamped folder without touching the live champion (see the promote
#      cell for details).
#
# Usage:
#   Interactive: marimo edit notebooks/model_training/model_retrain.py
#   Script:      python notebooks/model_training/model_retrain.py
#   Modal:       modal run modal_jobs/model_retrain.py::model_retrain_weekly

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import io
    import sys
    import tempfile
    import pickle
    import json
    import pathlib
    from time import time

    import pandas as pd
    import boto3
    import torch
    import warnings
    import logging
    from dotenv import load_dotenv
    from darts.models import TFTModel, TiDEModel, TSMixerModel, NaiveEnsembleModel

    warnings.filterwarnings("ignore")
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("model_retrain")
    torch.set_float32_matmul_precision("medium")
    t0 = time()

    # Add project root and src/ to sys.path for imports
    _project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    for _p in [_project_root, os.path.join(_project_root, "src")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    log.info(f"project root: {_project_root}")
    return (
        NaiveEnsembleModel,
        TFTModel,
        TSMixerModel,
        TiDEModel,
        boto3,
        io,
        json,
        log,
        os,
        pd,
        pickle,
        t0,
        tempfile,
        time,
        torch,
    )


@app.cell
def _():
    import data_engineering as de
    import parameters
    import utils
    from modeling import build_fit_tsmixerx, build_fit_tft, build_fit_tide

    return (
        build_fit_tft,
        build_fit_tide,
        build_fit_tsmixerx,
        de,
        parameters,
        utils,
    )


@app.cell
def _(log, parameters):
    log.info(f"FORECAST_HORIZON: {parameters.FORECAST_HORIZON}")
    log.info(f"INPUT_CHUNK_LENGTH: {parameters.INPUT_CHUNK_LENGTH}")
    log.info(f"MODEL_NAME: {parameters.MODEL_NAME}")
    return


@app.cell
def _(log, os):
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    # AWS_S3_FOLDER is a legacy prefix from Databricks Unity Catalog paths.
    # Not needed for Modal jobs or the Posit Connect app — defaults to "".
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER", "")
    log.info(f"{AWS_S3_FOLDER = }")
    return AWS_S3_BUCKET, AWS_S3_FOLDER


@app.cell
def _(boto3, os):
    s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))
    return (s3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Connect to database and prepare data
    """)
    return


@app.cell
def _(de):
    con = de.create_database()
    return (con,)


@app.cell
def _(con, de, log):
    log.info("preparing covariate data")
    all_df_pd = de.all_df_to_pandas(de.prep_all_df(con))
    all_df_pd.info()

    lmp_all, train_all, test_all, train_test_all = de.get_train_test_all(con)
    con.close()
    return all_df_pd, lmp_all, test_all, train_all, train_test_all


@app.cell
def _(all_df_pd, de, lmp_all, test_all, train_all, train_test_all):
    all_series = de.get_series(lmp_all)
    train_test_all_series = de.get_series(train_test_all)
    test_series = de.get_series(test_all)

    futr_cov = de.get_futr_cov(all_df_pd)
    past_cov = de.get_past_cov(all_df_pd)
    return all_series, futr_cov, past_cov, test_series, train_test_all_series


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train models
    """)
    return


@app.cell
def _(
    build_fit_tsmixerx,
    futr_cov,
    parameters,
    past_cov,
    test_series,
    train_test_all_series,
):
    models_tsmixer = []
    if parameters.USE_TSMIXER:
        for _i, _param in enumerate(parameters.TSMIXER_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {_i} \t" + "*" * 25, flush=True)
            _model = build_fit_tsmixerx(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **_param,
            )
            models_tsmixer += [_model]
    return (models_tsmixer,)


@app.cell
def _(
    build_fit_tide,
    futr_cov,
    parameters,
    past_cov,
    test_series,
    train_test_all_series,
):
    models_tide = []
    if parameters.USE_TIDE:
        for _i, _param in enumerate(parameters.TIDE_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {_i} \t" + "*" * 25, flush=True)
            _model = build_fit_tide(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **_param,
            )
            models_tide += [_model]
    return (models_tide,)


@app.cell
def _(
    build_fit_tft,
    futr_cov,
    parameters,
    past_cov,
    test_series,
    train_test_all_series,
):
    models_tft = []
    if parameters.USE_TFT:
        for _i, _param in enumerate(parameters.TFT_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {_i} \t" + "*" * 25, flush=True)
            _model = build_fit_tft(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **_param,
            )
            models_tft += [_model]
    return (models_tft,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save and upload models
    """)
    return


@app.cell
def _(AWS_S3_FOLDER, log, pd, utils):
    utc_timestamp = pd.Timestamp.now("UTC")
    log.info(f"{utc_timestamp = }")

    folder_time = utc_timestamp.strftime("%Y-%m-%d_%H-%M-%S") + "/"
    log.info(f"{folder_time = }")

    artifact_folder = utils.RETRAINS_PREFIX + folder_time
    log.info(f"{artifact_folder = }")

    artifact_path = AWS_S3_FOLDER + artifact_folder
    log.info(f"{artifact_path = }")
    return artifact_folder, artifact_path, folder_time, utc_timestamp


@app.cell
def _(
    AWS_S3_BUCKET,
    artifact_path,
    io,
    log,
    models_tft,
    models_tide,
    models_tsmixer,
    os,
    pickle,
    s3,
    tempfile,
    utc_timestamp,
):
    upload_paths = []

    def model_to_tmp_upload(
        m,
        name: str,
        _bucket: str = AWS_S3_BUCKET,
        _prefix: str = artifact_path,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, name)
            m.save(model_path)

            upload_path = _prefix + name
            s3.upload_file(model_path, _bucket, upload_path)
            log.info(f"Uploaded: {upload_path}")

            ckpt_path = model_path + ".ckpt"
            if os.path.exists(ckpt_path):
                ckpt_upload_path = upload_path + ".ckpt"
                s3.upload_file(ckpt_path, _bucket, ckpt_upload_path)
                log.info(f"Uploaded: {ckpt_upload_path}")
        return upload_path

    # Upload training timestamp
    _buffer = io.BytesIO()
    pickle.dump(utc_timestamp, _buffer)
    _buffer.seek(0)
    _upload_path = artifact_path + "TRAIN_TIMESTAMP.pkl"
    s3.put_object(Bucket=AWS_S3_BUCKET, Key=_upload_path, Body=_buffer)
    log.info(f"Uploaded: {_upload_path}")
    upload_paths += [_upload_path]

    for _i, _m in enumerate(models_tide):
        upload_paths += [model_to_tmp_upload(_m, f"tide_{_i}.pt")]
    for _i, _m in enumerate(models_tsmixer):
        upload_paths += [model_to_tmp_upload(_m, f"tsmixer_{_i}.pt")]
    for _i, _m in enumerate(models_tft):
        upload_paths += [model_to_tmp_upload(_m, f"tft_{_i}.pt")]
    return


@app.cell
def _(
    AWS_S3_BUCKET,
    artifact_path,
    de,
    io,
    json,
    log,
    parameters,
    s3,
    torch,
    train_test_all_series,
    utc_timestamp,
    utils,
):
    # Save training_config.json next to the checkpoints: records exactly what
    # the model was trained on (covariates, nodes, quantiles, versions, train
    # window) so the app can verify its inputs still match before serving it.
    import darts as _darts
    import node_list as _node_list

    _model_types = utils.active_model_types(
        parameters.USE_TIDE, parameters.USE_TSMIXER, parameters.USE_TFT
    )
    _cfg = utils.build_training_config(
        train_timestamp=str(utc_timestamp),
        future_covariates=de.FUTR_COLS,
        past_covariates=de.PAST_COLS,
        nodes=_node_list.MODEL_APP_NODES,
        quantiles=parameters.QUANTILES,
        model_name=parameters.MODEL_NAME,
        model_types=_model_types,
        forecast_horizon=parameters.FORECAST_HORIZON,
        input_chunk_length=parameters.INPUT_CHUNK_LENGTH,
        train_start=str(train_test_all_series[0].start_time()),
        train_end=str(train_test_all_series[0].end_time()),
        darts_version=_darts.__version__,
        torch_version=torch.__version__,
    )
    _cfg_key = artifact_path + utils.TRAINING_CONFIG_FILENAME
    s3.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=_cfg_key,
        Body=io.BytesIO(json.dumps(_cfg, indent=2).encode("utf-8")),
    )
    log.info(f"Uploaded: {_cfg_key}")
    log.info(f"training_config: {_cfg}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test loading models from S3 and doing inference
    """)
    return


@app.cell
def _(artifact_folder, log, tempfile, utils):
    from src.modeling import load_ensemble_from_dir

    log.info("downloading checkpoints and building ensemble for test")
    with tempfile.TemporaryDirectory() as tmpdir:
        utils.download_checkpoints(artifact_folder, tmpdir)
        loaded_model, _train_timestamp = load_ensemble_from_dir(tmpdir)
    return (loaded_model,)


@app.cell
def _(
    all_series,
    loaded_model,
    futr_cov,
    log,
    parameters,
    past_cov,
    pd,
):
    log.info("test getting predictions")
    plot_ind = 3
    plot_series = all_series[plot_ind]

    plot_end_time = plot_series.end_time() - pd.Timedelta(
        f"{parameters.INPUT_CHUNK_LENGTH + 1}h"
    )
    log.info(f"plot_end_time: {plot_end_time}")

    _node_series = plot_series.drop_after(plot_end_time)
    log.info(f"node_series.end_time(): {_node_series.end_time()}")

    pred = loaded_model.predict(
        series=_node_series,
        past_covariates=past_cov[0],
        future_covariates=futr_cov[0],
        n=5,
        num_samples=2,
    )
    log.info(f"pred: {pred}")
    return (pred,)


@app.cell
def _(pred):
    assert pred is not None
    return


@app.cell
def _(pred):
    pred.to_dataframe()
    return


@app.cell
def _(
    AWS_S3_BUCKET,
    AWS_S3_FOLDER,
    artifact_folder,
    artifact_path,
    folder_time,
    io,
    json,
    log,
    os,
    pred,
    s3,
    utils,
):
    # Promote by updating champion.json to point at the new model's folder.
    # The app loads models directly from model_retrains/<timestamp>/ via
    # champion_artifact_folder, so no file copying to S3_models/ is needed.
    # To revert to a previous model, repoint champion.json at the old folder
    # with `python scripts/r2_promote_champion.py <timestamp> --promote`
    # (run it with --list to see the available model_retrains/ folders).
    #
    # Set PROMOTE_CHAMPION=false to train + save the checkpoints to the
    # timestamped folder WITHOUT overwriting champion.json — used to stage a
    # new model (e.g. the first RTO West retrain) before its serving code is
    # deployed, so the live app keeps its current champion until they promote
    # together.
    promote = os.environ.get("PROMOTE_CHAMPION", "true").lower() != "false"
    if pred is not None and promote:
        champion_json = utils.build_champion_config(
            folder_time, artifact_folder, artifact_path
        )
        _buffer = io.BytesIO(json.dumps(champion_json).encode("utf-8"))
        champion_key = AWS_S3_FOLDER + utils.CHAMPION_KEY_SUFFIX
        s3.put_object(Bucket=AWS_S3_BUCKET, Key=champion_key, Body=_buffer)
        log.info(f"Uploaded champion model json: {champion_key}")
        log.info(f"champion_json: {champion_json}")
    elif pred is not None:
        log.info(
            f"PROMOTE_CHAMPION=false: saved checkpoints to {artifact_folder} "
            "but did NOT promote champion.json (staged, not live)."
        )
    else:
        log.warning("Prediction failed, not promoting champion")
    return


@app.cell
def _(all_series, futr_cov, loaded_model, log, past_cov):
    # Score the freshly-trained ensemble on the West holdout so every retrain
    # reports its backtest metrics (CRPS, coverage/width, MAE/RMSE/bias, tail).
    # Runs after promotion and is wrapped so a scoring error never aborts a
    # retrain that already staged/promoted.
    try:
        from src.evaluation import backtest_report

        backtest_report(loaded_model, all_series, past_cov, futr_cov)
    except Exception as _e:
        log.warning(f"backtest scoring skipped: {_e}")
    return


@app.cell
def _(log, t0, time):
    _t1 = time()
    log.info("finished retraining")
    log.info(f"total time (min): {(_t1 - t0) / 60:.2f}")
    return


if __name__ == "__main__":
    app.run()

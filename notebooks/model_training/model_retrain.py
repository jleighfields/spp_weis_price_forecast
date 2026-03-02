# Model retraining notebook for SPP WEIS nodal price forecasting.
#
# Workflow:
#   1. Connect to S3-backed database and prepare LMP + covariate data
#   2. Train TSMixer, TiDE, and TFT models using top-N hyperparameter sets
#   3. Save trained models to S3 using Darts' native serialization (.pt + .pt.ckpt)
#   4. Reload models from S3 and verify predictions via a NaiveEnsembleModel
#   5. Upload champion.json so the Shiny app knows which model folder to load
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
    log.info("preparing lmp data")
    lmp = de.prep_lmp(con)
    lmp_df = lmp.to_pandas().rename(
        columns={
            "LMP": "LMP_HOURLY",
            "unique_id": "node",
            "timestamp_mst": "time",
        }
    )

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
    train_series = de.get_series(train_all)
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
def _(AWS_S3_FOLDER, log, pd):
    utc_timestamp = pd.Timestamp.now("UTC")
    log.info(f"{utc_timestamp = }")

    folder_time = utc_timestamp.strftime("%Y-%m-%d_%H-%M-%S") + "/"
    log.info(f"{folder_time = }")

    artifact_folder = "model_retrains/" + folder_time
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
def _(artifact_folder, utils):
    loaded_models_for_test = utils.get_loaded_models(artifact_folder)
    loaded_models_for_test
    return (loaded_models_for_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test loading models from S3 and doing inference
    """)
    return


@app.cell
def _(
    AWS_S3_BUCKET,
    TFTModel,
    TSMixerModel,
    TiDEModel,
    loaded_models_for_test,
    log,
    os,
    s3,
    tempfile,
    torch,
):
    def get_checkpoints(model_filter: str):
        """Filter S3 keys to just the main .pt files (not .ckpt or .pkl)."""
        return [
            f
            for f in loaded_models_for_test
            if model_filter in f
            and ".pt" in f
            and ".ckpt" not in f
            and "TRAIN_TIMESTAMP.pkl" not in f
        ]

    def load_model_from_s3(model_class, key):
        """Download a model's .pt + .pt.ckpt files to a temp dir and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = key.split("/")[-1]
            local_path = os.path.join(tmpdir, filename)
            s3.download_file(Bucket=AWS_S3_BUCKET, Key=key, Filename=local_path)
            try:
                s3.download_file(
                    Bucket=AWS_S3_BUCKET,
                    Key=key + ".ckpt",
                    Filename=local_path + ".ckpt",
                )
            except Exception:
                log.warning(f"No checkpoint file found for {key}")
            log.info(f"loading model: {key}")
            model = model_class.load(local_path, map_location=torch.device("cpu"))
        return model

    ts_mixer_forecasting_models = [
        load_model_from_s3(TSMixerModel, m) for m in get_checkpoints("tsmixer_")
    ]
    tide_forecasting_models = [
        load_model_from_s3(TiDEModel, m) for m in get_checkpoints("tide_")
    ]
    tft_forecasting_models = [
        load_model_from_s3(TFTModel, m) for m in get_checkpoints("tft_")
    ]

    forecasting_models = (
        ts_mixer_forecasting_models
        + tide_forecasting_models
        + tft_forecasting_models
    )
    return (forecasting_models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create ensemble model and test predictions
    """)
    return


@app.cell
def _(
    NaiveEnsembleModel,
    all_series,
    forecasting_models,
    futr_cov,
    log,
    parameters,
    past_cov,
    pd,
):
    log.info("loading model from checkpoints")
    loaded_model = NaiveEnsembleModel(
        forecasting_models=forecasting_models,
        train_forecasting_models=False,
    )

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
    pred,
    s3,
):
    # Promote by updating champion.json to point at the new model's folder.
    # The app loads models directly from model_retrains/<timestamp>/ via
    # champion_artifact_folder, so no file copying to S3_models/ is needed.
    # To revert to a previous model, just update champion.json to point at
    # the old folder (see scripts/r2_promote_champion.py or the plan).
    if pred is not None:
        champion_json = {
            "champion": folder_time,
            "champion_artifact_folder": artifact_folder,
            "champion_artifact_path": artifact_path,
        }
        _buffer = io.BytesIO(json.dumps(champion_json).encode("utf-8"))
        champion_key = AWS_S3_FOLDER + "S3_models/champion.json"
        s3.put_object(Bucket=AWS_S3_BUCKET, Key=champion_key, Body=_buffer)
        log.info(f"Uploaded champion model json: {champion_key}")
        log.info(f"champion_json: {champion_json}")
    else:
        log.warning("Prediction failed, not promoting champion")
    return


@app.cell
def _(log, t0, time):
    _t1 = time()
    log.info("finished retraining")
    log.info(f"total time (min): {(_t1 - t0) / 60:.2f}")
    return


if __name__ == "__main__":
    app.run()

"""Modal model retrain job for SPP WEIS price forecast.

Based on the Databricks model retrain notebook:
- model_retrain: notebooks/model_training/model_retrain.ipynb

Test:  modal run modal_jobs/model_retrain.py::model_retrain_weekly
Deploy: modal deploy modal_jobs/model_retrain.py

TODO: Migrate model retrain notebook from Jupyter to marimo and update
      this Modal job to use the marimo notebook directly.
"""

import modal

app = modal.App("spp-weis-model-retrain")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "polars==1.37.1",
        "pyarrow==19.0.1",
        "boto3==1.35.92",
        "duckdb==1.4.3",
        "requests",
        "tqdm==4.67.1",
        "polars-xdt==0.17.1",
        "pandas",
        "joblib",
        "torch==2.5.1",
        "darts==0.31.0",
        "scikit-learn",
    )
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    schedule=modal.Cron("0 20 * * 0"),  # Sundays at 8 PM UTC
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=7200,  # 2 hours
    cpu=8.0,  # 8 physical cores
    memory=32768,  # 32 GiB
    gpu="A10G",
)
def model_retrain_weekly():
    """Retrain ensemble models (TiDE, TSMixer, TFT) and promote champion."""
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    import os
    import io
    import tempfile
    import pickle
    import json
    from time import time

    import pandas as pd
    import boto3
    import torch
    import warnings
    import logging

    from darts.models import (
        TFTModel,
        TiDEModel,
        TSMixerModel,
        NaiveEnsembleModel,
    )

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("model_retrain")
    t0 = time()

    import data_engineering as de
    import parameters
    import utils
    from modeling import build_fit_tsmixerx, build_fit_tft, build_fit_tide

    log.info(f"FORECAST_HORIZON: {parameters.FORECAST_HORIZON}")
    log.info(f"INPUT_CHUNK_LENGTH: {parameters.INPUT_CHUNK_LENGTH}")
    log.info(f"MODEL_NAME: {parameters.MODEL_NAME}")

    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER")
    log.info(f"{AWS_S3_FOLDER = }")

    s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))

    # Connect to database and prepare data
    con = de.create_database()

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

    # Convert DataFrames to Darts TimeSeries objects
    all_series = de.get_series(lmp_all)
    train_test_all_series = de.get_series(train_test_all)
    train_series = de.get_series(train_all)
    test_series = de.get_series(test_all)

    futr_cov = de.get_futr_cov(all_df_pd)
    past_cov = de.get_past_cov(all_df_pd)

    # Train models
    models_tsmixer = []
    if parameters.USE_TSMIXER:
        for i, param in enumerate(parameters.TSMIXER_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {i} \t" + "*" * 25, flush=True)
            model_tsmixer = build_fit_tsmixerx(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **param,
            )
            models_tsmixer += [model_tsmixer]

    models_tide = []
    if parameters.USE_TIDE:
        for i, param in enumerate(parameters.TIDE_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {i} \t" + "*" * 25, flush=True)
            model_tide = build_fit_tide(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **param,
            )
            models_tide += [model_tide]

    models_tft = []
    if parameters.USE_TFT:
        for i, param in enumerate(parameters.TFT_PARAMS[: parameters.TOP_N]):
            print(f"\ni: {i} \t" + "*" * 25, flush=True)
            model_tft = build_fit_tft(
                series=train_test_all_series,
                val_series=test_series,
                future_covariates=futr_cov,
                past_covariates=past_cov,
                **param,
            )
            models_tft += [model_tft]

    # Save and upload models to S3
    utc_timestamp = pd.Timestamp.utcnow()
    log.info(f"{utc_timestamp = }")

    folder_time = utc_timestamp.strftime("%Y-%m-%d_%H-%M-%S") + "/"
    log.info(f"{folder_time = }")

    artifact_folder = "model_retrains/" + folder_time
    log.info(f"{artifact_folder = }")

    artifact_path = AWS_S3_FOLDER + artifact_folder
    log.info(f"{artifact_path = }")

    def model_to_tmp_upload(m, name):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, name)
            m.save(model_path)

            upload_path = artifact_path + name
            s3.upload_file(model_path, AWS_S3_BUCKET, upload_path)
            log.info(f"Uploaded: {upload_path}")

            ckpt_path = model_path + ".ckpt"
            if os.path.exists(ckpt_path):
                ckpt_upload_path = upload_path + ".ckpt"
                s3.upload_file(ckpt_path, AWS_S3_BUCKET, ckpt_upload_path)
                log.info(f"Uploaded: {ckpt_upload_path}")

        return upload_path

    # Upload training timestamp
    buffer = io.BytesIO()
    pickle.dump(utc_timestamp, buffer)
    buffer.seek(0)
    upload_path = artifact_path + "TRAIN_TIMESTAMP.pkl"
    s3.put_object(Bucket=AWS_S3_BUCKET, Key=upload_path, Body=buffer)
    log.info(f"Uploaded: {upload_path}")

    upload_paths = [upload_path]
    for i, m in enumerate(models_tide):
        upload_paths += [model_to_tmp_upload(m, f"tide_{i}.pt")]
    for i, m in enumerate(models_tsmixer):
        upload_paths += [model_to_tmp_upload(m, f"tsmixer_{i}.pt")]
    for i, m in enumerate(models_tft):
        upload_paths += [model_to_tmp_upload(m, f"tft_{i}.pt")]

    # Verify models can be loaded from S3
    loaded_models_for_test = utils.get_loaded_models(artifact_folder)
    log.info(f"loaded_models_for_test: {loaded_models_for_test}")

    def get_checkpoints(model_filter):
        return [
            f
            for f in loaded_models_for_test
            if model_filter in f
            and ".pt" in f
            and ".ckpt" not in f
            and "TRAIN_TIMESTAMP.pkl" not in f
        ]

    def load_model_from_s3(model_class, key):
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
        ts_mixer_forecasting_models + tide_forecasting_models + tft_forecasting_models
    )

    # Create ensemble and test predictions
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
    node_series = plot_series.drop_after(plot_end_time)
    future_cov_series = futr_cov[0]
    past_cov_series = past_cov[0]

    pred = loaded_model.predict(
        series=node_series,
        past_covariates=past_cov_series,
        future_covariates=future_cov_series,
        n=5,
        num_samples=2,
    )
    log.info(f"pred: {pred}")

    assert pred is not None

    # Promote to champion if prediction succeeded
    if pred is not None:
        champion_json = {
            "champion": folder_time,
            "champion_artifact_folder": artifact_folder,
            "champion_artifact_path": artifact_path,
        }
        buffer = io.BytesIO(json.dumps(champion_json).encode("utf-8"))
        champion_key = AWS_S3_FOLDER + "S3_models/champion.json"
        s3.put_object(Bucket=AWS_S3_BUCKET, Key=champion_key, Body=buffer)
        log.info(f"Uploaded champion model json: {champion_key}")
        log.info(f"champion_json: {champion_json}")
    else:
        log.warning("Prediction failed, not saving json")

    t1 = time()
    log.info("finished retraining")
    log.info(f"total time (min): {(t1 - t0) / 60:.2f}")

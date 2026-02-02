# Model retraining notebook for SPP WEIS nodal price forecasting.
#
# Workflow:
#   1. Connect to S3-backed database and prepare LMP + covariate data
#   2. Train TSMixer, TiDE, and TFT models using top-N hyperparameter sets
#   3. Save trained models to S3 using Darts' native serialization (.pt + .pt.ckpt)
#   4. Reload models from S3 and verify predictions via a NaiveEnsembleModel
#   5. Upload champion.json so the Shiny app knows which model folder to load

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    import io
    import tempfile
    import shutil
    import pickle
    import json
    import sys
    from time import time

    from typing import List
    import tempfile

    import pandas as pd
    import boto3
    import torch
    import warnings
    import logging
    from dotenv import load_dotenv

    from darts.models import (
        TFTModel,
        TiDEModel,
        TSMixerModel,
        NaiveEnsembleModel,
    )

    warnings.filterwarnings("ignore")
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("model_retrain")
    t0 = time()
    return (
        List,
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
        sys,
        t0,
        tempfile,
        time,
        torch,
    )


@app.cell
def _(log, os, sys):
    # Navigate to project root so src/ imports work regardless of where
    # the notebook is launched from (local dev, Databricks, CI, etc.)
    while 'src' not in os.listdir():
        os.chdir('..')

    src_path = "src"
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

    log.info(f"os.getcwd(): {os.getcwd()}")

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
def _(dbutils, os):
    # On Databricks, pull AWS creds from the secrets store.
    # Locally, fall back to .env file via python-dotenv.
    if 'dbutils' in locals():
        os.environ['AWS_ACCESS_KEY_ID'] = dbutils.secrets.get(scope="aws", key="AWS_ACCESS_KEY_ID")
        os.environ['AWS_SECRET_ACCESS_KEY'] = dbutils.secrets.get(scope="aws", key="AWS_SECRET_ACCESS_KEY")
        os.environ['AWS_S3_BUCKET'] = dbutils.secrets.get(scope="aws", key="AWS_S3_BUCKET")
        os.environ['AWS_S3_FOLDER'] = dbutils.secrets.get(scope="aws", key="AWS_S3_FOLDER")
        import mlflow
        mlflow.autolog(disable=True)
    else:
        print('not on DBX')

    from dotenv import load_dotenv
    load_dotenv()
    return


@app.cell
def _(log, parameters):
    log.info(f"FORECAST_HORIZON: {parameters.FORECAST_HORIZON}")
    log.info(f"INPUT_CHUNK_LENGTH: {parameters.INPUT_CHUNK_LENGTH}")
    log.info(f"MODEL_NAME: {parameters.MODEL_NAME}")
    return


@app.cell
def _(log, os):
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_S3_FOLDER = os.getenv("AWS_S3_FOLDER")
    log.info(f'{AWS_S3_FOLDER = }')
    return AWS_S3_BUCKET, AWS_S3_FOLDER


@app.cell
def _(boto3):
    s3 = boto3.client("s3")
    return (s3,)


@app.cell
def _(mo):
    mo.md("""
    ## Connect to database and prepare data
    """)
    return


@app.cell
def _(de):
    # Create a DuckDB connection backed by S3 parquet files
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
    # Convert DataFrames to Darts TimeSeries objects for model training/prediction
    all_series = de.get_series(lmp_all)
    train_test_all_series = de.get_series(train_test_all)
    train_series = de.get_series(train_all)
    test_series = de.get_series(test_all)

    # Future covariates (known ahead of time, e.g. weather/load forecasts)
    # and past covariates (only known up to current time)
    futr_cov = de.get_futr_cov(all_df_pd)
    past_cov = de.get_past_cov(all_df_pd)
    return all_series, futr_cov, past_cov, test_series, train_test_all_series


@app.cell
def _(mo):
    mo.md("""
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
    def _():
        # Train TOP_N TSMixer models with different hyperparameter sets
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

        return models_tsmixer 

    models_tsmixer = _()
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
    def _():
        # Train TOP_N TiDE models with different hyperparameter sets
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

        return models_tide


    models_tide = _()
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
    def _():
        # Train TOP_N TFT models with different hyperparameter sets
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

        return models_tft

    models_tft = _()
    return (models_tft,)


@app.cell
def _(mo):
    mo.md("""
    ## Save and upload models
    """)
    return


@app.cell
def _(AWS_S3_FOLDER, log, pd):
    # Create a timestamped folder path for this retrain run's artifacts.
    # artifact_folder: relative path passed to utils.get_loaded_models()
    # artifact_path:   full S3 key prefix for uploading files
    utc_timestamp = pd.Timestamp.utcnow()
    log.info(f'{utc_timestamp = }')

    folder_time = utc_timestamp.strftime('%Y-%m-%d_%H-%M-%S') + '/'
    log.info(f'{folder_time = }')

    artifact_folder = 'model_retrains/' + folder_time
    log.info(f'{artifact_folder = }')

    artifact_path = AWS_S3_FOLDER + artifact_folder
    log.info(f'{artifact_path = }')
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
    def _():
        # Upload all trained models to S3 under the timestamped artifact path.
        # Darts' .save() produces two files per model:
        #   <name>.pt      - pickled model wrapper (config, training state)
        #   <name>.pt.ckpt - PyTorch Lightning checkpoint (neural network weights)
        # Both are required for Darts' .load() to fully restore a trained model.

        upload_paths = []

        def model_to_tmp_upload(
            m,
            name: str,
            AWS_S3_BUCKET: str=AWS_S3_BUCKET,
            artifact_path: str=artifact_path,
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, name)
                m.save(model_path)

                # Upload the model wrapper file
                upload_path = artifact_path + name
                s3.upload_file(model_path, AWS_S3_BUCKET, upload_path)
                log.info(f'Uploaded: {upload_path}')

                # Upload the checkpoint file with neural network weights
                ckpt_path = model_path + '.ckpt'
                if os.path.exists(ckpt_path):
                    ckpt_upload_path = upload_path + '.ckpt'
                    s3.upload_file(ckpt_path, AWS_S3_BUCKET, ckpt_upload_path)
                    log.info(f'Uploaded: {ckpt_upload_path}')

            return upload_path

        # Upload training timestamp so the Shiny app can display when models were last trained
        buffer = io.BytesIO()
        pickle.dump(utc_timestamp, buffer)
        buffer.seek(0)
        upload_path = artifact_path + "TRAIN_TIMESTAMP.pkl"
        s3.put_object(
            Bucket=AWS_S3_BUCKET, 
            Key=upload_path, 
            Body=buffer,
        )
        log.info(f'Uploaded: {upload_path}')
        upload_paths+=[upload_path]

        for i, m in enumerate(models_tide):
            upload_paths+=[model_to_tmp_upload(m, f"tide_{i}.pt")]

        for i, m in enumerate(models_tsmixer):
            upload_paths+=[model_to_tmp_upload(m, f"tsmixer_{i}.pt")]

        for i, m in enumerate(models_tft):
            upload_paths+=[model_to_tmp_upload(m, f"tft_{i}.pt")]

        return upload_paths

    upload_paths = _()
    return


@app.cell
def _(artifact_folder, utils):
    # List all model files just uploaded to S3 for verification and loading
    loaded_models_for_test = utils.get_loaded_models(artifact_folder)
    loaded_models_for_test
    return (loaded_models_for_test,)


@app.cell
def _(mo):
    mo.md("""
    ## Test loading models from S3 and doing inference
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Load models by type
    """)
    return


@app.cell
def _(
    AWS_S3_BUCKET,
    List,
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
    def _():
        # Re-download models from S3 and load them to verify the save/load
        # round-trip works before promoting to champion.
        # Each model needs both .pt and .pt.ckpt files in the same directory
        # for Darts' .load() to fully restore the trained model.

        def get_checkpoints(
            model_filter: str, # 'tsmixer_', 'tide_', or 'tft_'
            loaded_models_for_test: List[str]=loaded_models_for_test,
        ):
            """Filter S3 keys to just the main .pt files (not .ckpt or .pkl)."""
            return [
                f for f in loaded_models_for_test
                if model_filter in f and ".pt" in f and ".ckpt" not in f
                and "TRAIN_TIMESTAMP.pkl" not in f
            ]

        def load_model_from_s3(model_class, key):
            """Download a model's .pt + .pt.ckpt files to a temp dir and load."""
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = key.split('/')[-1]
                local_path = os.path.join(tmpdir, filename)

                # Download the model wrapper file
                s3.download_file(Bucket=AWS_S3_BUCKET, Key=key, Filename=local_path)

                # Download the checkpoint file with neural network weights
                try:
                    s3.download_file(
                        Bucket=AWS_S3_BUCKET,
                        Key=key + '.ckpt',
                        Filename=local_path + '.ckpt',
                    )
                except Exception:
                    log.warning(f"No checkpoint file found for {key}")

                log.info(f"loading model: {key}")
                model = model_class.load(local_path, map_location=torch.device("cpu"))

            return model

        ts_mixer_ckpts = get_checkpoints("tsmixer_")
        ts_mixer_forecasting_models = [load_model_from_s3(TSMixerModel, m) for m in ts_mixer_ckpts]

        tide_ckpts = get_checkpoints("tide_")
        tide_forecasting_models = [load_model_from_s3(TiDEModel, m) for m in tide_ckpts]

        tft_ckpts = get_checkpoints("tft_")
        tft_forecasting_models = [load_model_from_s3(TFTModel, m) for m in tft_ckpts]

        forecasting_models = (
            ts_mixer_forecasting_models
            + tide_forecasting_models
            + tft_forecasting_models
        )

        return forecasting_models

    forecasting_models = _()
    return (forecasting_models,)


@app.cell
def _(mo):
    mo.md("""
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
    def _():
        # Combine all reloaded models into a NaiveEnsembleModel (simple average
        # of predictions). train_forecasting_models=False since they're already trained.
        log.info("loading model from checkpoints")
        loaded_model = NaiveEnsembleModel(
            forecasting_models=forecasting_models,
            train_forecasting_models=False,
        )

        # Smoke test: run a short prediction on one node to verify the
        # ensemble works end-to-end before promoting to champion.
        log.info("test getting predictions")
        plot_ind = 3
        plot_series = all_series[plot_ind]

        plot_end_time = plot_series.end_time() - pd.Timedelta(
            f"{parameters.INPUT_CHUNK_LENGTH + 1}h"
        )
        log.info(f"plot_end_time: {plot_end_time}")

        plot_node_name = plot_series.static_covariates.unique_id.LMP
        node_series = plot_series.drop_after(plot_end_time)
        log.info(f"plot_end_time: {plot_end_time}")
        log.info(f"node_series.end_time(): {node_series.end_time()}")
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

        return pred

    pred = _()
    return (pred,)


@app.cell
def _(pred):
    assert pred is not None
    return


@app.cell
def _(pred):
    pred.pd_dataframe()
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
    def _():
        # Upload champion.json to S3_models/ so the Shiny app knows which
        # retrain folder contains the current production models.
        # Only promoted if the smoke-test prediction succeeded.
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

    _()
    return


@app.cell
def _(log, t0, time):
    # from time import time as _time

    t1 = time()
    log.info("finished retraining")
    log.info(f"total time (min): {(t1 - t0) / 60:.2f}")
    return


if __name__ == "__main__":
    app.run()

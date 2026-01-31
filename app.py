'''
Shiny for Python interface for SPP Weis LMP forecasting endpoint
'''

# pylint: disable=W0621,C0103,W1203

# base imports
import os
import random
import pickle
import logging
from pathlib import Path
from typing import List

# data
import numpy as np
import pandas as pd
import boto3

# user interface
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly
import plotly.io as pio
import shinyswatch

# forecasting data
import torch

from darts.models import (
    TFTModel,
    TiDEModel,
    TSMixerModel,
    NaiveEnsembleModel,
)

# custom modules
import src.data_engineering as de
from src import utils
from src import plotting

# max absolute value for LMPs in given to forecast
MAX_LMP = 200.0

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# load env
from dotenv import load_dotenv
load_dotenv()


###############################################################
# Helper functions
###############################################################

def get_price_nodes(lmp_df: pd.DataFrame) -> List[str]:
    '''
    get list of LMP nodes for drop down menu
    args:
        lmp_df: pd.DataFrame hourly LMP data with 'unique_id'
    returns: List[str] of LMP names
    '''
    price_node_list = np.sort(lmp_df.unique_id.unique()).tolist()
    log.info(f'price_node_list: {price_node_list}')
    return price_node_list


def get_hour_list(fcast_date, lmp_pd_df: pd.DataFrame) -> List[str]:
    '''
    get list of hours for drop down menu
    args:
        fcast_date: date object from ui.input_date
        lmp_pd_df: pd.DataFrame with timestamp index
    returns: List[str] of zero padded hours
    '''
    today = lmp_pd_df.index.max()
    log.info(f'type(today): {type(today)}')
    log.info(f'today: {today}')

    if fcast_date == today.date():
        last_hour = today.hour
    else:
        last_hour = 23

    hour_list = [str(h).zfill(2) for h in range(last_hour + 1)]
    return hour_list


def get_fcast_time(fcast_date: str, fcast_hour: str) -> str:
    '''
    get the forecast time string
    '''
    return f'{fcast_date}T{fcast_hour}:00:00.000000000'


###############################################################
# UI Layout
###############################################################

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Select forecast start date"),
        ui.input_date("fcast_date", "Forecast date"),
        ui.input_select("fcast_hour", "Forecast hour", choices=[]),
        ui.hr(),
        ui.h4("Select LMP node"),
        ui.input_select("node_name", "LMP node", choices=[]),
        ui.input_select(
            "n_days",
            "Number of days to forecast",
            choices={"5": "5", "4": "4", "3": "3", "2": "2", "1": "1"},
            selected="5",
        ),
        ui.input_select(
            "lookback_days",
            "Number of days to lookback",
            choices={
                "7": "7", "6": "6", "5": "5", "4": "4",
                "3": "3", "2": "2", "1": "1",
            },
            selected="7",
        ),
        ui.input_action_button(
            "get_fcast_btn", "Get forecast", class_="btn-primary"
        ),
        ui.hr(),
        ui.markdown("**NOTES:**"),
        ui.markdown("Data is updated every 4-6 hours"),
        ui.markdown("Model last trained:"),
        ui.output_ui("train_timestamp_display"),
        width=300,
    ),
    # Main content
    ui.tags.img(
        src="wind_farm.png",
        style="width: 100%; max-height: 400px; object-fit: cover; display: block;",
    ),
    ui.div(
        ui.br(),
        # ui.h1("SPP Weis Nodal Price Forecast"),
        ui.row(
            ui.column(2, ui.input_action_button("refresh_data", "Refresh data")),
            ui.column(
                10,
                ui.markdown(
                    "**[SPP Weis price map](https://pricecontourmap.spp.org/pricecontourmapwest/)** | "
                    "**[SPP Weis load and resource forecasts](https://portal.spp.org/pages/weis-forecast-summary)** | "
                    "**[SPP Weis generation mix](https://portal.spp.org/pages/weis-generation-mix)**"
                ),
                style="display: flex; align-items: center;",
            ),
        ),
        ui.hr(),
        ui.output_ui("forecast_header"),
        ui.output_ui("forecast_placeholder"),
        output_widget("forecast_plot"),
        ui.hr(),
        ui.output_ui("forecast_data_section"),
        style="padding-left: 2rem; padding-right: 2rem;",
    ),
    title="SPP Weis Nodal Price Forecast",
    fillable=False,
    theme=shinyswatch.theme.flatly(),
)


###############################################################
# Server
###############################################################

def server(input, output, session):

    # Reactive values for stored state
    all_df_pd_val = reactive.Value(None)
    lmp_pd_df_val = reactive.Value(None)
    loaded_model_val = reactive.Value(None)
    train_timestamp_val = reactive.Value("")
    preds_val = reactive.Value(None)
    plot_cov_df_val = reactive.Value(None)
    fcast_node_name_val = reactive.Value(None)
    fcast_time_val = reactive.Value(None)

    ###############################################################
    # Load data on startup and on refresh
    ###############################################################

    @reactive.effect
    def _load_data():
        # dependency on refresh button; also runs on startup (value starts at 0)
        input.refresh_data()

        with ui.Progress(min=0, max=3) as p:
            p.set(message="Loading data...")

            p.set(1, detail="Connecting to data in S3...")
            log.info('getting lmp data from s3')
            con = de.create_database()
            log.info('finished getting data from s3')

            p.set(2, detail="Preparing data...")
            log.info('preparing all_df_pd')
            all_df_pd_val.set(de.all_df_to_pandas(de.prep_all_df(con)))
            log.info('preparing lmp')
            lmp_result = de.prep_lmp(con)
            log.info('preparing lmp_pd_df')
            lmp_pd = lmp_result.to_pandas().set_index('timestamp_mst')
            lmp_pd_df_val.set(lmp_pd)
            con.close()

            p.set(3, detail="Done")

        ui.notification_show("Done loading and preparing data", type="message")

    ###############################################################
    # Load models once on startup
    ###############################################################

    @reactive.effect
    def _load_models():
        if loaded_model_val() is not None:
            return

        with ui.Progress(min=0, max=2) as p:
            p.set(message="Loading models from S3...")

            # remove old model files
            files_to_remove = [
                f for f in os.listdir('.')
                if f.endswith('.pt') or f.endswith('.ckpt') or f.endswith('.pkl')
            ]
            for f in files_to_remove:
                log.info(f'removing: {f}')
                os.remove(f)

            log.info('downloading model checkpoints')
            AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
            s3_client = boto3.client('s3')
            loaded_models_list = utils.get_loaded_models()

            log.info(f'loaded_models: {loaded_models_list}')
            for lm in loaded_models_list:
                log.info(f'downloading: {lm}')
                s3_client.download_file(
                    Bucket=AWS_S3_BUCKET, Key=lm, Filename=lm.split('/')[-1]
                )

            p.set(1, detail="Loading model weights...")

            tsmixer_ckpts = [
                f for f in os.listdir('.')
                if 'tsmixer' in f and '.pt' in f
                and '.ckpt' not in f and 'TRAIN_TIMESTAMP.pkl' not in f
            ]
            tsmixer_forecasting_models = []
            for m_ckpt in tsmixer_ckpts:
                log.info(f'loading model: {m_ckpt}')
                tsmixer_forecasting_models.append(
                    TSMixerModel.load(m_ckpt, map_location=torch.device('cpu'))
                )

            tide_ckpts = [
                f for f in os.listdir('.')
                if 'tide_' in f and '.pt' in f
                and '.ckpt' not in f and 'TRAIN_TIMESTAMP.pkl' not in f
            ]
            tide_forecasting_models = []
            for m_ckpt in tide_ckpts:
                log.info(f'loading model: {m_ckpt}')
                tide_forecasting_models.append(
                    TiDEModel.load(m_ckpt, map_location=torch.device('cpu'))
                )

            tft_ckpts = [
                f for f in os.listdir('.')
                if 'tft' in f and '.pt' in f
                and '.ckpt' not in f and 'TRAIN_TIMESTAMP.pkl' not in f
            ]
            tft_forecasting_models = []
            for m_ckpt in tft_ckpts:
                log.info(f'loading model: {m_ckpt}')
                tft_forecasting_models.append(
                    TFTModel.load(m_ckpt, map_location=torch.device('cpu'))
                )

            forecasting_models = (
                tsmixer_forecasting_models
                + tide_forecasting_models
                + tft_forecasting_models
            )
            model = NaiveEnsembleModel(
                forecasting_models=forecasting_models,
                train_forecasting_models=False,
            )

            log.info(f'loaded_model: {model}')
            loaded_model_val.set(model)

            # get model training timestamp
            with open('TRAIN_TIMESTAMP.pkl', 'rb') as handle:
                TRAIN_TIMESTAMP = pickle.load(handle)

            log.info(f'TRAIN_TIMESTAMP: {TRAIN_TIMESTAMP}')
            train_timestamp_val.set(str(TRAIN_TIMESTAMP))

            p.set(2, detail="Done")

        ui.notification_show("Done loading models", type="message")

    ###############################################################
    # Update sidebar inputs when data is loaded
    ###############################################################

    @reactive.effect
    def _update_inputs():
        df = lmp_pd_df_val()
        if df is None:
            return

        today = df.index.max()
        min_date = today - pd.Timedelta('60D')

        ui.update_date(
            "fcast_date",
            value=today.date(),
            min=min_date.date(),
            max=today.date(),
        )

        nodes = get_price_nodes(df)
        ui.update_select(
            "node_name",
            choices=dict(zip(nodes, nodes)),
            selected=nodes[0] if nodes else None,
        )

    @reactive.effect
    def _update_hours():
        df = lmp_pd_df_val()
        if df is None:
            return
        fcast_date = input.fcast_date()
        hours = get_hour_list(fcast_date, df)
        ui.update_select(
            "fcast_hour",
            choices=dict(zip(hours, hours)),
            selected=hours[-1] if hours else None,
        )

    ###############################################################
    # Render static outputs
    ###############################################################

    @render.ui
    def train_timestamp_display():
        ts = train_timestamp_val()
        if ts:
            return ui.strong(ts)
        return ui.div()

    ###############################################################
    # Run forecast on button click
    ###############################################################

    @reactive.effect
    @reactive.event(input.get_fcast_btn)
    def _run_forecast():
        df = lmp_pd_df_val()
        all_df = all_df_pd_val()
        model = loaded_model_val()

        if df is None or all_df is None or model is None:
            ui.notification_show(
                "Data or model not loaded yet", type="warning"
            )
            return

        node_name = input.node_name()
        n_days = int(input.n_days())
        fcast_date = input.fcast_date()
        fcast_hour = input.fcast_hour()

        fcast_time = get_fcast_time(fcast_date, fcast_hour)
        fcast_time = pd.Timestamp(fcast_time) + pd.Timedelta('1h')
        fcast_node_name_val.set(node_name)
        fcast_time_val.set(fcast_time)
        log.info(f'fcast_time: {fcast_time}')

        log.info('USER INPUTS:')
        log.info(f'\tnode_name: {node_name}')
        log.info(f'\tfcast_time: {fcast_time}')
        log.info(f'\tn_days: {n_days}')

        with ui.Progress(min=0, max=2) as p:
            p.set(message="Running forecast...")

            # get prices for user selected node
            idx = df.unique_id == node_name
            price_df = df[idx]

            # get covariates for user selected node
            idx = all_df.unique_id == node_name
            plot_cov_df = all_df[idx]

            # prepare data for getting predictions
            trimmed_price_df = price_df.copy()
            trimmed_price_df.loc[
                trimmed_price_df.LMP > MAX_LMP, 'LMP'
            ] = MAX_LMP
            trimmed_price_df.loc[
                trimmed_price_df.LMP < -MAX_LMP, 'LMP'
            ] = -MAX_LMP
            log.info(f'max trimmed lmp: {trimmed_price_df.LMP.max()}')
            plot_series = de.get_series(trimmed_price_df)[0]
            future_cov_series = de.get_futr_cov(plot_cov_df)[0]
            past_cov_series = de.get_past_cov(plot_cov_df)[0]
            node_series = plot_series
            if fcast_time <= node_series.end_time():
                node_series = node_series.drop_after(fcast_time)

            p.set(1, detail="Generating predictions...")

            log.info(f'n_days: {n_days}')
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)
            preds = model.predict(
                series=node_series,
                past_covariates=past_cov_series,
                future_covariates=future_cov_series,
                n=n_days * 24,
                num_samples=500,
            )

            cov_df = future_cov_series.pd_dataframe()
            cov_df['re_ratio'] = (
                (cov_df.Wind_Forecast_MW + cov_df.Solar_Forecast_MW)
                / cov_df.MTLF
            )
            cov_df = (
                cov_df
                .reset_index()
                .rename(columns={
                    'timestamp_mst': 'time',
                    're_ratio': 'Ratio',
                })
            )

            preds_val.set(preds)
            plot_cov_df_val.set(cov_df)

            p.set(2, detail="Done")

        ui.notification_show("Forecast complete", type="message")

    ###############################################################
    # Shared computed forecast data
    ###############################################################

    @reactive.calc
    def forecast_display_data():
        '''Compute the merged plot DataFrame used by chart, table, and download.'''
        preds = preds_val()
        if preds is None:
            return None

        node_name = fcast_node_name_val()
        plot_cov_df = plot_cov_df_val()

        lmp_df = (
            lmp_pd_df_val()
            .reset_index()
            .rename(columns={
                'LMP': 'LMP_HOURLY',
                'unique_id': 'node',
                'timestamp_mst': 'time',
            })
        )

        plot_df = plotting.get_plot_df(preds, plot_cov_df, lmp_df, node_name)
        plot_df.rename(columns={'mean': 'mean_fcast'}, inplace=True)
        return plot_df

    ###############################################################
    # Forecast outputs
    ###############################################################

    @render.ui
    def forecast_header():
        preds = preds_val()
        if preds is None:
            return ui.div()

        node_name = fcast_node_name_val()
        fcast_time = fcast_time_val()
        return ui.div(
            ui.h3(f"{node_name} forecasts"),
            ui.p(f"Forecast start time: {fcast_time}"),
        )

    @render.ui
    def forecast_placeholder():
        if preds_val() is not None:
            return ui.div()
        return ui.p(
            "Click 'Get forecast' to generate a plot.",
            style="color: gray; font-style: italic; padding: 2rem 0;",
        )

    @render_plotly
    def forecast_plot():
        plot_df = forecast_display_data()
        if plot_df is None:
            return None

        node_name = fcast_node_name_val()
        lookback_days = int(input.lookback_days())

        log.info('formatting data for plotting')
        fig = plotting.plotly_forecast(
            plot_df,
            node_name=node_name,
            lookback=f'{lookback_days}D',
            show_fig=False,
        )
        log.info(f'type(fig): {type(fig)}')
        # make the chart fill the available width
        fig.update_layout(width=None, autosize=True)
        # round-trip through plotly's JSON serializer to convert NaN/Inf to null
        fig = pio.from_json(fig.to_json())
        return fig

    @render.ui
    def forecast_data_section():
        '''Show table and download button only after a forecast is generated.'''
        preds = preds_val()
        if preds is None:
            return ui.div()

        return ui.div(
            ui.h3("Forecast data"),
            ui.output_data_frame("forecast_table"),
            ui.download_button("download_data", "Download data"),
        )

    @reactive.calc
    def forecast_table_data():
        '''Prepare display data with string column names for table and download.'''
        plot_df = forecast_display_data()
        if plot_df is None:
            return None

        plot_idx = plotting.get_plot_idx(plot_df)
        display_data = plot_df[plot_idx]
        download_cols = [
            'node', 'time', 'LMP_HOURLY', 'mean_fcast', 0.1, 0.5, 0.9,
            'MTLF', 'Wind_Forecast_MW', 'Solar_Forecast_MW', 'Ratio',
        ]
        display_data = display_data.loc[:, download_cols].copy()
        # rename float quantile columns to strings for Shiny DataGrid compatibility
        display_data = display_data.rename(columns={0.1: 'q10', 0.5: 'q50', 0.9: 'q90'})
        # round numeric columns to 2 decimal places
        numeric_cols = display_data.select_dtypes(include='number').columns
        for col in numeric_cols:
            display_data[col] = display_data[col].map(lambda x: f'{x:.2f}' if pd.notna(x) else '')
        log.info(f'display_data.columns: {display_data.columns}')
        return display_data

    @render.data_frame
    def forecast_table():
        display_data = forecast_table_data()
        if display_data is None:
            return None
        return render.DataGrid(display_data)

    @render.download(
        filename=lambda: (
            f"price-forecast-{fcast_node_name_val()}-{fcast_time_val()}.csv"
        )
    )
    def download_data():
        display_data = forecast_table_data()
        if display_data is None:
            return
        yield display_data.to_csv(index=False)


app = App(app_ui, server, static_assets=Path(__file__).parent / "imgs")

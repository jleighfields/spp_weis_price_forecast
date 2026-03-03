'''
Shiny for Python interface for SPP Weis LMP forecasting endpoint
'''

# pylint: disable=W0621,C0103,W1203

# base imports
import asyncio
import random
import logging
from pathlib import Path
from typing import List

# data
import numpy as np
import pandas as pd

# user interface
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly
import plotly.io as pio
import shinyswatch

# forecasting data
import torch

# custom modules
import src.data_engineering as de
from src import utils
from src import plotting
from src.modeling import load_ensemble_from_dir

# max absolute value for LMPs in given to forecast
MAX_LMP = 200.0

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# load env
from dotenv import load_dotenv
load_dotenv(override=True)


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
        ui.markdown("Data is updated every 6 hours"),
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
                ui.HTML(
                    '<strong><a href="https://pricecontourmap.spp.org/pricecontourmapwest/" target="_blank">SPP Weis price map</a></strong>'
                    ' &nbsp;|&nbsp; '
                    '<strong><a href="https://portal.spp.org/pages/weis-forecast-summary" target="_blank">SPP Weis load and resource forecasts</a></strong>'
                    ' &nbsp;|&nbsp; '
                    '<strong><a href="https://portal.spp.org/pages/weis-generation-mix" target="_blank">SPP Weis generation mix</a></strong>'
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
    # Load data and models on startup (parallel), refresh reloads data only
    ###############################################################

    def _do_load_data():
        '''Blocking: connect to DuckDB/R2 and return (all_df_pd, lmp_pd).'''
        log.info('getting lmp data from R2')
        con = de.create_database()
        log.info('finished getting data from R2')

        log.info('preparing all_df_pd')
        all_df_pd = de.all_df_to_pandas(de.prep_all_df(con))
        log.info('preparing lmp')
        lmp_result = de.prep_lmp(con)
        log.info('preparing lmp_pd_df')
        lmp_pd = lmp_result.to_pandas().set_index('timestamp_mst')
        con.close()
        return all_df_pd, lmp_pd

    def _do_load_models():
        '''Blocking: download champion checkpoints from R2 and return (model, train_timestamp).'''
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            utils.download_champion_checkpoints(tmpdir)
            return load_ensemble_from_dir(tmpdir)

    @reactive.effect
    async def _load_startup():
        """Load data and models on startup; reload only data on refresh.

        On first load, data fetching (DuckDB/R2) and model loading (S3
        checkpoint download + PyTorch) run in parallel via asyncio.gather,
        roughly halving startup time. On subsequent refreshes, only data
        is reloaded since models are already in memory.

        Uses asyncio.to_thread to run blocking I/O off the event loop so
        the Shiny UI stays responsive during loading.
        """
        # Take a reactive dependency on the refresh button.
        # Also fires once on startup because the button value starts at 0.
        input.refresh_data()

        # Use reactive.isolate() to check the model state without
        # subscribing — otherwise setting loaded_model_val below would
        # immediately re-trigger this effect and cause a second load.
        with reactive.isolate():
            need_models = loaded_model_val() is None

        with ui.Progress(min=0, max=2) as p:
            p.set(
                message="Loading data and models..."
                if need_models
                else "Refreshing data...",
            )
            p.set(1, detail="Loading from R2...")

            if need_models:
                # First startup: run data and model loading concurrently.
                # Each helper is blocking I/O, so we push them to threads.
                data_result, model_result = await asyncio.gather(
                    asyncio.to_thread(_do_load_data),
                    asyncio.to_thread(_do_load_models),
                )
                loaded_model_val.set(model_result[0])
                train_timestamp_val.set(str(model_result[1]))
            else:
                # Refresh click: models already loaded, just reload data.
                data_result = await asyncio.to_thread(_do_load_data)

            all_df_pd_val.set(data_result[0])
            lmp_pd_df_val.set(data_result[1])
            p.set(2, detail="Done")

        ui.notification_show("Done loading data and models", type="message")

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

            cov_df = future_cov_series.to_dataframe()
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

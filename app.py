'''
Shiny for Python interface for SPP IM West LMP forecasting endpoint
'''

# pylint: disable=W0621,C0103,W1203

# base imports
import asyncio
import random
import logging
import tempfile
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
from src import parameters
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


###############################################################
# UI Layout
###############################################################

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Forecast type"),
        ui.input_select(
            "target",
            "Market",
            choices={"da": "Day-ahead (DA)", "rt": "Real-time (RT)"},
            selected=parameters.DEFAULT_TARGET,
        ),
        ui.hr(),
        ui.h4("Select forecast start date"),
        ui.input_date("fcast_date", "Forecast date"),
        ui.input_select("fcast_hour", "Forecast hour", choices=[]),
        ui.hr(),
        ui.h4("Select LMP hub"),
        ui.input_select("node_name", "LMP hub", choices=[]),
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
        ui.markdown("Data is updated every 4 hours"),
        ui.markdown("Model last trained:"),
        ui.output_ui("train_timestamp_display"),
        width=300,
    ),
    # Main content
    ui.tags.img(
        src="wind_farm.png",
        style="width: 100%; max-height: 400px; object-fit: cover; display: block;",
    ),
    ui.busy_indicators.use(),
    ui.div(
        ui.br(),
        ui.row(
            ui.column(2, ui.input_action_button("refresh_data", "Refresh data")),
            ui.column(
                10,
                ui.HTML(
                    '<strong><a href="https://pricecontourmap.spp.org/pricecontourmap/" target="_blank">SPP IM price map</a></strong>'
                    ' &nbsp;|&nbsp; '
                    '<strong><a href="https://portal.spp.org/pages/integrated-marketplace-swpw-forecast-vs.-actual" target="_blank">SPP IM West load forecast</a></strong>'
                    ' &nbsp;|&nbsp; '
                    '<strong><a href="https://portal.spp.org/pages/integrated-marketplace-swpw-generation-mix" target="_blank">SPP IM West generation mix</a></strong>'
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
    title="SPP IM West Nodal Price Forecast",
    fillable=False,
    theme=shinyswatch.theme.flatly(),
)


###############################################################
# Server
###############################################################

def _do_load_data(target):
    """Blocking: connect to DuckDB/R2 and return (all_df_pd, lmp_pd) for target.

    Module-level (not a server() closure) so tests/e2e/app_for_test.py can
    monkeypatch it with fixture data — a name re-defined inside server() could
    not be rebound from the module.
    """
    log.info(f'getting {target} lmp data from R2')
    con = de.create_database(target=target)
    log.info('finished getting data from R2')

    log.info('preparing all_df_pd')
    all_df_pd = de.all_df_to_pandas(de.prep_all_df(con))
    log.info('preparing lmp')
    lmp_result = de.prep_lmp(con)
    log.info('preparing lmp_pd_df')
    lmp_pd = lmp_result.to_pandas().set_index('timestamp_mst')
    con.close()
    return all_df_pd, lmp_pd


def _do_load_models(target):
    """Blocking: download the target's champion checkpoints -> (model, train_timestamp).

    Module-level (see _do_load_data) so the e2e harness can monkeypatch it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        utils.download_champion_checkpoints(tmpdir, target=target)
        # Verify the champion was trained on the same covariates this app
        # now builds; a mismatch (e.g. a covariate added/removed since the
        # model was trained) would otherwise surface as a cryptic
        # component-mask error on every forecast. Raise loud instead so the
        # model stays unloaded and the reason is obvious in the logs.
        config = utils.load_training_config(tmpdir)
        if config is not None:
            utils.validate_model_covariates(config, de.FUTR_COLS, de.PAST_COLS)
        return load_ensemble_from_dir(tmpdir)


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
    # Which forecast target (parameters.TARGETS) the loaded data + model are for.
    # None until the first load.
    loaded_target_val = reactive.Value(None)
    # Champion models cached per target, so switching the market reuses an
    # already-downloaded model instead of re-fetching it from R2 each time.
    # Session-scoped: {target: (model, train_timestamp)}.
    model_cache = {}

    ###############################################################
    # Load data and models on startup (parallel), refresh reloads data only.
    # The blocking loaders (_do_load_data / _do_load_models) are module-level
    # above so the e2e harness can monkeypatch them.
    ###############################################################

    @reactive.effect
    async def _load_startup():
        """Load data and model on startup, market switch, or refresh.

        The champion model for a target is downloaded once and cached in
        ``model_cache``, so switching the market to a target loaded earlier this
        session reuses the cached model instead of re-downloading it. Data is
        always (re)loaded for the active target — it is cheap relative to the
        model download and stays fresh. On the first load of a target, data and
        model fetch in parallel via ``asyncio.gather``.

        Uses asyncio.to_thread to run blocking I/O off the event loop so
        the Shiny UI stays responsive during loading.
        """
        # Reactive dependencies: the refresh button (also fires once on startup,
        # value starts at 0) and the market selector (switch loads the new
        # target, from cache if seen before).
        input.refresh_data()
        target = input.target()

        with reactive.isolate():
            need_model = target not in model_cache

        with ui.Progress(min=0, max=2) as p:
            p.set(
                message="Loading data and model..."
                if need_model
                else "Loading data...",
            )
            p.set(1, detail="Loading from R2...")

            if need_model:
                # First load of this target: fetch its data and champion
                # concurrently (both are blocking I/O -> threads), then cache
                # the model.
                data_result, model_result = await asyncio.gather(
                    asyncio.to_thread(_do_load_data, target),
                    asyncio.to_thread(_do_load_models, target),
                )
                model_cache[target] = model_result
            else:
                # Model already cached (from startup or a prior switch): reload
                # only the fresh data for this target.
                data_result = await asyncio.to_thread(_do_load_data, target)

            model, train_ts = model_cache[target]
            loaded_model_val.set(model)
            train_timestamp_val.set(str(train_ts))
            loaded_target_val.set(target)
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
        # default to the SWPW_HUB flagship hub when present
        default_node = 'SWPW_HUB' if 'SWPW_HUB' in nodes else (nodes[0] if nodes else None)
        ui.update_select(
            "node_name",
            choices=dict(zip(nodes, nodes)),
            selected=default_node,
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
    # Clear stale forecast when inputs change
    ###############################################################

    @reactive.effect
    def _clear_stale_forecast():
        # Take dependency on all forecast inputs (incl. the market selector, so
        # switching DA<->RT drops a forecast made for the other market).
        input.node_name()
        input.n_days()
        input.fcast_date()
        input.fcast_hour()
        input.target()
        # Clear previous results so stale data doesn't persist
        preds_val.set(None)

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

        fcast_time = pd.Timestamp(fcast_date) + pd.Timedelta(hours=int(fcast_hour) + 1)
        fcast_node_name_val.set(node_name)
        fcast_time_val.set(fcast_time)
        log.info(f'fcast_time: {fcast_time}')

        log.info('USER INPUTS:')
        log.info(f'\tnode_name: {node_name}')
        log.info(f'\tfcast_time: {fcast_time}')
        log.info(f'\tn_days: {n_days}')

        try:
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
        except Exception as e:
            log.error(f'Forecast failed for {node_name}: {e}')
            ui.notification_show(
                f"Forecast failed for {node_name}: insufficient data or unsupported node.",
                type="error",
                duration=10,
            )

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
        market = {"da": "Day-ahead", "rt": "Real-time"}.get(
            loaded_target_val(), loaded_target_val()
        )
        return ui.div(
            ui.h3(f"{node_name} {market} forecasts"),
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
        # round numeric columns to 2 decimal places (keep as numbers for sortability)
        display_data = display_data.round(2)
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
            f"price-forecast-{fcast_node_name_val()}-{fcast_time_val().strftime('%Y-%m-%dT%H-%M')}.csv"
        )
    )
    def download_data():
        display_data = forecast_table_data()
        if display_data is None:
            return
        yield display_data.to_csv(index=False)


app = App(app_ui, server, static_assets=Path(__file__).parent / "imgs")

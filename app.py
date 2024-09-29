'''
Streamlit interface for SPP Weis LMP forecasting endpoint
'''

# pylint: disable=W0621,C0103,W1203

# base imports
import os
import logging
import json
import datetime
from typing import List, Tuple
import requests

# data
import numpy as np
import pandas as pd

# user interface
import streamlit as st

# forecasting data
from darts import TimeSeries, concatenate
import mlflow

# custom modules
import src.data_engineering as de
from src import plotting
from src import params

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# stream lit configs
st.set_page_config(
    page_title="SPP price forecast",
    layout="wide",
)


###################################################
# get sql warehouse connection
###################################################
from dotenv import load_dotenv
load_dotenv()


###############################################################
# Helper functions
###############################################################

@st.cache_data
def get_price_nodes(lmp_df: pd.DataFrame) -> List[str]:
    '''
    get list of LMP nodes for drop down menu
    args:
        price_df: pd.DataFrame hourly LMP data with 'node' that contains
            LMP names
    returns: List[str] of LMP names
    '''
    price_node_list = lmp_df.unique_id.unique().tolist()

    return price_node_list


# @st.cache_data
def get_hour_list(fcast_date: str, lmp_pd_df: pd.DataFrame) -> List[str]:
    '''
    get list of hours for drop down menu
    args:
        fcast_date: str date returned from st.date_input, i.e. '2023-08-03'
    returns: List[str] of zero padded hours, ['00', '01', ..., '23']
    '''
    # today = datetime.datetime.now()
    # today = datetime.datetime.utcnow()
    today = lmp_pd_df.index.max()
    log.info(f'type(today): {type(today)}')
    log.info(f'today: {today}')

    # if today, then limit list to previous hours
    if fcast_date == today.date():
        last_hour = today.hour
    else:
        last_hour = 23

    hour_list = [str(h).zfill(2) for h in range(last_hour+1)]
    return hour_list


@st.cache_data
def get_fcast_time(fcast_date: str, fcast_hour: str) -> str:
    '''
    get the forecast time, we drop all price observations after this
    time and begin forecasting for the next time step. used in the
    get_forecast() function.
    args:
        fcast_date: str - date from user input selection
        fcast_hour: str - hour from user input selection
    returns: str formated as datetime str
    '''
    return f'{fcast_date}T{fcast_hour}:00:00.000000000'


# cache the model endpoint call so we can change
# the plotting parameters without calling the endpoint again
@st.cache_data
def get_forecast(
    price_df: pd.DataFrame,
    node_name: str,
    mtlf_df: pd.DataFrame,
    mtrf_df: pd.DataFrame,
    fcast_time: str,
    n_days: str,
    ) -> Tuple[str, pd.DataFrame]:
    '''
    formats input for endpoint call and calls endpoint
    args:
        price_df: pd.DataFrame - LMPs data
        node_name: str - from user selection
        mtlf_df: pd.DataFrame - mid-term load forecast data
        mtrf_df: pd.DataFrame - dit-term resource forecast data
        fcast_time: str - datetime str created from
            date and hour user inputs
        n_days: str - number of days to forecast from user inputs
    returns: Tuple[str, pd.DataFrame] - str with json formated
        output from the endpoint call, pd.DataFrame with the
        covariates used in the forecast (this data is used for plotting)
    '''

    # remove duplicate time values
    dups = mtlf_df.GMTIntervalEnd.duplicated()
    log.info(f'mtlf_df duplicated: {dups.sum()}')
    mtlf_df = mtlf_df[~dups]

    dups = mtrf_df.GMTIntervalEnd.duplicated()
    log.info(f'mtrf_df duplicated: {dups.sum()}')
    mtrf_df = mtrf_df[~dups]

    # create data to pass to model endpoint call
    common_times = np.intersect1d(mtlf_df.GMTIntervalEnd, mtlf_df.GMTIntervalEnd)
    mtlf_idx = [t in common_times for t in mtlf_df.GMTIntervalEnd]
    mtrf_idx = [t in common_times for t in mtrf_df.GMTIntervalEnd]

    mtrf_df = mtrf_df[mtrf_idx]
    mtlf_df = mtlf_df[mtlf_idx]
    log.info(f'mtrf_df.shape: {mtrf_df.shape}')
    log.info(f'mtlf_df.shape: {mtlf_df.shape}')

    # mtlf series
    log.info(f'mtlf_df.columns: {mtlf_df.columns}')
    mtlf_series, avg_act_series = de.create_mtlf_series(mtlf_df)

    # mtrf series feature engineering
    mtrf_ratio_df = de.add_enrgy_ratio_to_mtrf(mtlf_df, mtrf_df)
    mtrf_ratio_df = de.add_enrgy_ratio_diff_to_mtrf(mtrf_ratio_df)
    mtrf_series = de.create_mtrf_series(mtrf_ratio_df)

    future_covariates = concatenate([mtlf_series, mtrf_series], axis=1)
    past_covariates = avg_act_series

    plot_cov_df = future_covariates.pd_dataframe()
    plot_cov_df = plot_cov_df.reset_index().rename(columns={'GMTIntervalEnd':'time'})

    lmp_series_df = price_df[price_df.node == node_name].drop('node', axis=1)
    lmp_series_df.rename(columns={'LMP_HOURLY':node_name}, inplace=True)
    lmp_series_df = lmp_series_df.sort_values('time')
    dups = lmp_series_df.time.duplicated()
    log.info(f'lmp_series_df duplicated: {dups.sum()}')
    lmp_series_df = lmp_series_df[~dups]


    lmp_series = TimeSeries.from_dataframe(
            lmp_series_df,
            time_col='time',
            value_cols=node_name,
            fill_missing_dates=True,
            freq='H',
            fillna_value=0,
            static_covariates=None,
            hierarchy=None
        ).astype(np.float32)

    FCAST_TIME = pd.Timestamp(fcast_time)
    # drop after is inclusive...
    if FCAST_TIME < lmp_series.end_time():
        lmp_series = lmp_series.drop_after(FCAST_TIME + pd.Timedelta('1H'))

    # subtract hour to account for the shift above and lag in covariate updates
    forecast_horizon = 24*n_days - 1
    log.info(f'forecast_horizon: {forecast_horizon}')

    data = {
        'series': [lmp_series.to_json()],
        'past_covariates': [past_covariates.to_json()],
        'future_covariates': [future_covariates.to_json()],
        'n': forecast_horizon,
        'num_samples': 200
    }
    df = pd.DataFrame(data)

    log.info('calling endpoint')
    endpoint_pred = score_model(df)
    return endpoint_pred, plot_cov_df


@st.cache_resource
def convert_df(df: pd.DataFrame):
    '''
    save dataframe to csv file for downloading
    args:
        df: pd.DataFrame to be converted to csv file
    '''
    return df.to_csv(index=False).encode('utf-8')


###############################################################
# Design Streamlit app Layout
###############################################################

forcasted_data = st.container()

with forcasted_data:

    from PIL import Image
    # image = Image.open('./imgs/app_background.PNG')
    # st.image(image)

    st.title('SPP Weis Nodal Price Forecast')

    st.session_state.refresh_data = st.button('Refresh data')
    log.info(f'st.session_state.refresh_data: {st.session_state.refresh_data}')
    log.info(f'all_df_pd not in st.session_state: {"all_df_pd" not in st.session_state}')

    ###############################################################
    # Get data
    ###############################################################

    if ('all_df_pd' not in st.session_state):
        log.info('loading data')

        with st.spinner('Loading LMP data...'):
            st.session_state['all_df_pd'] = de.all_df_to_pandas(de.prep_all_df())
            st.session_state['lmp'] = de.prep_lmp()
            st.session_state['lmp_pd_df'] = (
                st.session_state['lmp']
                .to_pandas()
                .set_index('timestamp_mst')
            )

        st.toast('Done loading data')

    if 'loaded_model' not in st.session_state:
        log.info('loading model')

        with st.spinner('Loading model'):
            log.info(f'mlflow.get_tracking_uri(): {mlflow.get_tracking_uri()}')
            exp_name = 'spp_weis'
            exp = mlflow.get_experiment_by_name(exp_name)
            runs = mlflow.search_runs(
                experiment_ids=exp.experiment_id,
                # order_by=['metrics.test_mae']
                order_by=['end_time']
            )
            runs.sort_values('end_time', ascending=False, inplace=True)
            best_run_id = runs.run_id.iloc[0]
            model_path = runs['artifact_uri'].iloc[0] + '/GlobalForecasting'
            st.session_state['loaded_model'] = mlflow.pyfunc.load_model(model_path)

        st.toast('Done loading model')


###############################################################
# Get user iputs
###############################################################
with st.sidebar:

    st.header('Select forecast start date')

    # today = datetime.datetime.now()
    today = st.session_state.lmp_pd_df.index.max()
    current_hour = today.hour
    min_date = today - pd.Timedelta('60D')

    fcast_date = st.date_input('Forecast date', today, min_date, today)
    st.session_state.fcast_date = fcast_date

    fcast_hour = st.selectbox(
        'Forecast hour',
        get_hour_list(st.session_state.fcast_date, st.session_state.lmp_pd_df),
        index=len(get_hour_list(st.session_state.fcast_date, st.session_state.lmp_pd_df))-1,
        )


    log.info(f'\ttoday: {today}')
    log.info(f'\ttoday.date: {today.date()}')
    log.info(f'\tmin_date: {min_date}')
    log.info(f'\tfcast_date: {fcast_date}')
    log.info(f'\tcurrent_hour: {current_hour}')
    log.info(f'\tfcast_hour: {fcast_hour}')

    st.header('Select LMP node')
    with st.form('form1'):
        # get data from session state
        # price_df = st.session_state.price_df
        # mtrf_df = st.session_state.mtrf_df
        # mtlf_df = st.session_state.mtlf_df

        # select node to forecast
        node_name = st.selectbox(
            'LMP node',
            get_price_nodes(st.session_state['lmp_pd_df']),
            index=0)

        # select days to forecast and lookback window for plot
        n_days = st.selectbox('Number of days to forecast', [5,4,3,2,1], index=0)
        lookback_days = st.selectbox('Number of days to lookback', [7,6,5,4,3,2,1], index=0)

        st.session_state.submitted1 = st.form_submit_button('Get forecast')

log.info(f'st.session_state.submitted1: {st.session_state.submitted1}')


###############################################################
# get forecasts
###############################################################

if st.session_state.submitted1:

    fcast_time = get_fcast_time(fcast_date, fcast_hour)
    st.session_state.node_name = node_name
    st.session_state.fcast_time = fcast_time

    log.info('USER INPUTS:')
    log.info(f'\tnode_name: {node_name}')
    log.info(f'\tfcast_time: {fcast_time}')
    log.info(f'\tn_days: {n_days}')


    idx = st.session_state.lmp_pd_df.unique_id == node_name
    price_df = st.session_state.lmp_pd_df[idx]

    idx = st.session_state.all_df_pd.unique_id == node_name
    plot_cov_df = st.session_state.all_df_pd[idx]

    plot_series = de.get_all_series(price_df)[0]
    future_cov_series = de.get_futr_cov(plot_cov_df)[0]
    past_cov_series = de.get_past_cov(plot_cov_df)[0]

    node_series = plot_series.drop_after(pd.Timestamp(fcast_time))

    data = {
        'series': [node_series.to_json()],
        'past_covariates': [past_cov_series.to_json()],
        'future_covariates': [future_cov_series.to_json()],
        'n': params.FORECAST_HORIZON,
        'num_samples': 200
    }
    df = pd.DataFrame(data)

    plot_cov_df = future_cov_series.pd_dataframe()
    plot_cov_df = (
        plot_cov_df
        .reset_index()
        .rename(columns={'timestamp_mst': 'time', 're_ratio': 'Ratio'})
    )

    # Predict on a Pandas DataFrame.
    df['num_samples'] = 500
    preds = st.session_state.loaded_model.predict(df)
    preds = TimeSeries.from_json(preds)

    st.session_state.preds = preds
    st.session_state.plot_cov_df = plot_cov_df


###############################################################
# Plot forecasts
###############################################################

if 'preds' in st.session_state:
    node_name = st.session_state.node_name
    fcast_time = st.session_state.fcast_time

    st.write(f'### {node_name} forcasts')
    st.write(f'Forecast start time: {fcast_time}')

    preds = st.session_state.preds
    plot_cov_df = st.session_state.plot_cov_df

    log.info('formatting data for plotting')

    q_df = plotting.get_quantile_df(preds)
    # plot_df = plotting.get_mean_df(preds).merge(
    #     plotting.get_quantile_df(preds),
    #     left_index=True,
    #     right_index=True,
    # )

    lmp_df = st.session_state.lmp.to_pandas().rename(
        columns={
            'LMP': 'LMP_HOURLY',
            'unique_id': 'node',
            'timestamp_mst': 'time'
        })

    plot_df = plotting.get_plot_df(
        preds,
        plot_cov_df,
        lmp_df,
        node_name,
    )
    plot_df.rename(columns={'mean': 'mean_fcast'}, inplace=True)

    # fig = plotting.plot_fcast(plot_df, node_name, lookback = f'{lookback_days}D')
    # st.pyplot(fig)
    fig = plotting.plotly_forecast(
        plot_df,
        node_name=node_name,
        lookback=f'{lookback_days}D',
        show_fig=False
        )

    log.info(f'type(fig): {type(fig)}')
    st.plotly_chart(fig, node_name=node_name, use_container_width=False)

    plot_idx = plotting.get_plot_idx(plot_df)
    display_data = plot_df[plot_idx]
    csv = convert_df(display_data)

    st.write('### Forcast data')
    st.dataframe(display_data)

    # the download button resets the page see this link
    # for download link workaround
    # https://github.com/streamlit/streamlit/issues/4382
    # st.download_button(
    #     "Download data",
    #     csv,
    #     f"price-forecast-{node_name}-{fcast_time}.csv",
    #     "text/csv",
    #     key='download-csv'
    #     )
    import base64
    def create_download_link(val, filename):
        '''
        create a link to download csv data
        '''
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.csv">Download file</a>'

    download_url = create_download_link(csv, f"price-forecast-{node_name}-{fcast_time}.csv")
    st.markdown(download_url, unsafe_allow_html=True)

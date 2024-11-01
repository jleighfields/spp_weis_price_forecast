'''
Streamlit interface for SPP Weis LMP forecasting endpoint
'''

# pylint: disable=W0621,C0103,W1203

# base imports
import os
import pickle
import logging
from typing import List

# data
import pandas as pd
import ibis

# user interface
import streamlit as st

# forecasting data
from darts import TimeSeries, concatenate
import mlflow

# custom modules
import src.data_engineering as de
from src import plotting
from src import parameters

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
    returns: str formatted as datetime str
    '''
    return f'{fcast_date}T{fcast_hour}:00:00.000000000'


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
    image = Image.open('./imgs/wind_farm.png')
    st.image(image)

    st.title('SPP Weis Nodal Price Forecast')
    st.write('**[SPP Weis price map](https://pricecontourmap.spp.org/pricecontourmapwest/)**')

    st.session_state.refresh_data = st.button('Refresh data')
    log.info(f'st.session_state.refresh_data: {st.session_state.refresh_data}')
    log.info(f'all_df_pd not in st.session_state: {"all_df_pd" not in st.session_state}')

    ###############################################################
    # Get data
    ###############################################################

    if ('all_df_pd' not in st.session_state) or st.session_state.refresh_data:
        log.info('loading data')

        with st.spinner('Loading LMP data...'):
            con = ibis.duckdb.connect("data/spp.ddb", read_only=True)
            # con = ibis.duckdb.connect(
            #     "/teamspace/studios/data-collection/spp_weis_price_forecast/data/spp.ddb", 
            #     read_only=True
            #     )
            st.session_state['all_df_pd'] = de.all_df_to_pandas(de.prep_all_df(con))
            st.session_state['lmp'] = de.prep_lmp(con)
            st.session_state['lmp_pd_df'] = (
                st.session_state['lmp']
                .to_pandas()
                .set_index('timestamp_mst')
            )
            con.disconnect()

        st.toast('Done loading data')

    if 'loaded_model' not in st.session_state:
        log.info('loading model')

        with st.spinner('Loading model'):
            os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlruns.db'
            # os.environ['MLFLOW_TRACKING_URI'] = 'file:///teamspace/studios/model-train/spp_weis_price_forecast/mlruns'
            log.info(f'mlflow.get_tracking_uri(): {mlflow.get_tracking_uri()}')

            # model uri for the above model
            model_uri = "models:/spp_weis@champion"

            # Load the model and access the custom metadata
            loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
            log.info(f'loaded_model: {loaded_model}')
            st.session_state['loaded_model'] = loaded_model

            # get model training timestamp
            load_model_dict = loaded_model.metadata.to_dict()
            from mlflow import MlflowClient
            client = MlflowClient()
            local_dir = "./" # existing and accessible DBFS folder
            run_id = load_model_dict['run_id']
            artifact_path = 'GlobalForecasting/artifacts/TRAIN_TIMESTAMP.pkl'
            local_path = client.download_artifacts(run_id, artifact_path, local_dir)
            with open(artifact_path, 'rb') as handle:
                TRAIN_TIMESTAMP = pickle.load(handle)
            
            log.info(f'TRAIN_TIMESTAMP: {TRAIN_TIMESTAMP}')
            os.remove(artifact_path)

            st.session_state['TRAIN_TIMESTAMP'] = TRAIN_TIMESTAMP

        st.toast('Done loading model')


###############################################################
# Get user inputs
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

        # select node to forecast
        node_name = st.selectbox(
            'LMP node',
            get_price_nodes(st.session_state['lmp_pd_df']),
            index=0)

        # select days to forecast and lookback window for plot
        n_days = st.selectbox('Number of days to forecast', [5,4,3,2,1], index=0)
        lookback_days = st.selectbox('Number of days to lookback', [7,6,5,4,3,2,1], index=0)

        st.session_state.get_fcast_btn = st.form_submit_button('Get forecast')

    # model_trained = st.session_state.loaded_model.TRAIN_TIMESTAMP
    st.markdown('**NOTES:**')
    st.markdown('Data is updated every 4-6 hours')
    st.markdown('Model last trained:')
    st.markdown(f'**{st.session_state.TRAIN_TIMESTAMP}**')
    

log.info(f'st.session_state.submitted1: {st.session_state.get_fcast_btn}')


###############################################################
# get forecasts
###############################################################

if st.session_state.get_fcast_btn:

    fcast_time = get_fcast_time(fcast_date, fcast_hour)
    fcast_time = pd.Timestamp(fcast_time) + pd.Timedelta('1h')
    st.session_state.node_name = node_name
    st.session_state.fcast_time = fcast_time
    log.info(f'fcast_time: {fcast_time}')

    log.info('USER INPUTS:')
    log.info(f'\tnode_name: {node_name}')
    log.info(f'\tfcast_time: {fcast_time}')
    log.info(f'\tn_days: {n_days}')

    # get prices for user selected node
    idx = st.session_state.lmp_pd_df.unique_id == node_name
    price_df = st.session_state.lmp_pd_df[idx]

    # get covariates for user selected node
    idx = st.session_state.all_df_pd.unique_id == node_name
    plot_cov_df = st.session_state.all_df_pd[idx]

    # prepare data for getting predictions
    plot_series = de.get_series(price_df)[0]
    future_cov_series = de.get_futr_cov(plot_cov_df)[0]
    past_cov_series = de.get_past_cov(plot_cov_df)[0]
    node_series = plot_series
    if fcast_time <= node_series.end_time():
        node_series = node_series.drop_after(fcast_time)

    data = {
        'series': [node_series.to_json()],
        'past_covariates': [past_cov_series.to_json()],
        'future_covariates': [future_cov_series.to_json()],
        'n': parameters.FORECAST_HORIZON,
        'num_samples': 200
    }
    df = pd.DataFrame(data)

    # Predict on a Pandas DataFrame.
    df['num_samples'] = 500
    preds_json = st.session_state.loaded_model.predict(df)
    preds = TimeSeries.from_json(preds_json)

    plot_cov_df = future_cov_series.pd_dataframe()
    plot_cov_df = (
        plot_cov_df
        .reset_index()
        .rename(columns={'timestamp_mst': 'time', 're_ratio': 'Ratio'})
    )

    # save data for plotting
    st.session_state.preds = preds
    st.session_state.plot_cov_df = plot_cov_df


###############################################################
# Plot forecasts
###############################################################

if 'preds' in st.session_state:
    node_name = st.session_state.node_name
    fcast_time = st.session_state.fcast_time

    st.write(f'### {node_name} forecasts')
    st.write(f'Forecast start time: {fcast_time}')

    preds = st.session_state.preds
    plot_cov_df = st.session_state.plot_cov_df

    log.info('formatting data for plotting')

    q_df = plotting.get_quantile_df(preds)

    lmp_df = (
        st.session_state.lmp_pd_df
        .reset_index()
        .rename(
        columns={
            'LMP': 'LMP_HOURLY',
            'unique_id': 'node',
            'timestamp_mst': 'time'
        })
    )

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
    # configure columns for displaying and downloading data
    download_cols = [
        'node', 'time', 'LMP_HOURLY', 'mean_fcast', 0.1, 0.5, 0.9,
        'MTLF', 'Wind_Forecast_MW', 'Solar_Forecast_MW', 'Ratio'
        ]
    display_data = display_data.loc[:, download_cols]
    log.info(f'display_data.columns: {display_data.columns}')
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

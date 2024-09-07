'''
plotting utilities for app
'''

# pylint: disable=W0621,C0103,W1203,R1735

# base python
import logging
import warnings

# data
import numpy as np
import pandas as pd

# modeling
from sklearn.metrics import mean_absolute_error as mae

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

# forecasting data
from darts import TimeSeries

# define log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# settings
warnings.filterwarnings("ignore")


def get_quantile_df(preds: TimeSeries) -> pd.DataFrame:
    '''
    create a quantile dataframe from the n forecasts
    returned from the endpoint, this will be used to
    plot confidence intervals
    args:
        preds: TimeSeries - TimeSeries object created from
            endpoint response in json format that will have
            a number (n) of simulations based on the api call,
            i.e. preds = TimeSeries.from_json(endpoint_response)
    returns: pd.DataFrame
    '''

    # get dataframe from preds TimeSeries
    plot_df = (
        preds.pd_dataframe()
        .reset_index()
        .melt(id_vars='time')
        .rename(columns={'component':'node'})
    )

    # remove sample numbers
    plot_df.node = ['_'.join(n.split('_')[:-1]) for n in plot_df.node]

    # get quanitles
    q_df = plot_df.groupby(['time', 'node']).quantile([0.1, 0.5, 0.9])

    # create columns from quantiles
    q_pivot = q_df.reset_index().pivot(columns='level_2', index=['time', 'node'])

    # level from columns after pivot
    q_pivot.columns = q_pivot.columns.droplevel()

    # remove index level name
    q_pivot.columns.name = None

    return q_pivot


def get_mean_df(preds: TimeSeries) -> pd.DataFrame:
    '''
    get the mean forecast from the n forecasts
    returned from the endpoint, this will be used to
    plot the expected forecast
    args:
        preds: TimeSeries - TimeSeries object created from
            endpoint response in json format that will have
            a number (n) of simulations based on the api call,
            i.e. preds = TimeSeries.from_json(endpoint_response)
    returns: pd.DataFrame
    '''

    plot_df = (
        preds.pd_dataframe()
        .reset_index()
        .melt(id_vars='time')
        .rename(columns={'component':'node'})
    )

    # remove sample numbers
    plot_df.node = ['_'.join(n.split('_')[:-1]) for n in plot_df.node]

    # get mean
    mean_df = plot_df.groupby(['time', 'node']).mean()
    mean_df.rename(columns={'value':'mean_fcast'}, inplace=True)
    return mean_df


def get_plot_df(
        preds: TimeSeries,
        plot_cov_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        node_name: str,
    ) -> pd.DataFrame:
    '''
    get the mean forecast and quantiles from the n forecasts
    returned from the endpoint, this will be used to
    plot the expected forecast
    args:
        preds: TimeSeries - TimeSeries object created from
            endpoint response in json format that will have
            a number (n) of simulations based on the api call,
            i.e. preds = TimeSeries.from_json(endpoint_response)
        plot_cov_df: pd.DataFrame - contains all the covariates used
            in the forecast, returned from the get_forecast() function
            in the app.py file
        prices_df: pd.DataFrame - hourly lmp prices
        node_name: str - name of the node used to filter prices_df
    returns: pd.DataFrame - contains the data needed for plotting
    '''

    fcast_df = get_mean_df(preds).merge(
        get_quantile_df(preds),
        left_index=True,
        right_index=True,
    )

    fcast_df.reset_index().drop('node', axis=1)

    plot_df = fcast_df.reset_index().drop('node', axis=1).merge(
        plot_cov_df,
        on=['time'],
        how='right',
    ).sort_values('time')

    plot_df = plot_df.merge(
        prices_df[prices_df.node == node_name],
        on=['time'],
        how='left',
    ).sort_values('time')

    return plot_df

def get_plot_idx(
        plot_df: pd.DataFrame,
        lookback: str='7D'
    ) -> pd.Series:
    '''
    get the boolean series used to filter the plotting data to
    the forecast horizon and lookback period
    args:
        plot_df: pd.DataFrame - contains the data needed for plotting,
            returned from get_plot_df()
        lookback: str - string formatted for pd.Timedelta to create
            lookback period
    returns: pd.Series - boolean series used to filter plot_df for
        the timeframe to plot
    '''

    min_fcast_time = plot_df.time[~plot_df.mean_fcast.isna()].min() - pd.Timedelta(lookback)
    max_fcast_time = plot_df.time[~plot_df.mean_fcast.isna()].max()
    plot_idx = (plot_df.time >= min_fcast_time) & (plot_df.time <= max_fcast_time)
    return plot_idx


def plot_fcast(
        plot_df: pd.DataFrame,
        node_name: str=None,
        lookback: str='7D',
        show_plot: bool=False
    ) -> mpl.figure.Figure:

    '''
    create a matplotlib figure to display in the app
    args:
        plot_df: pd.DataFrame - contains the data needed for plotting,
            returned from get_plot_df()
        node_name: str - inserted in plot title if not None
        lookback: str - string formatted for pd.Timedelta to create
            lookback period
        show_plot: bool - whether to display plot
    returns: mpl.figure.Figure - matplotlib figure
    '''

    fig, (ax1, ax2) = plt.subplots(2)

    # min_fcast_time = plot_df.time[~plot_df.mean_fcast.isna()].min() - pd.Timedelta(lookback)
    # max_fcast_time = plot_df.time[~plot_df.mean_fcast.isna()].max()
    # plot_idx = (plot_df.time >= min_fcast_time) & (plot_df.time <= max_fcast_time)
    plot_idx = get_plot_idx(plot_df, lookback)

    plot_data = plot_df[plot_idx]
    plot_data[['time', 'mean_fcast', 'LMP_HOURLY']].plot(x='time', ax=ax1)
    idx = ~plot_data['mean_fcast'].isna()

    acc_data = plot_data[['mean_fcast', 'LMP_HOURLY']].dropna(axis=0)
    if acc_data.shape[0] > 0:
        err = np.round(mae(acc_data.LMP_HOURLY, acc_data.mean_fcast), 2)
    else:
        err = '-'
    title_text = f'MAE forecast error: ${err}'
    if node_name:
        title_text = node_name + '\n' + title_text

    # https://stackoverflow.com/questions/29329725/pandas-and-matplotlib-fill-between-vs-datetime64/29329823#29329823
    ax1.fill_between(plot_data.time.values, plot_data[0.1], plot_data[0.9], where=idx, alpha=0.3)
    ax1.set_xlabel('')
    ax1.set_ylabel('$')
    ax1.set_title(title_text)

    plot_df.loc[plot_idx, ['time', 'Ratio']].plot(x='time', ax=ax2)
    ax2.set_ylabel('RE gen / load')

    fig.set_size_inches(6, 6)
    plt.tight_layout()

    plt.show()

    if show_plot:
        plt.show()

    return fig


def plotly_forecast(
        plot_df: pd.DataFrame,
        node_name: str=None,
        lookback: str='7D',
        show_fig=False,
        is_job=False
    ) -> plotly.graph_objs._figure.Figure:
    '''
    create a plotly figure to display in the app and in the job notebook.
    if is_job is True, that plot is used in the job and does not show the plot for ratio.
    args:
        plot_df: pd.DataFrame - contains the data needed for plotting,
            returned from get_plot_df()
        node_name: str - inserted in plot title if not None
        lookback: str - string formatted for pd.Timedelta to create
            lookback period
        show_fig: bool - whether to display plot
        is_job: bool - whether the plot is for jobs orthe app
    returns: plotly.graph_objs._figure.Figure - plotly figure
    '''

    # start_time = plot_df[~plot_df.mean_fcast.isna()].time.min() - pd.Timedelta(lookback)
    # end_time = plot_df[~plot_df.mean_fcast.isna()].time.max()
    # plotly_idx = (plot_df.time >= start_time) & (plot_df.time <= end_time)
    plot_idx = get_plot_idx(plot_df, lookback)
    plotly_data = plot_df[plot_idx]

    x_actual = plotly_data.time
    y_actual = plotly_data.LMP_HOURLY

    # confidence interval
    x_fcast = plotly_data.time
    y_u_int = plotly_data[0.9]
    y_l_int = plotly_data[0.1]

    # mean forecast and RE ratio
    y_fcast = plotly_data.mean_fcast
    y_ratio = plotly_data.Ratio

    # get accuracy
    acc_data = plotly_data[['mean_fcast', 'LMP_HOURLY']].dropna(axis=0)
    if acc_data.shape[0] > 0:
        err = np.round(mae(acc_data.LMP_HOURLY, acc_data.mean_fcast), 2)
    else:
        err = '-'

    ci_idx = ~plotly_data.mean_fcast.isna()
    plotly_data['ci_error'] = (plotly_data.LMP_HOURLY < plotly_data[0.1]) | (plotly_data.LMP_HOURLY > plotly_data[0.9])
    ci_error = plotly_data.ci_error[ci_idx].mean()

    # create title
    title_text = f'MAE forecast error: ${err} - CI coverage: {1-ci_error:0.3f}'
    log.info(f'node_name: {node_name}')
    log.info(f'title_text: {title_text}')
    if node_name:
        title_text = node_name + ' - ' + title_text
    log.info(f'title_text: {title_text}')

    # set up figure
    if is_job:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    else:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)

    # confidence interval
    fig.append_trace(
        go.Scatter(
                x=(
                    x_fcast.tolist() +
                    x_fcast[::-1].tolist()
                ),
                y=(
                    y_u_int.tolist() +
                    y_l_int[::-1].tolist()
                ),
                fill='toself',
                fillcolor='rgba(200,40,40,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='Forecast CI'
                ), row=1, col=1)
    
    # Actual values (history)
    fig.append_trace(
        go.Scatter(
                x=x_actual,
                y=y_actual,
                line=dict(color='rgb(10,10,10)'),
                mode='lines',
                name='Actual'
            ), row=1, col=1)

    # point forecast
    fig.append_trace(
        go.Scatter(
                x=x_fcast,
                y=y_fcast,
                line=dict(color='rgb(200,40,40)'),
                mode='lines',
                name='Forecast'
            ), row=1, col=1)

    if not is_job:    
        # energy ratio
        fig.append_trace(
            go.Scatter(
                    x=x_fcast,
                    y=y_ratio,
                    line=dict(color='rgb(40,40,200)'),
                    mode='lines',
                    name='energy_ratio',
                ), row=2, col=1,
        )


    fig.update_layout(
            title=title_text,
            height=800,
            width=1000,
            yaxis_tickformat = ',',
            plot_bgcolor="rgb(240, 240, 250, 1.0)",
            xaxis=dict(
                type="date"
            )
        )

    if is_job:
        fig.update_yaxes(title_text="$")
    else:
        fig.update_yaxes(title_text="$", row=1, col=1)
        fig.update_yaxes(title_text=" RE / Load", row=2, col=1)
        # range slider for subplots
        # https://community.plotly.com/t/subplot-with-shared-x-axis-and-range-slider/3148
        fig.update_layout(xaxis2_rangeslider_visible=True,
                        xaxis2_rangeslider_thickness=0.1)


    fig.for_each_trace(lambda t: t.update(
        hoveron='points+fills',
        hovertemplate = '%{y:,.2f} - %{x}',
        ))

    if show_fig:
        fig.show()

    return fig

"""
Plotting utilities for the SPP WEIS LMP forecasting Streamlit app.

This module provides functions to create visualizations of LMP price forecasts,
including:

- Quantile-based confidence intervals from probabilistic forecasts
- Mean forecast extraction and plotting
- Interactive Plotly and static Matplotlib visualizations
- Forecast accuracy metrics display (MAE, bias, CI coverage)

Dependencies:
    - matplotlib: Static plot generation
    - plotly: Interactive plot generation
    - darts: TimeSeries data handling
    - sklearn: Error metric calculations
"""

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


def get_quantile_df(preds: TimeSeries, node_name: str) -> pd.DataFrame:
    """
    Create a quantile DataFrame from forecast predictions.

    Extracts 10th, 50th, and 90th percentiles from probabilistic forecast
    samples to create confidence interval bounds for plotting.

    Args:
        preds: TimeSeries object containing forecast samples, typically
            created from endpoint response or model predictions.
        node_name: Price node identifier to label the output.

    Returns:
        pd.DataFrame: Pivoted DataFrame with columns [0.1, 0.5, 0.9] representing
            quantiles, indexed by (time, node).
    """

    # get dataframe from preds TimeSeries
    plot_df = (
        preds.pd_dataframe()
        .reset_index()
        .rename(columns={'timestamp_mst': 'time'})
        .melt(id_vars='time')
        .rename(columns={'component':'node'})
    )

    # remove sample numbers
    # plot_df.node = ['_'.join(n.split('_')[:-1]) for n in plot_df.node]
    plot_df['node'] = node_name
    log.info(plot_df.head())

    # get quanitles
    # q_df = plot_df.groupby(['time', 'node']).quantile([0.1, 0.5, 0.9])
    q_df = plot_df.drop('variable', axis=1).groupby(['time', 'node']).quantile([0.1, 0.5, 0.9])

    # create columns from quantiles
    q_pivot = q_df.reset_index().pivot(columns='level_2', index=['time', 'node'])

    # level from columns after pivot
    q_pivot.columns = q_pivot.columns.droplevel()

    # remove index level name
    q_pivot.columns.name = None

    return q_pivot


def get_mean_df(preds: TimeSeries, node_name: str) -> pd.DataFrame:
    """
    Extract mean forecast from probabilistic predictions.

    Computes the mean across forecast samples to get the expected
    point forecast for plotting.

    Args:
        preds: TimeSeries object containing forecast samples, typically
            created from endpoint response or model predictions.
        node_name: Price node identifier to label the output.

    Returns:
        pd.DataFrame: DataFrame with 'mean_fcast' column, indexed by (time, node).
    """

    plot_df = (
        preds.pd_dataframe()
        .reset_index()
        .rename(columns={'timestamp_mst': 'time'})
        .melt(id_vars='time')
        .rename(columns={'component':'node'})
    )

    # remove sample numbers
    # plot_df.node = ['_'.join(n.split('_')[:-1]) for n in plot_df.node]
    plot_df['node'] = node_name

    # get mean
    mean_df = plot_df.drop('variable', axis=1).groupby(['time', 'node']).mean()
    mean_df.rename(columns={'value':'mean_fcast'}, inplace=True)
    return mean_df


def get_plot_df(
        preds: TimeSeries,
        plot_cov_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        node_name: str,
    ) -> pd.DataFrame:
    """
    Combine forecast predictions with covariates and actual prices.

    Merges mean forecast, quantiles, covariates, and actual LMP prices
    into a single DataFrame ready for visualization.

    Args:
        preds: TimeSeries object containing forecast samples.
        plot_cov_df: DataFrame with covariates used in the forecast,
            returned from get_forecast() in app.py.
        prices_df: DataFrame with hourly LMP prices containing 'node'
            and 'time' columns.
        node_name: Price node identifier to filter prices_df.

    Returns:
        pd.DataFrame: Combined DataFrame with columns for forecast,
            quantiles, covariates, and actual prices, sorted by time.
    """

    fcast_df = get_mean_df(preds, node_name).merge(
        get_quantile_df(preds, node_name),
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
        lookback: str = '7D'
    ) -> pd.Series:
    """
    Get boolean index for filtering plot data to display window.

    Creates a boolean Series to filter data from lookback period before
    the first forecast through the last forecast timestamp.

    Args:
        plot_df: DataFrame from get_plot_df() containing 'time' and
            'mean_fcast' columns.
        lookback: Timedelta-compatible string for lookback period
            (e.g., '7D' for 7 days, '24H' for 24 hours).

    Returns:
        pd.Series: Boolean series where True indicates rows within the
            display window.
    """

    min_fcast_time = plot_df.time[~plot_df.mean_fcast.isna()].min() - pd.Timedelta(lookback)
    max_fcast_time = plot_df.time[~plot_df.mean_fcast.isna()].max()
    plot_idx = (plot_df.time >= min_fcast_time) & (plot_df.time <= max_fcast_time)
    return plot_idx


def plot_fcast(
        plot_df: pd.DataFrame,
        node_name: str = None,
        lookback: str = '7D',
        show_plot: bool = False
    ) -> mpl.figure.Figure:
    """
    Create a matplotlib figure displaying forecast with confidence interval.

    Generates a two-panel figure: top panel shows actual vs forecast prices
    with 80% confidence band, bottom panel shows net load (load minus renewables).

    Args:
        plot_df: DataFrame from get_plot_df() with forecast and actual data.
        node_name: Price node name for plot title. If None, omitted from title.
        lookback: Timedelta-compatible string for historical data display.
        show_plot: If True, calls plt.show() to display the figure.

    Returns:
        mpl.figure.Figure: Matplotlib figure with forecast visualization.
    """

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

    # plot_df.loc[plot_idx, ['time', 'Ratio']].plot(x='time', ax=ax2)
    plot_df.loc[plot_idx, ['time', 'load_net_re']].plot(x='time', ax=ax2)
    ax2.set_ylabel('RE gen / load')

    fig.set_size_inches(6, 6)
    plt.tight_layout()

    plt.show()

    if show_plot:
        plt.show()

    return fig


def plotly_forecast(
        plot_df: pd.DataFrame,
        node_name: str = None,
        lookback: str = '7D',
        show_fig: bool = False,
        is_job: bool = False
    ) -> plotly.graph_objs._figure.Figure:
    """
    Create an interactive Plotly figure for forecast visualization.

    Generates an interactive figure with actual prices, forecast, and
    confidence interval. Includes accuracy metrics (MAE, bias, CI coverage)
    in the title. When is_job=False, includes a secondary panel for net load.

    Args:
        plot_df: DataFrame from get_plot_df() with forecast and actual data.
        node_name: Price node name for plot title. If None, omitted from title.
        lookback: Timedelta-compatible string for historical data display.
        show_fig: If True, calls fig.show() to display the figure.
        is_job: If True, creates single-panel plot for batch jobs.
            If False, creates two-panel plot with net load for app display.

    Returns:
        plotly.graph_objs._figure.Figure: Interactive Plotly figure with
            hover tooltips and optional range slider.
    """

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
    # y_ratio = plotly_data.Ratio
    y_ax_2 = plotly_data.load_net_re

    # get accuracy
    acc_data = plotly_data[['mean_fcast', 'LMP_HOURLY']].dropna(axis=0)
    if acc_data.shape[0] > 0:
        err = np.round(mae(acc_data.LMP_HOURLY, acc_data.mean_fcast), 2)
        bias = np.round(np.mean(acc_data.mean_fcast - acc_data.LMP_HOURLY), 2)
    else:
        err = '-'
        bias = '-'

    ci_idx = (~plotly_data.mean_fcast.isna()) & (~plotly_data.LMP_HOURLY.isna())
    plotly_data['ci_error'] = (plotly_data.LMP_HOURLY < plotly_data[0.1]) | (plotly_data.LMP_HOURLY > plotly_data[0.9])
    ci_error = plotly_data.ci_error[ci_idx].mean()

    # create title
    title_text = f'MAE forecast error: ${err} - Bias: ${bias} - CI coverage: {1-ci_error:0.3f}'
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
                    y=y_ax_2,
                    line=dict(color='rgb(40,40,200)'),
                    mode='lines',
                    name='Net load',
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
        fig.update_yaxes(title_text=" Net load", row=2, col=1)
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

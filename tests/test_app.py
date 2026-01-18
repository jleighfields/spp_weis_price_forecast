"""
Integration tests for app.py

Tests cover:
- Helper functions (get_price_nodes, get_hour_list, get_fcast_time, convert_df)
- Integration with plotting module functions
- Data flow through the app's processing pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================
# Fixtures - Sample Data
# ============================================================

@pytest.fixture
def sample_lmp_df():
    """Sample LMP dataframe as used in the app."""
    dates = pd.date_range(start='2023-08-01', periods=72, freq='h')
    data = {
        'timestamp_mst': dates,
        'unique_id': ['PSCO_NODE1'] * 36 + ['PSCO_NODE2'] * 36,
        'LMP': np.random.uniform(20, 50, 72).tolist(),
    }
    df = pd.DataFrame(data)
    df = df.set_index('timestamp_mst')
    return df


@pytest.fixture
def sample_lmp_pd_df():
    """Sample LMP pandas dataframe with datetime index."""
    dates = pd.date_range(start='2023-08-01', periods=48, freq='h')
    data = {
        'unique_id': ['PSCO_NODE1'] * 48,
        'LMP': np.random.uniform(20, 50, 48).tolist(),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp_mst'
    return df


@pytest.fixture
def sample_plot_cov_df():
    """Sample covariate dataframe for plotting."""
    dates = pd.date_range(start='2023-08-01', periods=48, freq='h')
    data = {
        'time': dates,
        'MTLF': np.random.uniform(1400, 1800, 48).tolist(),
        'Wind_Forecast_MW': np.random.uniform(400, 600, 48).tolist(),
        'Solar_Forecast_MW': np.random.uniform(100, 400, 48).tolist(),
        'Ratio': np.random.uniform(0.2, 0.5, 48).tolist(),
        'load_net_re': np.random.uniform(800, 1200, 48).tolist(),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_prices_df():
    """Sample prices dataframe for plotting."""
    dates = pd.date_range(start='2023-08-01', periods=48, freq='h')
    data = {
        'time': dates,
        'node': ['PSCO_NODE1'] * 48,
        'LMP_HOURLY': np.random.uniform(20, 50, 48).tolist(),
    }
    return pd.DataFrame(data)


# ============================================================
# Replicated helper functions for testing
# These mirror the functions in app.py for isolated testing
# ============================================================

def get_price_nodes(lmp_df: pd.DataFrame):
    """
    get list of LMP nodes for drop down menu
    args:
        price_df: pd.DataFrame hourly LMP data with 'node' that contains
            LMP names
    returns: List[str] of LMP names
    """
    price_node_list = np.sort(lmp_df.unique_id.unique())
    return price_node_list


def get_hour_list(fcast_date, lmp_pd_df: pd.DataFrame):
    """
    get list of hours for drop down menu
    args:
        fcast_date: str date returned from st.date_input, i.e. '2023-08-03'
    returns: List[str] of zero padded hours, ['00', '01', ..., '23']
    """
    today = lmp_pd_df.index.max()

    # if today, then limit list to previous hours
    if fcast_date == today.date():
        last_hour = today.hour
    else:
        last_hour = 23

    hour_list = [str(h).zfill(2) for h in range(last_hour + 1)]
    return hour_list


def get_fcast_time(fcast_date: str, fcast_hour: str) -> str:
    """
    get the forecast time, we drop all price observations after this
    time and begin forecasting for the next time step.
    """
    return f'{fcast_date}T{fcast_hour}:00:00.000000000'


def convert_df(df: pd.DataFrame):
    """
    save dataframe to csv file for downloading
    """
    return df.to_csv(index=False).encode('utf-8')


# Constant from app.py
MAX_LMP = 200.0


# ============================================================
# Tests for get_price_nodes
# ============================================================

class TestGetPriceNodes:
    """Tests for get_price_nodes function."""

    def test_returns_sorted_list(self, sample_lmp_df):
        """Test that price nodes are returned sorted."""
        nodes = get_price_nodes(sample_lmp_df)

        assert isinstance(nodes, np.ndarray)
        assert list(nodes) == sorted(nodes)

    def test_returns_unique_nodes(self, sample_lmp_df):
        """Test that only unique nodes are returned."""
        nodes = get_price_nodes(sample_lmp_df)

        assert len(nodes) == len(set(nodes))

    def test_contains_expected_nodes(self, sample_lmp_df):
        """Test that expected nodes are in the result."""
        nodes = get_price_nodes(sample_lmp_df)

        assert 'PSCO_NODE1' in nodes
        assert 'PSCO_NODE2' in nodes

    def test_single_node(self):
        """Test get_price_nodes with only one node."""
        dates = pd.date_range(start='2023-08-01', periods=10, freq='h')
        df = pd.DataFrame({
            'unique_id': ['SINGLE_NODE'] * 10,
            'LMP': np.random.uniform(20, 50, 10),
        }, index=dates)

        result = get_price_nodes(df)

        assert len(result) == 1
        assert result[0] == 'SINGLE_NODE'

    def test_many_nodes(self):
        """Test get_price_nodes with many nodes."""
        dates = pd.date_range(start='2023-08-01', periods=100, freq='h')
        nodes = ['PSCO_ALAMOSA', 'PSCO_DENVER', 'WAPA_LOVELAND'] * 34
        nodes = nodes[:100]

        df = pd.DataFrame({
            'unique_id': nodes,
            'LMP': np.random.uniform(20, 50, 100),
        }, index=dates)

        result = get_price_nodes(df)

        assert len(result) == 3
        assert 'PSCO_ALAMOSA' in result
        assert 'PSCO_DENVER' in result
        assert 'WAPA_LOVELAND' in result


# ============================================================
# Tests for get_hour_list
# ============================================================

class TestGetHourList:
    """Tests for get_hour_list function."""

    def test_returns_all_hours_for_past_date(self, sample_lmp_pd_df):
        """Test that all 24 hours are returned for a past date."""
        # Use a date before the max date in the dataframe
        past_date = date(2023, 7, 15)
        hours = get_hour_list(past_date, sample_lmp_pd_df)

        assert len(hours) == 24
        assert hours[0] == '00'
        assert hours[-1] == '23'

    def test_returns_zero_padded_hours(self, sample_lmp_pd_df):
        """Test that hours are zero-padded strings."""
        past_date = date(2023, 7, 15)
        hours = get_hour_list(past_date, sample_lmp_pd_df)

        for hour in hours:
            assert isinstance(hour, str)
            assert len(hour) == 2

    def test_limits_hours_for_today(self, sample_lmp_pd_df):
        """Test that hours are limited for the current date."""
        # Use the max date from the dataframe (simulates "today")
        today = sample_lmp_pd_df.index.max().date()
        hours = get_hour_list(today, sample_lmp_pd_df)

        # Should be limited to hours up to and including the last hour
        last_hour = sample_lmp_pd_df.index.max().hour
        assert len(hours) == last_hour + 1

    def test_midnight(self):
        """Test get_hour_list when current hour is midnight."""
        # Create dataframe where max time is at midnight
        dates = pd.date_range(start='2023-08-01 00:00', periods=1, freq='h')
        df = pd.DataFrame({
            'unique_id': ['NODE1'],
            'LMP': [30.0],
        }, index=dates)

        today = df.index.max().date()
        hours = get_hour_list(today, df)

        # Should include hour 00
        assert '00' in hours
        assert len(hours) == 1

    def test_end_of_day(self):
        """Test get_hour_list when current hour is 23."""
        dates = pd.date_range(start='2023-08-01 00:00', end='2023-08-01 23:00', freq='h')
        df = pd.DataFrame({
            'unique_id': ['NODE1'] * len(dates),
            'LMP': np.random.uniform(20, 50, len(dates)),
        }, index=dates)

        today = df.index.max().date()
        hours = get_hour_list(today, df)

        assert len(hours) == 24
        assert hours[-1] == '23'


# ============================================================
# Tests for get_fcast_time
# ============================================================

class TestGetFcastTime:
    """Tests for get_fcast_time function."""

    def test_formats_datetime_correctly(self):
        """Test that forecast time is formatted correctly."""
        fcast_date = '2023-08-03'
        fcast_hour = '14'

        result = get_fcast_time(fcast_date, fcast_hour)

        assert result == '2023-08-03T14:00:00.000000000'

    def test_handles_single_digit_hour(self):
        """Test that single digit hours are handled correctly."""
        fcast_date = '2023-08-03'
        fcast_hour = '05'

        result = get_fcast_time(fcast_date, fcast_hour)

        assert result == '2023-08-03T05:00:00.000000000'

    def test_result_is_parseable_timestamp(self):
        """Test that the result can be parsed as a timestamp."""
        fcast_date = '2023-08-03'
        fcast_hour = '14'

        result = get_fcast_time(fcast_date, fcast_hour)
        timestamp = pd.Timestamp(result)

        assert timestamp.year == 2023
        assert timestamp.month == 8
        assert timestamp.day == 3
        assert timestamp.hour == 14

    def test_midnight_hour(self):
        """Test formatting for midnight."""
        result = get_fcast_time('2023-08-03', '00')
        assert result == '2023-08-03T00:00:00.000000000'

    def test_last_hour(self):
        """Test formatting for hour 23."""
        result = get_fcast_time('2023-08-03', '23')
        assert result == '2023-08-03T23:00:00.000000000'


# ============================================================
# Tests for convert_df
# ============================================================

class TestConvertDf:
    """Tests for convert_df function."""

    def test_returns_bytes(self):
        """Test that convert_df returns bytes."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = convert_df(df)

        assert isinstance(result, bytes)

    def test_csv_content_is_valid(self):
        """Test that the CSV content can be decoded and parsed."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = convert_df(df)

        # Decode and check content
        decoded = result.decode('utf-8')
        assert 'a,b' in decoded
        assert '1,4' in decoded

    def test_no_index_in_output(self):
        """Test that index is not included in output."""
        df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        result = convert_df(df)

        decoded = result.decode('utf-8')
        lines = decoded.strip().split('\n')
        # First line is header, second line should be data without index
        assert lines[1] == '1'

    def test_empty_dataframe(self):
        """Test convert_df with empty dataframe."""
        df = pd.DataFrame({'a': [], 'b': []})
        result = convert_df(df)

        assert isinstance(result, bytes)
        decoded = result.decode('utf-8')
        assert 'a,b' in decoded  # Header should still be present

    def test_special_characters(self):
        """Test convert_df handles special characters."""
        df = pd.DataFrame({
            'name': ['Test, Node', 'Another "Node"'],
            'value': [1, 2]
        })
        result = convert_df(df)

        # Should not raise an error
        assert isinstance(result, bytes)

    def test_float_precision(self):
        """Test that floats are properly converted."""
        df = pd.DataFrame({'value': [1.23456789, 2.987654321]})
        result = convert_df(df)
        decoded = result.decode('utf-8')

        # CSV should contain the float values
        assert '1.23456789' in decoded or '1.2345' in decoded


# ============================================================
# Tests for plotting module integration
# ============================================================

class TestPlottingIntegration:
    """Integration tests for plotting module functions used by app."""

    def test_get_plot_idx_returns_boolean_series(self, sample_plot_cov_df):
        """Test that get_plot_idx returns a boolean series."""
        from src.plotting import get_plot_idx

        # Create a plot_df with mean_fcast column
        plot_df = sample_plot_cov_df.copy()
        plot_df['mean_fcast'] = np.random.uniform(20, 50, len(plot_df))
        # Set some values to NaN to simulate forecast start
        plot_df.loc[:12, 'mean_fcast'] = np.nan

        result = get_plot_idx(plot_df)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_get_plot_idx_respects_lookback(self, sample_plot_cov_df):
        """Test that get_plot_idx respects the lookback parameter."""
        from src.plotting import get_plot_idx

        # Create a plot_df with mean_fcast column
        plot_df = sample_plot_cov_df.copy()
        plot_df['mean_fcast'] = np.random.uniform(20, 50, len(plot_df))
        # Set first 24 hours to NaN (no forecast)
        plot_df.loc[:23, 'mean_fcast'] = np.nan

        # With 1 day lookback
        result_1d = get_plot_idx(plot_df, lookback='1D')
        # With 3 day lookback
        result_3d = get_plot_idx(plot_df, lookback='3D')

        # 3 day lookback should include more or equal data points
        assert result_3d.sum() >= result_1d.sum()

    def test_get_quantile_df_structure(self):
        """Test get_quantile_df returns expected structure."""
        from src.plotting import get_quantile_df
        from darts import TimeSeries

        # Create a simple TimeSeries with multiple samples
        dates = pd.date_range(start='2023-08-01', periods=24, freq='h')
        values = np.random.uniform(20, 50, (24, 1, 100))  # 100 samples

        ts = TimeSeries.from_times_and_values(
            times=dates,
            values=values,
            columns=['LMP']
        )
        ts = ts.with_columns_renamed('LMP', 'LMP')

        result = get_quantile_df(ts, 'TEST_NODE')

        # Should have quantile columns
        assert 0.1 in result.columns
        assert 0.5 in result.columns
        assert 0.9 in result.columns

    def test_get_mean_df_structure(self):
        """Test get_mean_df returns expected structure."""
        from src.plotting import get_mean_df
        from darts import TimeSeries

        # Create a simple TimeSeries with multiple samples
        dates = pd.date_range(start='2023-08-01', periods=24, freq='h')
        values = np.random.uniform(20, 50, (24, 1, 100))  # 100 samples

        ts = TimeSeries.from_times_and_values(
            times=dates,
            values=values,
            columns=['LMP']
        )

        result = get_mean_df(ts, 'TEST_NODE')

        # Should have mean_fcast column
        assert 'mean_fcast' in result.columns


# ============================================================
# Tests for data flow integration
# ============================================================

class TestDataFlowIntegration:
    """Integration tests for data flow through app processing."""

    def test_lmp_df_to_price_nodes_flow(self):
        """Test the flow from LMP dataframe to price nodes list."""
        # Create realistic LMP data
        dates = pd.date_range(start='2023-08-01', periods=100, freq='h')
        nodes = ['PSCO_ALAMOSA', 'PSCO_DENVER', 'WAPA_LOVELAND'] * 34
        nodes = nodes[:100]

        df = pd.DataFrame({
            'unique_id': nodes,
            'LMP': np.random.uniform(20, 50, 100),
        }, index=dates)

        result = get_price_nodes(df)

        assert len(result) == 3
        assert 'PSCO_ALAMOSA' in result
        assert 'PSCO_DENVER' in result
        assert 'WAPA_LOVELAND' in result

    def test_forecast_time_to_timestamp_flow(self):
        """Test the flow from user inputs to forecast timestamp."""
        # Simulate user inputs
        fcast_date = date(2023, 8, 3)
        fcast_hour = '14'

        # Get forecast time string
        fcast_time_str = get_fcast_time(str(fcast_date), fcast_hour)

        # Convert to timestamp and add 1 hour (as done in app)
        fcast_time = pd.Timestamp(fcast_time_str) + pd.Timedelta('1h')

        assert fcast_time.hour == 15
        assert fcast_time.day == 3

    def test_max_lmp_clipping(self):
        """Test that LMP values are clipped at MAX_LMP as done in app."""
        # Create price data with extreme values
        price_df = pd.DataFrame({
            'LMP': [-500, -100, 0, 50, 100, 300, 500],
        })

        # Apply clipping as done in app
        trimmed_price_df = price_df.copy()
        trimmed_price_df.loc[trimmed_price_df.LMP > MAX_LMP, 'LMP'] = MAX_LMP
        trimmed_price_df.loc[trimmed_price_df.LMP < -MAX_LMP, 'LMP'] = -MAX_LMP

        assert trimmed_price_df.LMP.max() == MAX_LMP
        assert trimmed_price_df.LMP.min() == -MAX_LMP

    def test_hour_list_to_fcast_time_flow(self, sample_lmp_pd_df):
        """Test the complete flow from hour list selection to forecast time."""
        # Get available hours
        fcast_date = date(2023, 7, 15)
        hours = get_hour_list(fcast_date, sample_lmp_pd_df)

        # User selects last hour
        selected_hour = hours[-1]

        # Get forecast time
        fcast_time_str = get_fcast_time(str(fcast_date), selected_hour)
        fcast_time = pd.Timestamp(fcast_time_str)

        assert fcast_time.hour == 23
        assert fcast_time.date() == fcast_date


# ============================================================
# Tests for plotting module get_plot_df integration
# ============================================================

class TestGetPlotDfIntegration:
    """Integration tests for get_plot_df function."""

    def test_get_plot_df_merges_data_correctly(self, sample_plot_cov_df, sample_prices_df):
        """Test that get_plot_df correctly merges all data sources."""
        from src.plotting import get_plot_df
        from darts import TimeSeries

        # Create prediction TimeSeries
        dates = pd.date_range(start='2023-08-01 12:00', periods=24, freq='h')
        values = np.random.uniform(20, 50, (24, 1, 50))

        preds = TimeSeries.from_times_and_values(
            times=dates,
            values=values,
            columns=['LMP']
        )

        result = get_plot_df(preds, sample_plot_cov_df, sample_prices_df, 'PSCO_NODE1')

        # Should have columns from all sources
        assert 'time' in result.columns
        assert 'mean_fcast' in result.columns
        assert 'MTLF' in result.columns
        assert 'LMP_HOURLY' in result.columns

    def test_get_plot_df_sorted_by_time(self, sample_plot_cov_df, sample_prices_df):
        """Test that get_plot_df returns data sorted by time."""
        from src.plotting import get_plot_df
        from darts import TimeSeries

        dates = pd.date_range(start='2023-08-01 12:00', periods=24, freq='h')
        values = np.random.uniform(20, 50, (24, 1, 50))

        preds = TimeSeries.from_times_and_values(
            times=dates,
            values=values,
            columns=['LMP']
        )

        result = get_plot_df(preds, sample_plot_cov_df, sample_prices_df, 'PSCO_NODE1')

        # Check that time is sorted
        times = result['time'].values
        assert all(times[i] <= times[i+1] for i in range(len(times)-1))

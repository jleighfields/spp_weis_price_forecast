"""
Unit tests for src/data_engineering.py

Tests cover:
- Data preparation functions (prep_lmp, prep_mtlf, prep_mtrf, prep_weather, prep_gen_cap)
- Feature engineering in prep_all_df
- Train/test split logic
- TimeSeries creation functions
"""

import pytest
import pandas as pd
import numpy as np
import polars as pl
import duckdb
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================
# Fixtures - Sample Data
# ============================================================

@pytest.fixture
def sample_lmp_data():
    """Sample LMP data as would be stored in DuckDB."""
    dates = pd.date_range(start='2023-06-01', periods=100, freq='h')
    data = {
        'Interval_HE': dates,
        'GMTIntervalEnd_HE': dates + pd.Timedelta('6h'),  # UTC offset
        'timestamp_mst_HE': dates,
        'Settlement_Location_Name': ['PSCO_NODE1'] * 50 + ['PSCO_NODE2'] * 50,
        'PNODE_Name': ['PNODE1'] * 50 + ['PNODE2'] * 50,
        'LMP': np.random.uniform(20, 50, 100).tolist(),
        'MLC': np.random.uniform(0, 2, 100).tolist(),
        'MCC': np.random.uniform(0, 1, 100).tolist(),
        'MEC': np.random.uniform(18, 48, 100).tolist(),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_mtlf_data():
    """Sample MTLF data as would be stored in DuckDB."""
    dates = pd.date_range(start='2023-06-01', periods=100, freq='h')
    data = {
        'Interval': dates,
        'GMTIntervalEnd': dates + pd.Timedelta('6h'),
        'timestamp_mst': dates,
        'MTLF': np.random.uniform(1400, 1800, 100).astype(int).tolist(),
        'Averaged_Actual': np.random.uniform(1380, 1820, 100).tolist(),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_mtrf_data():
    """Sample MTRF data as would be stored in DuckDB."""
    dates = pd.date_range(start='2023-06-01', periods=100, freq='h')
    data = {
        'Interval': dates,
        'GMTIntervalEnd': dates + pd.Timedelta('6h'),
        'timestamp_mst': dates,
        'Wind_Forecast_MW': np.random.uniform(400, 600, 100).tolist(),
        'Solar_Forecast_MW': np.random.uniform(100, 400, 100).tolist(),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_weather_data():
    """Sample weather data as would be stored in DuckDB."""
    dates = pd.date_range(start='2023-06-01', periods=100, freq='h')
    data = {
        'timestamp': dates + pd.Timedelta('6h'),  # UTC
        'timestamp_mst': dates,
        'temperature': np.random.uniform(10, 35, 100).tolist(),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_gen_cap_data():
    """Sample generation capacity data as would be stored in DuckDB."""
    dates = pd.date_range(start='2023-06-01', periods=100, freq='h')
    data = {
        'GMTIntervalEnd': dates + pd.Timedelta('6h'),
        'timestamp_mst': dates,
        'Coal_Market': np.random.uniform(80, 120, 100).tolist(),
        'Coal_Self': np.random.uniform(40, 60, 100).tolist(),
        'Hydro': np.random.uniform(25, 35, 100).tolist(),
        'Natural_Gas': np.random.uniform(180, 220, 100).tolist(),
        'Nuclear': np.random.uniform(145, 155, 100).tolist(),
        'Solar': np.random.uniform(70, 100, 100).tolist(),
        'Wind': np.random.uniform(100, 150, 100).tolist(),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_duckdb_connection(sample_lmp_data, sample_mtlf_data, sample_mtrf_data,
                           sample_weather_data, sample_gen_cap_data):
    """Create a real DuckDB connection with sample data."""
    con = duckdb.connect(':memory:')

    # Register DataFrames as tables
    con.execute("CREATE TABLE lmp AS SELECT * FROM sample_lmp_data")
    con.execute("CREATE TABLE mtlf AS SELECT * FROM sample_mtlf_data")
    con.execute("CREATE TABLE mtrf AS SELECT * FROM sample_mtrf_data")
    con.execute("CREATE TABLE weather AS SELECT * FROM sample_weather_data")
    con.execute("CREATE TABLE gen_cap AS SELECT * FROM sample_gen_cap_data")

    return con


# ============================================================
# Test Data Preparation Functions
# ============================================================

class TestPrepLmp:
    """Tests for prep_lmp function."""

    def test_returns_polars_dataframe(self, mock_duckdb_connection):
        """Test that prep_lmp returns a polars DataFrame."""
        import data_engineering as de

        result = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert isinstance(result, pl.DataFrame)

    def test_filters_by_location(self, mock_duckdb_connection):
        """Test that location filter works correctly."""
        import data_engineering as de

        result = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'), loc_filter='PSCO_')

        # All rows should contain PSCO_ in unique_id
        unique_ids = result['unique_id'].unique().to_list()
        assert all('PSCO_' in uid for uid in unique_ids)

    def test_excludes_arpa_locations(self, mock_duckdb_connection):
        """Test that ARPA locations are excluded."""
        import data_engineering as de

        # Add an ARPA location to the test data
        con = mock_duckdb_connection
        con.execute("""
            INSERT INTO lmp VALUES
            ('2023-06-01 00:00:00', '2023-06-01 06:00:00', '2023-06-01 00:00:00',
             'PSCO_ARPA_NODE', 'PNODE_ARPA', 30.0, 1.0, 0.5, 28.5)
        """)

        result = de.prep_lmp(con, start_time=pd.Timestamp('2023-01-01'))

        # No ARPA locations should be present
        unique_ids = result['unique_id'].unique().to_list()
        assert not any('_ARPA' in uid for uid in unique_ids)

    def test_filters_by_start_time(self, mock_duckdb_connection):
        """Test that start_time filter works."""
        import data_engineering as de

        start_time = pd.Timestamp('2023-06-02')
        result = de.prep_lmp(mock_duckdb_connection, start_time=start_time)

        min_time = result['timestamp_mst'].min()
        assert min_time >= start_time

    def test_filters_by_end_time(self, mock_duckdb_connection):
        """Test that end_time filter works."""
        import data_engineering as de

        end_time = pd.Timestamp('2023-06-03')
        result = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'), end_time=end_time)

        max_time = result['timestamp_mst'].max()
        assert max_time <= end_time

    def test_clips_outliers(self, mock_duckdb_connection):
        """Test that outlier clipping runs without error."""
        import data_engineering as de

        result_no_clip = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'), clip_outliers=False)
        result_with_clip = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'), clip_outliers=True)

        # Both should return valid DataFrames
        assert isinstance(result_no_clip, pl.DataFrame)
        assert isinstance(result_with_clip, pl.DataFrame)
        # Results should have same number of rows
        assert len(result_no_clip) == len(result_with_clip)

    def test_creates_lmp_diff_column(self, mock_duckdb_connection):
        """Test that lmp_diff column is created."""
        import data_engineering as de

        result = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'lmp_diff' in result.columns

    def test_drops_unnecessary_columns(self, mock_duckdb_connection):
        """Test that unnecessary columns are dropped."""
        import data_engineering as de

        result = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        # These columns should be dropped
        dropped_cols = ['Interval_HE', 'GMTIntervalEnd_HE', 'MLC', 'MCC', 'MEC']
        for col in dropped_cols:
            assert col not in result.columns

    def test_casts_lmp_to_float32(self, mock_duckdb_connection):
        """Test that LMP is cast to Float32."""
        import data_engineering as de

        result = de.prep_lmp(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert result['LMP'].dtype == pl.Float32


class TestPrepMtlf:
    """Tests for prep_mtlf function."""

    def test_returns_polars_dataframe(self, mock_duckdb_connection):
        """Test that prep_mtlf returns a polars DataFrame."""
        import data_engineering as de

        result = de.prep_mtlf(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert isinstance(result, pl.DataFrame)

    def test_filters_by_time_range(self, mock_duckdb_connection):
        """Test time range filtering."""
        import data_engineering as de

        start_time = pd.Timestamp('2023-06-02')
        end_time = pd.Timestamp('2023-06-03')
        result = de.prep_mtlf(mock_duckdb_connection, start_time=start_time, end_time=end_time)

        assert result['timestamp_mst'].min() >= start_time
        assert result['timestamp_mst'].max() <= end_time

    def test_drops_interval_columns(self, mock_duckdb_connection):
        """Test that Interval and GMTIntervalEnd columns are dropped."""
        import data_engineering as de

        result = de.prep_mtlf(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'Interval' not in result.columns
        assert 'GMTIntervalEnd' not in result.columns

    def test_casts_to_float32(self, mock_duckdb_connection):
        """Test that numeric columns are cast to Float32."""
        import data_engineering as de

        result = de.prep_mtlf(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert result['MTLF'].dtype == pl.Float32
        assert result['Averaged_Actual'].dtype == pl.Float32

    def test_sorted_by_timestamp(self, mock_duckdb_connection):
        """Test that result is sorted by timestamp_mst."""
        import data_engineering as de

        result = de.prep_mtlf(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        timestamps = result['timestamp_mst'].to_list()
        assert timestamps == sorted(timestamps)


class TestPrepMtrf:
    """Tests for prep_mtrf function."""

    def test_returns_polars_dataframe(self, mock_duckdb_connection):
        """Test that prep_mtrf returns a polars DataFrame."""
        import data_engineering as de

        result = de.prep_mtrf(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert isinstance(result, pl.DataFrame)

    def test_has_forecast_columns(self, mock_duckdb_connection):
        """Test that Wind and Solar forecast columns exist."""
        import data_engineering as de

        result = de.prep_mtrf(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'Wind_Forecast_MW' in result.columns
        assert 'Solar_Forecast_MW' in result.columns

    def test_casts_to_float32(self, mock_duckdb_connection):
        """Test that forecast columns are cast to Float32."""
        import data_engineering as de

        result = de.prep_mtrf(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert result['Wind_Forecast_MW'].dtype == pl.Float32
        assert result['Solar_Forecast_MW'].dtype == pl.Float32


class TestPrepWeather:
    """Tests for prep_weather function."""

    def test_returns_polars_dataframe(self, mock_duckdb_connection):
        """Test that prep_weather returns a polars DataFrame."""
        import data_engineering as de

        result = de.prep_weather(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert isinstance(result, pl.DataFrame)

    def test_has_temperature_column(self, mock_duckdb_connection):
        """Test that temperature column exists."""
        import data_engineering as de

        result = de.prep_weather(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'temperature' in result.columns

    def test_drops_utc_timestamp(self, mock_duckdb_connection):
        """Test that UTC timestamp column is dropped."""
        import data_engineering as de

        result = de.prep_weather(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'timestamp' not in result.columns
        assert 'timestamp_mst' in result.columns


class TestPrepGenCap:
    """Tests for prep_gen_cap function."""

    def test_returns_polars_dataframe(self, mock_duckdb_connection):
        """Test that prep_gen_cap returns a polars DataFrame."""
        import data_engineering as de

        result = de.prep_gen_cap(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert isinstance(result, pl.DataFrame)

    def test_creates_combined_coal_column(self, mock_duckdb_connection):
        """Test that Coal column is created from Coal_Market + Coal_Self."""
        import data_engineering as de

        result = de.prep_gen_cap(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'Coal' in result.columns
        # Original Coal columns should be dropped
        assert 'Coal_Market' not in result.columns
        assert 'Coal_Self' not in result.columns

    def test_has_fuel_type_columns(self, mock_duckdb_connection):
        """Test that all fuel type columns exist."""
        import data_engineering as de

        result = de.prep_gen_cap(mock_duckdb_connection, start_time=pd.Timestamp('2023-01-01'))

        expected_cols = ['Hydro', 'Natural_Gas', 'Nuclear', 'Solar', 'Wind', 'Coal']
        for col in expected_cols:
            assert col in result.columns


# ============================================================
# Test Feature Engineering (prep_all_df)
# ============================================================

class TestPrepAllDf:
    """Tests for prep_all_df function and feature engineering."""

    @pytest.fixture
    def extended_mock_connection(self):
        """Create a DuckDB connection with more data for feature engineering tests."""
        con = duckdb.connect(':memory:')

        # Create dates starting from 2023-05-20 to pass the 2023-05-15 filter
        dates = pd.date_range(start='2023-05-20', periods=200, freq='h')

        # LMP data with two locations
        lmp_data = pd.DataFrame({  # noqa: F841 - used by DuckDB SQL execution
            'Interval_HE': list(dates) * 2,
            'GMTIntervalEnd_HE': list(dates + pd.Timedelta('6h')) * 2,
            'timestamp_mst_HE': list(dates) * 2,
            'Settlement_Location_Name': ['PSCO_NODE1'] * 200 + ['PSCO_NODE2'] * 200,
            'PNODE_Name': ['PNODE1'] * 200 + ['PNODE2'] * 200,
            'LMP': np.random.uniform(20, 50, 400).tolist(),
            'MLC': np.random.uniform(0, 2, 400).tolist(),
            'MCC': np.random.uniform(0, 1, 400).tolist(),
            'MEC': np.random.uniform(18, 48, 400).tolist(),
        })

        # MTLF data
        mtlf_data = pd.DataFrame({  # noqa: F841 - used by DuckDB SQL execution
            'Interval': dates,
            'GMTIntervalEnd': dates + pd.Timedelta('6h'),
            'timestamp_mst': dates,
            'MTLF': np.random.uniform(1400, 1800, 200).astype(int).tolist(),
            'Averaged_Actual': np.random.uniform(1380, 1820, 200).tolist(),
        })

        # MTRF data
        mtrf_data = pd.DataFrame({  # noqa: F841 - used by DuckDB SQL execution
            'Interval': dates,
            'GMTIntervalEnd': dates + pd.Timedelta('6h'),
            'timestamp_mst': dates,
            'Wind_Forecast_MW': np.random.uniform(400, 600, 200).tolist(),
            'Solar_Forecast_MW': np.random.uniform(100, 400, 200).tolist(),
        })

        con.execute("CREATE TABLE lmp AS SELECT * FROM lmp_data")
        con.execute("CREATE TABLE mtlf AS SELECT * FROM mtlf_data")
        con.execute("CREATE TABLE mtrf AS SELECT * FROM mtrf_data")

        return con

    def test_returns_polars_dataframe(self, extended_mock_connection):
        """Test that prep_all_df returns a polars DataFrame."""
        import data_engineering as de

        result = de.prep_all_df(extended_mock_connection, start_time=pd.Timestamp('2023-01-01'))

        assert isinstance(result, pl.DataFrame)

    def test_creates_re_ratio_feature(self, extended_mock_connection):
        """Test that renewable energy ratio feature is created."""
        import data_engineering as de

        result = de.prep_all_df(extended_mock_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 're_ratio' in result.columns

    def test_creates_load_net_re_feature(self, extended_mock_connection):
        """Test that load net renewable energy feature is created."""
        import data_engineering as de

        result = de.prep_all_df(extended_mock_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'load_net_re' in result.columns

    def test_creates_rolling_features(self, extended_mock_connection):
        """Test that rolling window features are created."""
        import data_engineering as de

        result = de.prep_all_df(extended_mock_connection, start_time=pd.Timestamp('2023-01-01'))

        # Check for rolling features
        rolling_cols = ['lmp_diff_rolling_2', 'lmp_diff_rolling_3', 'lmp_diff_rolling_4',
                       'lmp_diff_rolling_6', 'load_net_re_diff_rolling_2']
        for col in rolling_cols:
            assert col in result.columns

    def test_filters_dates_after_may_2023(self, extended_mock_connection):
        """Test that data before 2023-05-15 is filtered out."""
        import data_engineering as de

        result = de.prep_all_df(extended_mock_connection, start_time=pd.Timestamp('2023-01-01'))

        min_date = result['timestamp_mst'].min()
        assert min_date >= pd.Timestamp('2023-05-15')

    def test_has_unique_id_column(self, extended_mock_connection):
        """Test that unique_id column exists."""
        import data_engineering as de

        result = de.prep_all_df(extended_mock_connection, start_time=pd.Timestamp('2023-01-01'))

        assert 'unique_id' in result.columns

    def test_joins_all_data_sources(self, extended_mock_connection):
        """Test that all data sources are joined."""
        import data_engineering as de

        result = de.prep_all_df(extended_mock_connection, start_time=pd.Timestamp('2023-01-01'))

        # Should have columns from all sources
        assert 'LMP' in result.columns  # from lmp
        assert 'MTLF' in result.columns  # from mtlf
        assert 'Wind_Forecast_MW' in result.columns  # from mtrf


# ============================================================
# Test Conversion and Split Functions
# ============================================================

class TestAllDfToPandas:
    """Tests for all_df_to_pandas function."""

    def test_returns_pandas_dataframe(self):
        """Test that result is a pandas DataFrame."""
        import data_engineering as de

        # Create a simple polars DataFrame with required columns
        all_df = pl.DataFrame({
            'timestamp_mst': pd.date_range('2023-06-01', periods=10, freq='h'),
            'unique_id': ['NODE1'] * 10,
            'LMP': np.random.uniform(20, 50, 10).tolist(),
            'Averaged_Actual': np.random.uniform(1400, 1600, 10).tolist(),
            'lmp_diff': np.random.uniform(-5, 5, 10).tolist(),
            'lmp_diff_rolling_2': np.random.uniform(-10, 10, 10).tolist(),
            'lmp_diff_rolling_3': np.random.uniform(-15, 15, 10).tolist(),
            'lmp_diff_rolling_4': np.random.uniform(-20, 20, 10).tolist(),
            'lmp_diff_rolling_6': np.random.uniform(-30, 30, 10).tolist(),
            'lmp_load_net_re': np.random.uniform(0.01, 0.05, 10).tolist(),
            'MTLF': np.random.uniform(1400, 1800, 10).tolist(),
            'Wind_Forecast_MW': np.random.uniform(400, 600, 10).tolist(),
            'Solar_Forecast_MW': np.random.uniform(100, 400, 10).tolist(),
            're_ratio': np.random.uniform(0.3, 0.6, 10).tolist(),
            're_diff': np.random.uniform(-0.1, 0.1, 10).tolist(),
            'load_net_re': np.random.uniform(800, 1200, 10).tolist(),
            'load_net_re_diff': np.random.uniform(-50, 50, 10).tolist(),
            'load_net_re_diff_rolling_2': np.random.uniform(-100, 100, 10).tolist(),
            'load_net_re_diff_rolling_3': np.random.uniform(-150, 150, 10).tolist(),
            'load_net_re_diff_rolling_4': np.random.uniform(-200, 200, 10).tolist(),
            'load_net_re_diff_rolling_6': np.random.uniform(-300, 300, 10).tolist(),
            'temperature': np.random.uniform(15, 30, 10).tolist(),
        })

        result = de.all_df_to_pandas(all_df)

        assert isinstance(result, pd.DataFrame)

    def test_sets_timestamp_as_index(self):
        """Test that timestamp_mst is set as index."""
        import data_engineering as de

        all_df = pl.DataFrame({
            'timestamp_mst': pd.date_range('2023-06-01', periods=10, freq='h'),
            'unique_id': ['NODE1'] * 10,
            'LMP': np.random.uniform(20, 50, 10).tolist(),
            'Averaged_Actual': np.random.uniform(1400, 1600, 10).tolist(),
            'lmp_diff': np.random.uniform(-5, 5, 10).tolist(),
            'lmp_diff_rolling_2': np.random.uniform(-10, 10, 10).tolist(),
            'lmp_diff_rolling_3': np.random.uniform(-15, 15, 10).tolist(),
            'lmp_diff_rolling_4': np.random.uniform(-20, 20, 10).tolist(),
            'lmp_diff_rolling_6': np.random.uniform(-30, 30, 10).tolist(),
            'lmp_load_net_re': np.random.uniform(0.01, 0.05, 10).tolist(),
            'MTLF': np.random.uniform(1400, 1800, 10).tolist(),
            'Wind_Forecast_MW': np.random.uniform(400, 600, 10).tolist(),
            'Solar_Forecast_MW': np.random.uniform(100, 400, 10).tolist(),
            're_ratio': np.random.uniform(0.3, 0.6, 10).tolist(),
            're_diff': np.random.uniform(-0.1, 0.1, 10).tolist(),
            'load_net_re': np.random.uniform(800, 1200, 10).tolist(),
            'load_net_re_diff': np.random.uniform(-50, 50, 10).tolist(),
            'load_net_re_diff_rolling_2': np.random.uniform(-100, 100, 10).tolist(),
            'load_net_re_diff_rolling_3': np.random.uniform(-150, 150, 10).tolist(),
            'load_net_re_diff_rolling_4': np.random.uniform(-200, 200, 10).tolist(),
            'load_net_re_diff_rolling_6': np.random.uniform(-300, 300, 10).tolist(),
            'temperature': np.random.uniform(15, 30, 10).tolist(),
        })

        result = de.all_df_to_pandas(all_df)

        assert result.index.name == 'timestamp_mst'


class TestGetTrainTestAll:
    """Tests for get_train_test_all function."""

    @pytest.fixture
    def large_mock_connection(self):
        """Create a DuckDB connection with enough data for train/test split."""
        con = duckdb.connect(':memory:')

        # Need enough data for INPUT_CHUNK_LENGTH calculations
        # INPUT_CHUNK_LENGTH = 24*7 = 168 hours
        # We need at least 2 * 168 + some buffer for train and test
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')

        lmp_data = pd.DataFrame({  # noqa: F841 - used by DuckDB SQL execution
            'Interval_HE': list(dates) * 2,
            'GMTIntervalEnd_HE': list(dates + pd.Timedelta('6h')) * 2,
            'timestamp_mst_HE': list(dates) * 2,
            'Settlement_Location_Name': ['PSCO_NODE1'] * 1000 + ['PSCO_NODE2'] * 1000,
            'PNODE_Name': ['PNODE1'] * 1000 + ['PNODE2'] * 1000,
            'LMP': np.random.uniform(20, 50, 2000).tolist(),
            'MLC': np.random.uniform(0, 2, 2000).tolist(),
            'MCC': np.random.uniform(0, 1, 2000).tolist(),
            'MEC': np.random.uniform(18, 48, 2000).tolist(),
        })

        con.execute("CREATE TABLE lmp AS SELECT * FROM lmp_data")

        return con

    def test_returns_four_dataframes(self, large_mock_connection):
        """Test that function returns four DataFrames."""
        import data_engineering as de

        lmp_all, train_all, test_all, train_test_all = de.get_train_test_all(
            large_mock_connection, start_time=pd.Timestamp('2023-01-01')
        )

        assert isinstance(lmp_all, pd.DataFrame)
        assert isinstance(train_all, pd.DataFrame)
        assert isinstance(test_all, pd.DataFrame)
        assert isinstance(train_test_all, pd.DataFrame)

    def test_train_before_test(self, large_mock_connection):
        """Test that train data comes before test data."""
        import data_engineering as de

        lmp_all, train_all, test_all, train_test_all = de.get_train_test_all(
            large_mock_connection, start_time=pd.Timestamp('2023-01-01')
        )

        if not train_all.empty and not test_all.empty:
            assert train_all.index.max() <= test_all.index.min()

    def test_train_test_all_contains_both(self, large_mock_connection):
        """Test that train_test_all contains data from both train and test."""
        import data_engineering as de

        lmp_all, train_all, test_all, train_test_all = de.get_train_test_all(
            large_mock_connection, start_time=pd.Timestamp('2023-01-01')
        )

        # train_test_all should span from train start to test end
        if not train_all.empty and not test_all.empty:
            assert train_test_all.index.min() <= train_all.index.min()
            assert train_test_all.index.max() >= test_all.index.max()


# ============================================================
# Test TimeSeries Functions
# ============================================================

class TestFillMissing:
    """Tests for fill_missing function."""

    def test_fills_missing_values(self):
        """Test that missing values are filled."""
        import data_engineering as de
        from darts import TimeSeries

        # Create a series with missing values
        dates = pd.date_range('2023-06-01', periods=10, freq='h')
        values = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
        df = pd.DataFrame({'value': values}, index=dates)

        series = [TimeSeries.from_dataframe(df, value_cols=['value'])]

        de.fill_missing(series)

        # Check no NaN values after filling
        assert not series[0].pd_dataframe().isna().any().any()


class TestGetSeries:
    """Tests for get_series function."""

    def test_returns_list_of_timeseries(self):
        """Test that function returns a list of TimeSeries."""
        import data_engineering as de
        from darts import TimeSeries

        # Create sample data with proper structure
        dates = pd.date_range('2023-06-01', periods=50, freq='h')
        lmp_all = pd.DataFrame({
            'unique_id': ['NODE1'] * 50,
            'LMP': np.random.uniform(20, 50, 50).tolist(),
        }, index=dates)

        result = de.get_series(lmp_all)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, TimeSeries) for s in result)


class TestGetFutrCov:
    """Tests for get_futr_cov function."""

    def test_returns_list_of_timeseries(self):
        """Test that function returns a list of TimeSeries."""
        import data_engineering as de
        from darts import TimeSeries

        # Create sample data with future covariate columns
        dates = pd.date_range('2023-06-01', periods=50, freq='h')
        all_df_pd = pd.DataFrame({
            'unique_id': ['NODE1'] * 50,
            'MTLF': np.random.uniform(1400, 1800, 50).tolist(),
            'Wind_Forecast_MW': np.random.uniform(400, 600, 50).tolist(),
            'Solar_Forecast_MW': np.random.uniform(100, 400, 50).tolist(),
            're_ratio': np.random.uniform(0.3, 0.6, 50).tolist(),
            're_diff': np.random.uniform(-0.1, 0.1, 50).tolist(),
            'load_net_re': np.random.uniform(800, 1200, 50).tolist(),
            'load_net_re_diff': np.random.uniform(-50, 50, 50).tolist(),
            'load_net_re_diff_rolling_2': np.random.uniform(-100, 100, 50).tolist(),
            'load_net_re_diff_rolling_3': np.random.uniform(-150, 150, 50).tolist(),
            'load_net_re_diff_rolling_4': np.random.uniform(-200, 200, 50).tolist(),
            'load_net_re_diff_rolling_6': np.random.uniform(-300, 300, 50).tolist(),
            'temperature': np.random.uniform(15, 30, 50).tolist(),
        }, index=dates)

        result = de.get_futr_cov(all_df_pd)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, TimeSeries) for s in result)


class TestGetPastCov:
    """Tests for get_past_cov function."""

    def test_returns_list_of_timeseries(self):
        """Test that function returns a list of TimeSeries."""
        import data_engineering as de
        from darts import TimeSeries

        # Create sample data with past covariate columns
        dates = pd.date_range('2023-06-01', periods=50, freq='h')
        all_df_pd = pd.DataFrame({
            'unique_id': ['NODE1'] * 50,
            'Averaged_Actual': np.random.uniform(1400, 1600, 50).tolist(),
            'lmp_diff': np.random.uniform(-5, 5, 50).tolist(),
            'lmp_diff_rolling_2': np.random.uniform(-10, 10, 50).tolist(),
            'lmp_diff_rolling_3': np.random.uniform(-15, 15, 50).tolist(),
            'lmp_diff_rolling_4': np.random.uniform(-20, 20, 50).tolist(),
            'lmp_diff_rolling_6': np.random.uniform(-30, 30, 50).tolist(),
            'lmp_load_net_re': np.random.uniform(0.01, 0.05, 50).tolist(),
        }, index=dates)

        result = de.get_past_cov(all_df_pd)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, TimeSeries) for s in result)


# ============================================================
# Test create_database (with mocked S3)
# ============================================================

class TestCreateDatabase:
    """Tests for create_database function."""

    def test_creates_duckdb_connection(self):
        """Test that function returns a DuckDB connection."""
        import data_engineering as de

        mock_s3 = MagicMock()

        # Create temporary parquet files for testing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample parquet files
            for ds in ['lmp', 'mtrf', 'mtlf', 'weather']:
                df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
                df.to_parquet(os.path.join(tmpdir, f'{ds}.parquet'))

            # Mock S3 to copy local files instead of downloading
            def mock_download(Bucket, Key, Filename):
                import shutil
                src = os.path.join(tmpdir, os.path.basename(Key))
                shutil.copy(src, Filename)

            mock_s3.download_file = mock_download

            # Mock S3 bucket contents listing to return expected parquet file keys
            mock_obj = MagicMock()
            mock_bucket_contents = []
            for ds in ['lmp', 'mtrf', 'mtlf', 'weather']:
                obj = MagicMock()
                obj.key = f'data/{ds}.parquet'
                mock_bucket_contents.append(obj)

            with patch('data_engineering.boto3.client', return_value=mock_s3):
                with patch('data_engineering.os.makedirs'):
                    with patch('data_engineering.utils.list_folder_contents_resource', return_value=mock_bucket_contents):
                        with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder'}):
                            con = de.create_database(datasets=['lmp', 'mtrf', 'mtlf', 'weather'])

            assert isinstance(con, duckdb.DuckDBPyConnection)
            con.close()

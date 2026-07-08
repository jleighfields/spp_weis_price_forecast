"""
Unit tests for src/data_collection_utils.py

The feed-agnostic collection helpers shared by the WEIS (data_collection)
and Integrated Marketplace (data_collection_im) collectors: hour-ending
columns, MST timestamps, column-name formatting, HTTP CSV reads, and the
tqdm-aware parallel runner.
"""

import os
import sys

import pytest
import pandas as pd
import polars as pl
from joblib import delayed
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@pytest.fixture
def sample_mtlf_csv():
    """Sample MTLF CSV data as returned from an SPP feed."""
    return """Interval,GMTIntervalEnd,MTLF,Averaged Actual
04/01/2023 07:00:00,04/01/2023 13:00:00,1500,1480
04/01/2023 08:00:00,04/01/2023 14:00:00,1550,1530
04/01/2023 09:00:00,04/01/2023 15:00:00,1600,1590
"""


class TestSetHE:
    """Tests for set_he (hour ending) function."""

    def test_adds_hour_ending_columns(self):
        """Test that HE columns are added with ceiling to hour."""
        import data_collection_utils as u

        df = pl.DataFrame({
            'Interval': [pd.Timestamp('2023-04-01 13:05:00'), pd.Timestamp('2023-04-01 13:55:00')],
            'GMTIntervalEnd': [pd.Timestamp('2023-04-01 13:05:00'), pd.Timestamp('2023-04-01 13:55:00')],
            'timestamp_mst': [pd.Timestamp('2023-04-01 06:05:00'), pd.Timestamp('2023-04-01 06:55:00')],
        })

        result = u.set_he(df)

        assert 'Interval_HE' in result.columns
        assert 'GMTIntervalEnd_HE' in result.columns
        assert 'timestamp_mst_HE' in result.columns

        # Check ceiling is applied correctly (both should ceil to 14:00)
        assert result['Interval_HE'][0] == pd.Timestamp('2023-04-01 14:00:00')
        assert result['Interval_HE'][1] == pd.Timestamp('2023-04-01 14:00:00')


class TestAddTimestampMst:
    """Tests for add_timestamp_mst function."""

    def test_adds_mst_timestamp(self):
        """Test that MST timestamp column is added correctly."""
        import data_collection_utils as u

        df = pl.DataFrame({
            'GMTIntervalEnd': [pd.Timestamp('2023-04-01 13:00:00'), pd.Timestamp('2023-04-01 14:00:00')]
        })

        result = u.add_timestamp_mst(df)

        assert 'timestamp_mst' in result.columns
        # UTC to MST is -7 hours
        assert result['timestamp_mst'][0] == pd.Timestamp('2023-04-01 06:00:00')


class TestFormatDfColnames:
    """Tests for format_df_colnames function."""

    def test_removes_spaces(self):
        """Test that spaces are replaced with underscores."""
        import data_collection_utils as u

        df = pl.DataFrame({'Column Name': [1], 'Another Column': [2]})
        u.format_df_colnames(df)

        assert 'Column_Name' in df.columns
        assert 'Another_Column' in df.columns

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        import data_collection_utils as u

        df = pl.DataFrame({' Column ': [1], '  Name  ': [2]})
        u.format_df_colnames(df)

        assert 'Column' in df.columns
        assert 'Name' in df.columns


class TestGetCsvFromUrl:
    """Tests for get_csv_from_url function."""

    def test_successful_fetch(self, sample_mtlf_csv):
        """Test successful CSV fetch from URL."""
        import data_collection_utils as u

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtlf_csv

        with patch('data_collection_utils.requests.get', return_value=mock_response):
            df = u.get_csv_from_url('http://test.url')

        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] > 0
        assert len(df) == 3
        assert 'Interval' in df.columns

    def test_failed_fetch_returns_empty_df(self):
        """Test that failed fetch returns empty DataFrame."""
        import data_collection_utils as u

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.reason = 'Not Found'

        with patch('data_collection_utils.requests.get', return_value=mock_response):
            df = u.get_csv_from_url('http://test.url')

        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()

    def test_exception_returns_empty_df(self):
        """Test that exception returns empty DataFrame."""
        import data_collection_utils as u

        with patch('data_collection_utils.requests.get', side_effect=Exception('Connection error')):
            df = u.get_csv_from_url('http://test.url')

        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()


class TestProgressParallel:
    """Tests for ProgressParallel class."""

    def test_parallel_execution(self):
        """Test that ProgressParallel executes jobs."""
        import data_collection_utils as u

        def simple_func(x):
            return x * 2

        parallel = u.ProgressParallel(n_jobs=2, total=3, use_tqdm=False)
        results = parallel(delayed(simple_func)(i) for i in [1, 2, 3])

        assert sorted(results) == [2, 4, 6]

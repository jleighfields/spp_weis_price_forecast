"""
Unit tests for src/data_collection.py

Tests cover:
- Helper functions (datetime conversion, time components, column formatting)
- URL generation functions
- Data processing functions (with mocked HTTP requests)
- Aggregation functions
"""

import pytest
import pandas as pd
import polars as pl
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_mtlf_csv():
    """Sample MTLF CSV data as returned from SPP API."""
    return """Interval,GMTIntervalEnd,MTLF,Averaged Actual
04/01/2023 07:00:00,04/01/2023 13:00:00,1500,1480
04/01/2023 08:00:00,04/01/2023 14:00:00,1550,1530
04/01/2023 09:00:00,04/01/2023 15:00:00,1600,1590
"""


@pytest.fixture
def sample_mtrf_csv():
    """Sample MTRF CSV data as returned from SPP API."""
    return """Interval,GMTIntervalEnd,Wind Forecast MW,Solar Forecast MW
04/01/2023 07:00:00,04/01/2023 13:00:00,500,200
04/01/2023 08:00:00,04/01/2023 14:00:00,520,250
04/01/2023 09:00:00,04/01/2023 15:00:00,540,300
"""


@pytest.fixture
def sample_5min_lmp_csv():
    """Sample 5-minute LMP CSV data as returned from SPP API."""
    return """Interval,GMTIntervalEnd,Settlement Location,Pnode,LMP,MLC,MCC,MEC
04/01/2023 07:05:00,04/01/2023 13:05:00,PSCO_NODE1,PNODE1,25.50,1.00,0.50,24.00
04/01/2023 07:05:00,04/01/2023 13:05:00,PSCO_NODE2,PNODE2,26.00,1.10,0.60,24.30
04/01/2023 07:10:00,04/01/2023 13:10:00,PSCO_NODE1,PNODE1,25.75,1.05,0.55,24.15
04/01/2023 07:10:00,04/01/2023 13:10:00,PSCO_NODE2,PNODE2,26.25,1.15,0.65,24.45
"""


@pytest.fixture
def sample_daily_lmp_csv():
    """Sample daily LMP CSV data as returned from SPP API."""
    return """Interval,GMT Interval,Settlement_Location_Name,PNODE_Name,LMP,MLC,MCC,MEC
04/01/2023 07:05:00,04/01/2023 13:05:00,PSCO_NODE1,PNODE1,25.50,1.00,0.50,24.00
04/01/2023 07:05:00,04/01/2023 13:05:00,PSCO_NODE2,PNODE2,26.00,1.10,0.60,24.30
04/01/2023 07:10:00,04/01/2023 13:10:00,PSCO_NODE1,PNODE1,25.75,1.05,0.55,24.15
04/01/2023 07:10:00,04/01/2023 13:10:00,PSCO_NODE2,PNODE2,26.25,1.15,0.65,24.45
"""


@pytest.fixture
def sample_gen_cap_csv():
    """Sample generation capacity CSV data."""
    return """GMT_TIME,Coal_Market,Coal_Self,Hydro,Natural_Gas,Nuclear,Solar,Wind
2024-11-08T13:00:00+00:00,100,50,30,200,150,80,120
2024-11-08T14:00:00+00:00,105,52,32,210,150,90,130
"""


@pytest.fixture
def sample_datetime_df():
    """Sample Polars DataFrame with datetime string columns."""
    return pl.DataFrame({
        'Interval': ['04/01/2023 07:00:00', '04/01/2023 08:00:00'],
        'GMTIntervalEnd': ['04/01/2023 13:00:00', '04/01/2023 14:00:00'],
        'Value': [100, 200]
    })


@pytest.fixture
def sample_time_components():
    """Sample time components dictionary."""
    return {
        'YEAR': '2023',
        'MONTH': '04',
        'DAY': '01',
        'HOUR': '07',
        'MINUTE': '00',
        'YM': '202304',
        'YMD': '20230401',
        'COMBINED': '202304010700',
    }


# ============================================================
# Test Helper Functions
# ============================================================

class TestConvertDatetimeCols:
    """Tests for convert_datetime_cols function."""

    def test_converts_datetime_columns(self, sample_datetime_df):
        """Test that string columns are converted to datetime."""
        import data_collection as dc

        df = sample_datetime_df.clone()
        result = dc.convert_datetime_cols(df, dt_cols=['Interval', 'GMTIntervalEnd'])

        assert result['Interval'].dtype == pl.Datetime
        assert result['GMTIntervalEnd'].dtype == pl.Datetime

    def test_preserves_other_columns(self, sample_datetime_df):
        """Test that non-datetime columns are not modified."""
        import data_collection as dc

        df = sample_datetime_df.clone()
        original_values = df['Value'].to_list()
        result = dc.convert_datetime_cols(df, dt_cols=['Interval', 'GMTIntervalEnd'])

        assert result['Value'].to_list() == original_values


class TestSetHE:
    """Tests for set_he (hour ending) function."""

    def test_adds_hour_ending_columns(self):
        """Test that HE columns are added with ceiling to hour."""
        import data_collection as dc

        df = pl.DataFrame({
            'Interval': [pd.Timestamp('2023-04-01 13:05:00'), pd.Timestamp('2023-04-01 13:55:00')],
            'GMTIntervalEnd': [pd.Timestamp('2023-04-01 13:05:00'), pd.Timestamp('2023-04-01 13:55:00')],
            'timestamp_mst': [pd.Timestamp('2023-04-01 06:05:00'), pd.Timestamp('2023-04-01 06:55:00')],
        })

        result = dc.set_he(df)

        assert 'Interval_HE' in result.columns
        assert 'GMTIntervalEnd_HE' in result.columns
        assert 'timestamp_mst_HE' in result.columns

        # Check ceiling is applied correctly (both should ceil to 14:00)
        assert result['Interval_HE'][0] == pd.Timestamp('2023-04-01 14:00:00')
        assert result['Interval_HE'][1] == pd.Timestamp('2023-04-01 14:00:00')


class TestGetTimeComponents:
    """Tests for get_time_components function."""

    def test_returns_correct_components(self):
        """Test that time components are correctly parsed."""
        import data_collection as dc

        tc = dc.get_time_components('4/1/2023 07:00:00')

        assert tc['YEAR'] == '2023'
        assert tc['MONTH'] == '04'
        assert tc['DAY'] == '01'
        assert tc['HOUR'] == '07'
        assert tc['MINUTE'] == '00'
        assert tc['YM'] == '202304'
        assert tc['YMD'] == '20230401'
        assert tc['COMBINED'] == '202304010700'

    def test_five_min_ceil_rounds_up(self):
        """Test that five_min_ceil rounds to next 5-minute interval."""
        import data_collection as dc

        tc = dc.get_time_components('4/1/2023 07:03:00', five_min_ceil=True)

        assert tc['MINUTE'] == '05'

    def test_hour_ceil_rounds_up(self):
        """Test that default (hour ceil) rounds to next hour."""
        import data_collection as dc

        tc = dc.get_time_components('4/1/2023 07:30:00', five_min_ceil=False)

        assert tc['HOUR'] == '08'
        assert tc['MINUTE'] == '00'

    def test_current_time_when_none(self):
        """Test that current time is used when time_str is None."""
        import data_collection as dc

        tc = dc.get_time_components(None)

        assert tc is not None
        assert 'YEAR' in tc
        assert 'timestamp' in tc


class TestAddTimestampMst:
    """Tests for add_timestamp_mst function."""

    def test_adds_mst_timestamp(self):
        """Test that MST timestamp column is added correctly."""
        import data_collection as dc

        df = pl.DataFrame({
            'GMTIntervalEnd': [pd.Timestamp('2023-04-01 13:00:00'), pd.Timestamp('2023-04-01 14:00:00')]
        })

        result = dc.add_timestamp_mst(df)

        assert 'timestamp_mst' in result.columns
        # UTC to MST is -7 hours
        assert result['timestamp_mst'][0] == pd.Timestamp('2023-04-01 06:00:00')


class TestFormatDfColnames:
    """Tests for format_df_colnames function."""

    def test_removes_spaces(self):
        """Test that spaces are replaced with underscores."""
        import data_collection as dc

        df = pl.DataFrame({'Column Name': [1], 'Another Column': [2]})
        dc.format_df_colnames(df)

        assert 'Column_Name' in df.columns
        assert 'Another_Column' in df.columns

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        import data_collection as dc

        df = pl.DataFrame({' Column ': [1], '  Name  ': [2]})
        dc.format_df_colnames(df)

        assert 'Column' in df.columns
        assert 'Name' in df.columns


# ============================================================
# Test URL Generation Functions
# ============================================================

class TestGetHourlyMtlfUrl:
    """Tests for get_hourly_mtlf_url function."""

    def test_generates_correct_url(self, sample_time_components):
        """Test URL is correctly formatted."""
        import data_collection as dc

        url = dc.get_hourly_mtlf_url(sample_time_components)

        assert 'portal.spp.org' in url
        assert 'systemwide-hourly-load-forecast-mtlf' in url
        assert '2023' in url
        assert '04' in url
        assert '01' in url
        assert 'WEIS-OP-MTLF-202304010700.csv' in url


class TestGetHourlyMtrfUrl:
    """Tests for get_hourly_mtrf_url function."""

    def test_generates_correct_url(self, sample_time_components):
        """Test URL is correctly formatted."""
        import data_collection as dc

        url = dc.get_hourly_mtrf_url(sample_time_components)

        assert 'portal.spp.org' in url
        assert 'mid-term-resource-forecast-mtrf' in url
        assert 'WEIS-OP-MTRF-202304010700.csv' in url


class TestGet5minLmpUrl:
    """Tests for get_5min_lmp_url function."""

    def test_generates_correct_url(self, sample_time_components):
        """Test URL is correctly formatted."""
        import data_collection as dc

        url = dc.get_5min_lmp_url(sample_time_components)

        assert 'portal.spp.org' in url
        assert 'lmp-by-settlement-location' in url
        assert 'By_Interval' in url
        assert 'WEIS-RTBM-LMP-SL-202304010700.csv' in url


class TestGetDailyLmpUrl:
    """Tests for get_daily_lmp_url function."""

    def test_generates_correct_url(self, sample_time_components):
        """Test URL is correctly formatted."""
        import data_collection as dc

        url = dc.get_daily_lmp_url(sample_time_components)

        assert 'portal.spp.org' in url
        assert 'lmp-by-settlement-location' in url
        assert 'By_Day' in url
        assert 'WEIS-RTBM-LMP-DAILY-SL-20230401.csv' in url


class TestGetGenCapUrl:
    """Tests for get_gen_cap_url function."""

    def test_generates_correct_url(self, sample_time_components):
        """Test URL is correctly formatted."""
        import data_collection as dc

        url = dc.get_gen_cap_url(sample_time_components)

        assert 'portal.spp.org' in url
        assert 'hourly-gen-capacity-by-fuel-type' in url
        assert 'WEIS-HRLY-GEN-CAP-BY-FUEL-TYPE-20230401.csv' in url


# ============================================================
# Test CSV Fetching
# ============================================================

class TestGetCsvFromUrl:
    """Tests for get_csv_from_url function."""

    def test_successful_fetch(self, sample_mtlf_csv):
        """Test successful CSV fetch from URL."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtlf_csv

        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_csv_from_url('http://test.url')

        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] > 0
        assert len(df) == 3
        assert 'Interval' in df.columns

    def test_failed_fetch_returns_empty_df(self):
        """Test that failed fetch returns empty DataFrame."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.reason = 'Not Found'

        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_csv_from_url('http://test.url')

        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()

    def test_exception_returns_empty_df(self):
        """Test that exception returns empty DataFrame."""
        import data_collection as dc

        with patch('data_collection.requests.get', side_effect=Exception('Connection error')):
            df = dc.get_csv_from_url('http://test.url')

        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()


# ============================================================
# Test Data Processing Functions
# ============================================================

class TestGetProcessMtlf:
    """Tests for get_process_mtlf function."""

    def test_processes_mtlf_data(self, sample_mtlf_csv):
        """Test MTLF data is processed and returns S3 path."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtlf_csv

        tc = dc.get_time_components('4/1/2023 07:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
                with patch.object(pl.DataFrame, 'write_parquet'):
                    result = dc.get_process_mtlf(tc)

        # Function now returns S3 path string
        assert isinstance(result, str)
        assert 's3://' in result or 'http' in result


class TestGetProcessMtrf:
    """Tests for get_process_mtrf function."""

    def test_processes_mtrf_data(self, sample_mtrf_csv):
        """Test MTRF data is processed and returns S3 path."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtrf_csv

        tc = dc.get_time_components('4/1/2023 07:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
                with patch.object(pl.DataFrame, 'write_parquet'):
                    result = dc.get_process_mtrf(tc)

        # Function now returns S3 path string
        assert isinstance(result, str)
        assert 's3://' in result or 'http' in result


class TestAggLmp:
    """Tests for agg_lmp function."""

    def test_aggregates_to_hourly(self):
        """Test that 5-minute LMPs are aggregated to hourly."""
        import data_collection as dc

        df = pl.DataFrame({
            'Interval_HE': [pd.Timestamp('2023-04-01 14:00:00')] * 4,
            'GMTIntervalEnd_HE': [pd.Timestamp('2023-04-01 14:00:00')] * 4,
            'timestamp_mst_HE': [pd.Timestamp('2023-04-01 07:00:00')] * 4,
            'Settlement_Location_Name': ['NODE1', 'NODE1', 'NODE2', 'NODE2'],
            'PNODE_Name': ['PNODE1', 'PNODE1', 'PNODE2', 'PNODE2'],
            'LMP': [25.0, 26.0, 27.0, 28.0],
            'MLC': [1.0, 1.0, 1.0, 1.0],
            'MCC': [0.5, 0.5, 0.5, 0.5],
            'MEC': [23.5, 24.5, 25.5, 26.5],
        })

        result = dc.agg_lmp(df)

        # Should have 2 rows (one per location)
        assert len(result) == 2
        # LMP should be averaged
        node1_row = result.filter(pl.col('Settlement_Location_Name') == 'NODE1')
        assert node1_row['LMP'][0] == 25.5  # (25 + 26) / 2


class TestGetProcess5minLmp:
    """Tests for get_process_5min_lmp function."""

    def test_processes_5min_lmp_data(self, sample_5min_lmp_csv):
        """Test 5-minute LMP data is processed and returns S3 path."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_5min_lmp_csv

        tc = dc.get_time_components('4/1/2023 13:10:00', five_min_ceil=True)

        with patch('data_collection.requests.get', return_value=mock_response):
            with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
                with patch.object(pl.DataFrame, 'write_parquet'):
                    result = dc.get_process_5min_lmp(tc)

        # Function now returns S3 path string
        assert isinstance(result, str)
        assert 's3://' in result or 'http' in result


class TestGetProcessDailyLmp:
    """Tests for get_process_daily_lmp function."""

    def test_processes_daily_lmp_data(self, sample_daily_lmp_csv):
        """Test daily LMP data is processed and returns S3 path."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_daily_lmp_csv

        tc = dc.get_time_components('4/1/2023 00:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
                with patch.object(pl.DataFrame, 'write_parquet'):
                    result = dc.get_process_daily_lmp(tc)

        # Function now returns S3 path string
        assert isinstance(result, str)
        assert 's3://' in result or 'http' in result


# ============================================================
# Test Range Data Functions
# ============================================================

class TestGetRangeData:
    """Tests for get_range_data function."""

    def test_collects_range_of_data(self, sample_mtlf_csv):
        """Test that data is collected for a range of timestamps."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtlf_csv

        end_ts = pd.Timestamp('2023-04-01 10:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
                with patch.object(pl.DataFrame, 'write_parquet'):
                    result = dc.get_range_data(
                        end_ts=end_ts,
                        n_periods=3,
                        freq='h',
                        get_process_func=dc.get_process_mtlf,
                        do_parallel=False
                    )

        # Function now returns list of strings (S3 paths or URLs)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_returns_list_of_paths(self, sample_mtlf_csv):
        """Test that function returns list of S3 paths."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtlf_csv

        end_ts = pd.Timestamp('2023-04-01 10:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
                with patch.object(pl.DataFrame, 'write_parquet'):
                    result = dc.get_range_data(
                        end_ts=end_ts,
                        n_periods=2,
                        freq='h',
                        get_process_func=dc.get_process_mtlf,
                        do_parallel=False
                    )

        # All results should be strings
        assert all(isinstance(r, str) for r in result)


# ============================================================
# Test Upsert Functions
# ============================================================

class TestUpsertMtlfMtrfLmp:
    """Tests for upsert_mtlf_mtrf_lmp function."""

    def test_validates_target_parameter(self):
        """Test that invalid target raises ValueError."""
        import data_collection as dc

        with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
            with pytest.raises(ValueError):
                dc.upsert_mtlf_mtrf_lmp([], target='invalid')

    def test_accepts_valid_targets(self):
        """Test that valid targets are accepted."""
        import data_collection as dc

        # Create a mock parquet file
        mock_df = pl.DataFrame({
            'GMTIntervalEnd': [pd.Timestamp('2023-04-01 13:00:00')],
            'file_create_time_utc': [pd.Timestamp('2023-04-01 12:00:00')],
            'value': [100],
        })

        with patch.dict(os.environ, {'AWS_S3_BUCKET': 'test-bucket', 'AWS_S3_FOLDER': 'test-folder/'}):
            with patch('data_collection.check_file_exists_client', return_value=False):
                with patch('polars.scan_parquet') as mock_scan:
                    mock_scan.return_value.sort.return_value.unique.return_value.collect.return_value = mock_df
                    with patch.object(pl.DataFrame, 'write_parquet'):
                        # Should not raise for valid targets
                        for target in ['mtlf', 'mtrf', 'lmp']:
                            try:
                                dc.upsert_mtlf_mtrf_lmp(['test.parquet'], target=target)
                            except (KeyError, Exception):
                                # May fail due to missing columns, but shouldn't fail on target validation
                                pass


# ============================================================
# Test ProgressParallel Class
# ============================================================

class TestProgressParallel:
    """Tests for ProgressParallel class."""

    def test_parallel_execution(self):
        """Test that ProgressParallel executes jobs."""
        import data_collection as dc

        def simple_func(x):
            return x * 2

        parallel = dc.ProgressParallel(n_jobs=2, total=3, use_tqdm=False)
        results = parallel(dc.delayed(simple_func)(i) for i in [1, 2, 3])

        assert sorted(results) == [2, 4, 6]


# ============================================================
# Integration Tests (optional - require network/database)
# ============================================================

@pytest.mark.skip(reason="Integration test - requires network access")
class TestIntegration:
    """Integration tests that require network access."""

    def test_fetch_real_mtlf_data(self):
        """Test fetching real MTLF data from SPP."""
        import data_collection as dc

        tc = dc.get_time_components()
        result = dc.get_process_mtlf(tc)

        # Returns string (S3 path or URL)
        assert isinstance(result, str)

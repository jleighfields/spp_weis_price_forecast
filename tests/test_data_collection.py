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
    return """Interval,GMT Interval,Settlement Location,Pnode,LMP,MLC,MCC,MEC
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
    """Sample DataFrame with datetime string columns."""
    return pd.DataFrame({
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
        # Import here to avoid module-level import issues
        import data_collection as dc

        df = sample_datetime_df.copy()
        dc.convert_datetime_cols(df, dt_cols=['Interval', 'GMTIntervalEnd'])

        assert pd.api.types.is_datetime64_any_dtype(df['Interval'])
        assert pd.api.types.is_datetime64_any_dtype(df['GMTIntervalEnd'])

    def test_preserves_other_columns(self, sample_datetime_df):
        """Test that non-datetime columns are not modified."""
        import data_collection as dc

        df = sample_datetime_df.copy()
        original_values = df['Value'].tolist()
        dc.convert_datetime_cols(df, dt_cols=['Interval', 'GMTIntervalEnd'])

        assert df['Value'].tolist() == original_values


class TestSetHE:
    """Tests for set_he (hour ending) function."""

    def test_adds_hour_ending_columns(self):
        """Test that HE columns are added with ceiling to hour."""
        import data_collection as dc

        df = pd.DataFrame({
            'Interval': pd.to_datetime(['2023-04-01 13:05:00', '2023-04-01 13:55:00']),
            'GMTIntervalEnd': pd.to_datetime(['2023-04-01 13:05:00', '2023-04-01 13:55:00']),
            'timestamp_mst': pd.to_datetime(['2023-04-01 06:05:00', '2023-04-01 06:55:00']),
        })

        dc.set_he(df)

        assert 'Interval_HE' in df.columns
        assert 'GMTIntervalEnd_HE' in df.columns
        assert 'timestamp_mst_HE' in df.columns

        # Check ceiling is applied correctly
        assert df['Interval_HE'].iloc[0] == pd.Timestamp('2023-04-01 14:00:00')
        assert df['Interval_HE'].iloc[1] == pd.Timestamp('2023-04-01 14:00:00')


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

        df = pd.DataFrame({
            'GMTIntervalEnd': pd.to_datetime(['2023-04-01 13:00:00', '2023-04-01 14:00:00'])
        })

        dc.add_timestamp_mst(df)

        assert 'timestamp_mst' in df.columns
        # UTC to MST is -7 hours
        assert df['timestamp_mst'].iloc[0] == pd.Timestamp('2023-04-01 06:00:00')


class TestFormatDfColnames:
    """Tests for format_df_colnames function."""

    def test_removes_spaces(self):
        """Test that spaces are replaced with underscores."""
        import data_collection as dc

        df = pd.DataFrame({'Column Name': [1], 'Another Column': [2]})
        dc.format_df_colnames(df)

        assert 'Column_Name' in df.columns
        assert 'Another_Column' in df.columns

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        import data_collection as dc

        df = pd.DataFrame({' Column ': [1], '  Name  ': [2]})
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

        assert not df.empty
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

        assert df.empty

    def test_exception_returns_empty_df(self):
        """Test that exception returns empty DataFrame."""
        import data_collection as dc

        with patch('data_collection.requests.get', side_effect=Exception('Connection error')):
            df = dc.get_csv_from_url('http://test.url')

        assert df.empty


# ============================================================
# Test Data Processing Functions
# ============================================================

class TestGetProcessMtlf:
    """Tests for get_process_mtlf function."""

    def test_processes_mtlf_data(self, sample_mtlf_csv):
        """Test MTLF data is processed correctly."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtlf_csv

        tc = dc.get_time_components('4/1/2023 07:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_process_mtlf(tc)

        assert not df.empty
        assert 'timestamp_mst' in df.columns
        assert 'MTLF' in df.columns
        assert 'Averaged_Actual' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['GMTIntervalEnd'])


class TestGetProcessMtrf:
    """Tests for get_process_mtrf function."""

    def test_processes_mtrf_data(self, sample_mtrf_csv):
        """Test MTRF data is processed correctly."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtrf_csv

        tc = dc.get_time_components('4/1/2023 07:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_process_mtrf(tc)

        assert not df.empty
        assert 'timestamp_mst' in df.columns
        assert 'Wind_Forecast_MW' in df.columns
        assert 'Solar_Forecast_MW' in df.columns


class TestAggLmp:
    """Tests for agg_lmp function."""

    def test_aggregates_to_hourly(self):
        """Test that 5-minute LMPs are aggregated to hourly."""
        import data_collection as dc

        df = pd.DataFrame({
            'Interval_HE': pd.to_datetime(['2023-04-01 14:00:00'] * 4),
            'GMTIntervalEnd_HE': pd.to_datetime(['2023-04-01 14:00:00'] * 4),
            'timestamp_mst_HE': pd.to_datetime(['2023-04-01 07:00:00'] * 4),
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
        node1_lmp = result[result['Settlement_Location_Name'] == 'NODE1']['LMP'].iloc[0]
        assert node1_lmp == 25.5  # (25 + 26) / 2


class TestGetProcess5minLmp:
    """Tests for get_process_5min_lmp function."""

    def test_processes_5min_lmp_data(self, sample_5min_lmp_csv):
        """Test 5-minute LMP data is processed and aggregated."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_5min_lmp_csv

        tc = dc.get_time_components('4/1/2023 13:10:00', five_min_ceil=True)

        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_process_5min_lmp(tc)

        assert not df.empty
        assert 'LMP' in df.columns
        assert 'Settlement_Location_Name' in df.columns
        # Should have HE columns after aggregation
        assert 'GMTIntervalEnd_HE' in df.columns


class TestGetProcessDailyLmp:
    """Tests for get_process_daily_lmp function."""

    def test_processes_daily_lmp_data(self, sample_daily_lmp_csv):
        """Test daily LMP data is processed."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_daily_lmp_csv

        tc = dc.get_time_components('4/1/2023 00:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_process_daily_lmp(tc)

        assert not df.empty
        assert 'LMP' in df.columns


class TestGetProcessGenCap:
    """Tests for get_process_gen_cap function."""

    def test_processes_gen_cap_data(self, sample_gen_cap_csv):
        """Test generation capacity data is processed."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_gen_cap_csv

        tc = dc.get_time_components('11/8/2024 00:00:00')

        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_process_gen_cap(tc)

        assert not df.empty
        assert 'Coal_Market' in df.columns
        assert 'Wind' in df.columns
        assert 'timestamp_mst' in df.columns


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

        # Use do_parallel=False for testing to avoid multiprocessing issues with mocks
        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_range_data(
                end_ts=end_ts,
                n_periods=3,
                freq='h',
                get_process_func=dc.get_process_mtlf,
                dup_cols=['GMTIntervalEnd'],
                do_parallel=False
            )

        assert not df.empty

    def test_deduplicates_data(self, sample_mtlf_csv):
        """Test that duplicate rows are removed."""
        import data_collection as dc

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = sample_mtlf_csv

        end_ts = pd.Timestamp('2023-04-01 10:00:00')

        # Use do_parallel=False for testing to avoid multiprocessing issues with mocks
        with patch('data_collection.requests.get', return_value=mock_response):
            df = dc.get_range_data(
                end_ts=end_ts,
                n_periods=3,
                freq='h',
                get_process_func=dc.get_process_mtlf,
                dup_cols=['GMTIntervalEnd'],
                do_parallel=False
            )

        # Check no duplicates on primary key
        assert not df['GMTIntervalEnd'].duplicated().any()


# ============================================================
# Test Upsert Functions (with mocked database)
# ============================================================

class TestUpsertMtlf:
    """Tests for upsert_mtlf function."""

    def test_upsert_creates_table_if_not_exists(self):
        """Test that table is created if it doesn't exist."""
        import data_collection as dc

        df = pd.DataFrame({
            'Interval': pd.to_datetime(['2023-04-01 07:00:00']),
            'GMTIntervalEnd': pd.to_datetime(['2023-04-01 13:00:00']),
            'timestamp_mst': pd.to_datetime(['2023-04-01 06:00:00']),
            'MTLF': [1500],
            'Averaged_Actual': [1480.0],
        })

        mock_con = MagicMock()
        mock_con.__enter__ = MagicMock(return_value=mock_con)
        mock_con.__exit__ = MagicMock(return_value=False)
        mock_con.sql.return_value.fetchall.return_value = [[0]]

        with patch('data_collection.duckdb.connect', return_value=mock_con):
            dc.upsert_mtlf(df)

        # Verify CREATE TABLE was called
        calls = [str(call) for call in mock_con.sql.call_args_list]
        assert any('CREATE TABLE IF NOT EXISTS mtlf' in str(call) for call in calls)

    def test_backfill_removes_nulls(self):
        """Test that backfill=True removes rows with null values."""
        import data_collection as dc

        df = pd.DataFrame({
            'Interval': pd.to_datetime(['2023-04-01 07:00:00', '2023-04-01 08:00:00']),
            'GMTIntervalEnd': pd.to_datetime(['2023-04-01 13:00:00', '2023-04-01 14:00:00']),
            'timestamp_mst': pd.to_datetime(['2023-04-01 06:00:00', '2023-04-01 07:00:00']),
            'MTLF': [1500, 1550],
            'Averaged_Actual': [1480.0, None],  # Second row has null
        })

        mock_con = MagicMock()
        mock_con.__enter__ = MagicMock(return_value=mock_con)
        mock_con.__exit__ = MagicMock(return_value=False)
        mock_con.sql.return_value.fetchall.return_value = [[0]]

        with patch('data_collection.duckdb.connect', return_value=mock_con):
            # Should not raise error - null row will be dropped
            dc.upsert_mtlf(df.copy(), backfill=True)


class TestUpsertLmp:
    """Tests for upsert_lmp function."""

    def test_upsert_removes_duplicates(self):
        """Test that duplicate primary keys are removed before upsert."""
        import data_collection as dc

        df = pd.DataFrame({
            'Interval_HE': pd.to_datetime(['2023-04-01 14:00:00'] * 2),
            'GMTIntervalEnd_HE': pd.to_datetime(['2023-04-01 14:00:00'] * 2),
            'timestamp_mst_HE': pd.to_datetime(['2023-04-01 07:00:00'] * 2),
            'Settlement_Location_Name': ['NODE1', 'NODE1'],  # Duplicate
            'PNODE_Name': ['PNODE1', 'PNODE1'],  # Duplicate
            'LMP': [25.0, 26.0],
            'MLC': [1.0, 1.0],
            'MCC': [0.5, 0.5],
            'MEC': [23.5, 24.5],
        })

        mock_con = MagicMock()
        mock_con.__enter__ = MagicMock(return_value=mock_con)
        mock_con.__exit__ = MagicMock(return_value=False)
        mock_con.sql.return_value.fetchall.return_value = [[0]]

        with patch('data_collection.duckdb.connect', return_value=mock_con):
            # Should not raise - duplicates will be removed
            dc.upsert_lmp(df)


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
        df = dc.get_process_mtlf(tc)

        # May be empty if no data for current hour
        assert isinstance(df, pd.DataFrame)

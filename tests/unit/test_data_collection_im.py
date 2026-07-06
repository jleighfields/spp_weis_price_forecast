"""
Unit tests for src/data_collection_im.py (Integrated Marketplace collectors).

Fixtures under tests/unit/fixtures/ are trimmed *real* portal files —
one pre-launch (no BAA column, East-only) and one post-launch (BAA
column, both BAAs, blank-BAA leading rows where the live feed has them)
per feed. LMP fixtures include the out-of-list node AEC to exercise the
hub/BA storage filter.
"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import data_collection_im as dcim
from node_list import STORED_NODES, WEST_HUB_BA_NODES, EAST_HUB_NODES

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def fixture_df(name: str) -> pl.DataFrame:
    """Read a trimmed portal CSV fixture as the raw downloaded DataFrame."""
    return pl.read_csv(os.path.join(FIXTURES, name))


@pytest.fixture
def out_dir(tmp_path):
    """Local base_path with the per-feed subdirs the processors write into."""
    for sub in ['mtlf', 'mtrf', 'lmp_5min', 'lmp_daily', 'rf_reserve_zone', 'da_lmp']:
        (tmp_path / sub).mkdir()
    return f'{tmp_path}/'


def tc_for(time_str: str, five_min: bool = False, dst_variant: bool = False) -> dict:
    """Shorthand for get_time_components_im."""
    return dcim.get_time_components_im(time_str, five_min_ceil=five_min, dst_variant=dst_variant)


def run_processor(process_func, fixture_name: str, tc: dict, base_path: str) -> pl.DataFrame:
    """Run a get_process_* with the fixture as the download and read the parquet back."""
    with patch.object(dcim, 'get_csv_from_url', return_value=fixture_df(fixture_name)):
        out_path = process_func(tc, base_path=base_path)
    assert out_path.endswith('.parquet'), f'processor returned a URL: {out_path}'
    return pl.read_parquet(out_path)


# ============================================================
# Node list
# ============================================================

class TestNodeList:

    def test_shapes_and_membership(self):
        assert len(WEST_HUB_BA_NODES) == 64
        assert len(STORED_NODES) == 74
        assert 'SWPW_HUB' in STORED_NODES
        assert 'CISO' in STORED_NODES
        assert set(EAST_HUB_NODES) == {
            'SPPNORTH_HUB', 'SPPSOUTH_HUB', 'CSWS_HUB', 'ETEC_HUB', 'GRDA_HUB',
            'GSEC_HUB', 'HAST_TNSK_HUB', 'KCPL_GMOC_HUB', 'LES_HUB', 'SECI_HUB',
        }
        assert len(set(STORED_NODES)) == len(STORED_NODES)


# ============================================================
# Time components / DST handling
# ============================================================

class TestGetTimeComponentsIm:

    def test_normal_hour(self):
        tc = tc_for('6/1/2026 07:30:00')
        assert tc['COMBINED'] == '202606010800'
        assert not tc['IS_AMBIGUOUS']
        assert not tc['DST_VARIANT']

    def test_five_min_ceil(self):
        tc = tc_for('6/1/2026 07:31:00', five_min=True)
        assert tc['COMBINED'] == '202606010735'

    def test_fallback_hour_is_ambiguous(self):
        # 2025-11-02 01:00-01:59 CT occurs twice
        tc = tc_for('11/2/2025 01:25:00', five_min=True)
        assert tc['COMBINED'] == '202511020125'
        assert tc['IS_AMBIGUOUS']

    def test_fallback_boundary_0200_is_ambiguous(self):
        # interval-ending semantics: the file ending 02:00 covers the
        # duplicated hour, so it has a d-variant (OP-MTLF-...0200d.csv)
        tc = tc_for('11/2/2025 01:30:00')
        assert tc['COMBINED'] == '202511020200'
        assert tc['IS_AMBIGUOUS']

    def test_dst_variants_differ_in_utc(self):
        first = tc_for('11/2/2025 01:25:00', five_min=True)
        second = tc_for('11/2/2025 01:25:00', five_min=True, dst_variant=True)
        assert first['COMBINED'] == second['COMBINED']
        assert second['timestamp_utc'] - first['timestamp_utc'] == pd.Timedelta(hours=1)

    def test_hour_after_fallback_not_ambiguous(self):
        tc = tc_for('11/2/2025 02:30:00')
        assert tc['COMBINED'] == '202511020300'
        assert not tc['IS_AMBIGUOUS']

    def test_spring_forward_keeps_wall_clock_label(self):
        # SPP publishes the interval ending inside the skipped hour
        tc = tc_for('3/8/2026 01:30:00')
        assert tc is not None
        assert tc['COMBINED'] == '202603080200'
        assert not tc['IS_AMBIGUOUS']


class TestGetRangeDataIm:

    def collect_urls(self, end_ts, n_periods, freq):
        seen = []

        def fake_process(tc, base_path=None):
            seen.append(dcim.get_hourly_mtlf_url(tc))
            return 'ok'

        dcim.get_range_data_im(end_ts, n_periods, freq, fake_process, do_parallel=False)
        return seen

    def test_fallback_day_adds_d_variant(self):
        urls = self.collect_urls(pd.Timestamp('2025-11-02 03:00:00'), 4, 'h')
        names = [u.split('%2F')[-1] for u in urls]
        assert 'OP-MTLF-202511020200.csv' in names
        assert 'OP-MTLF-202511020200d.csv' in names
        # 4 requested hours + 1 duplicate-hour variant
        assert len(names) == 5

    def test_normal_day_no_variants(self):
        urls = self.collect_urls(pd.Timestamp('2026-06-01 03:00:00'), 3, 'h')
        assert len(urls) == 3
        assert not any(u.endswith('d.csv') for u in urls)


# ============================================================
# URL builders
# ============================================================

class TestUrlBuilders:

    def test_urls(self):
        tc = tc_for('6/1/2026 11:58:00', five_min=True)
        assert dcim.get_5min_lmp_url(tc) == (
            'https://portal.spp.org/file-browser-api/download/rtbm-lmp-by-location'
            '?path=%2F2026%2F06%2FBy_Interval%2F01%2FRTBM-LMP-SL-202606011200.csv'
        )
        tc = tc_for('6/1/2026 09:30:00')
        assert dcim.get_hourly_mtlf_url(tc).endswith(
            'mtlf-vs-actual?path=%2F2026%2F06%2F01%2FOP-MTLF-202606011000.csv'
        )
        assert dcim.get_hourly_mtrf_url(tc).endswith(
            'midterm-resource-forecast?path=%2F2026%2F06%2F01%2FOP-MTRF-202606011000.csv'
        )
        assert dcim.get_rf_reserve_zone_url(tc).endswith(
            'resource-forecast-by-reserve-zone?path=%2F2026%2F06%2F01%2FRF_RESERVE_ZONE-202606011000.csv'
        )
        assert dcim.get_daily_lmp_url(tc).endswith(
            'rtbm-lmp-by-location?path=%2F2026%2F06%2FBy_Day%2FRTBM-LMP-DAILY-SL-20260601.csv'
        )
        assert dcim.get_da_lmp_url(tc).endswith(
            'da-lmp-by-settlement-location?path=%2F2026%2F06%2FBy_Day%2FDA-LMP-SL-202606010100.csv'
        )

    def test_dst_variant_suffix(self):
        tc = tc_for('11/2/2025 01:25:00', five_min=True, dst_variant=True)
        assert dcim.get_5min_lmp_url(tc).endswith('RTBM-LMP-SL-202511020125d.csv')


# ============================================================
# convert_datetime_cols
# ============================================================

class TestConvertDatetimeCols:

    def test_handles_both_da_formats(self):
        # seconded, seconds-less, and unpadded (older daily-rollup) forms
        df = pl.DataFrame(
            {'Interval': ['6/1/2026 01:00', '07/05/2026 06:00:00', '3/20/2026 0:05']}
        )
        out = dcim.convert_datetime_cols(df, ['Interval'])
        assert out['Interval'].dtype == pl.Datetime
        assert out['Interval'].null_count() == 0

    def test_raises_on_unknown_format(self):
        df = pl.DataFrame({'Interval': ['2026-06-01T01:00:00Z']})
        with pytest.raises(ValueError, match='no known datetime format'):
            dcim.convert_datetime_cols(df, ['Interval'])


# ============================================================
# ensure_baa
# ============================================================

class TestEnsureBaa:

    def test_fills_missing_column(self):
        df = pl.DataFrame({'x': [1, 2]})
        out = dcim.ensure_baa(df)
        assert out['BAA'].to_list() == ['SPP', 'SPP']

    def test_drops_null_rows(self):
        df = pl.DataFrame({'x': [1, 2], 'BAA': ['SWPW', None]})
        out = dcim.ensure_baa(df)
        assert out['BAA'].to_list() == ['SWPW']


# ============================================================
# Processors against real fixture files
# ============================================================

class TestMtlfMtrfProcessors:

    def test_mtlf_post_launch_keeps_both_baas(self, out_dir):
        tc = tc_for('7/4/2026 09:30:00')
        df = run_processor(dcim.get_process_mtlf, 'OP-MTLF-202607041000.csv', tc, out_dir)
        assert set(df['BAA'].unique()) == {'SPP', 'SWPW'}
        assert df['MTLF'].dtype == pl.Float32
        assert df['GMTIntervalEnd'].dtype == pl.Datetime
        assert 'timestamp_mst' in df.columns
        assert df['source'].unique().to_list() == ['im']

    def test_mtlf_pre_launch_fills_spp(self, out_dir):
        tc = tc_for('7/1/2025 09:30:00')
        df = run_processor(dcim.get_process_mtlf, 'OP-MTLF-202507011000.csv', tc, out_dir)
        assert df['BAA'].unique().to_list() == ['SPP']

    def test_mtrf_post_launch_drops_blank_baa_rows(self, out_dir):
        # the live file carries a leading unpopulated row with null BAA
        assert fixture_df('OP-MTRF-202607041000.csv')['BAA'].null_count() > 0
        tc = tc_for('7/4/2026 09:30:00')
        df = run_processor(dcim.get_process_mtrf, 'OP-MTRF-202607041000.csv', tc, out_dir)
        assert df['BAA'].null_count() == 0
        assert set(df['BAA'].unique()) == {'SPP', 'SWPW'}

    def test_mtrf_pre_launch_fills_spp(self, out_dir):
        tc = tc_for('7/1/2025 09:30:00')
        df = run_processor(dcim.get_process_mtrf, 'OP-MTRF-202507011000.csv', tc, out_dir)
        assert df['BAA'].unique().to_list() == ['SPP']
        assert df['Wind_Forecast_MW'].dtype == pl.Float32


class TestLmpProcessors:

    def test_5min_post_launch_filters_and_aggs(self, out_dir):
        tc = tc_for('6/1/2026 12:03:00', five_min=True)
        df = run_processor(dcim.get_process_5min_lmp, 'RTBM-LMP-SL-202606011205.csv', tc, out_dir)
        locs = set(df['Settlement_Location_Name'].unique())
        assert 'AEC' not in locs  # out-of-list node dropped at storage
        assert {'SWPW_HUB', 'CISO', 'PSCO', 'SPPNORTH_HUB', 'SPPSOUTH_HUB'} == locs
        assert set(df['BAA'].unique()) == {'SPP', 'SWPW'}
        assert 'GMTIntervalEnd_HE' in df.columns  # aggregated to hour ending
        assert df['LMP'].dtype == pl.Float32

    def test_5min_pre_launch_fills_spp(self, out_dir):
        tc = tc_for('7/1/2025 12:03:00', five_min=True)
        df = run_processor(dcim.get_process_5min_lmp, 'RTBM-LMP-SL-202507011205.csv', tc, out_dir)
        assert df['BAA'].unique().to_list() == ['SPP']
        assert 'AEC' not in df['Settlement_Location_Name'].unique().to_list()

    def test_daily_post_launch(self, out_dir):
        tc = tc_for('6/1/2026')
        df = run_processor(dcim.get_process_daily_lmp, 'RTBM-LMP-DAILY-SL-20260601.csv', tc, out_dir)
        assert 'AEC' not in df['Settlement_Location_Name'].unique().to_list()
        assert set(df['BAA'].unique()) == {'SPP', 'SWPW'}
        # fixture holds two hours of 5-min intervals -> two HE rows per node
        assert df.group_by('Settlement_Location_Name').len()['len'].max() == 2

    def test_daily_pre_launch_fills_spp(self, out_dir):
        tc = tc_for('7/1/2025')
        df = run_processor(dcim.get_process_daily_lmp, 'RTBM-LMP-DAILY-SL-20250701.csv', tc, out_dir)
        assert df['BAA'].unique().to_list() == ['SPP']

    def test_daily_unpadded_datetime_format(self, out_dir):
        # Older daily-rollup files use an unpadded, seconds-less timestamp
        # (e.g. '3/20/2026 0:05'); the LMP path must parse it, not just the
        # zero-padded seconded format. Regression for the backfill failure.
        tc = tc_for('3/20/2026')
        df = run_processor(dcim.get_process_daily_lmp, 'RTBM-LMP-DAILY-SL-20260320.csv', tc, out_dir)
        assert df['GMTIntervalEnd_HE'].dtype == pl.Datetime
        assert df['GMTIntervalEnd_HE'].null_count() == 0
        assert 'SWPW_HUB' in df['Settlement_Location_Name'].unique().to_list()

    def test_da_lmp_post_launch(self, out_dir):
        tc = tc_for('6/1/2026')
        df = run_processor(dcim.get_process_da_lmp, 'DA-LMP-SL-202606010100.csv', tc, out_dir)
        assert 'AEC' not in df['Settlement_Location_Name'].unique().to_list()
        assert set(df['BAA'].unique()) == {'SPP', 'SWPW'}
        # DA timestamps have no seconds; make sure they parsed
        assert df['GMTIntervalEnd'].dtype == pl.Datetime
        assert df['GMTIntervalEnd'].null_count() == 0
        assert df['LMP'].dtype == pl.Float32


class TestRfReserveZoneProcessor:

    def test_post_launch_drops_blanks_keeps_all_zones(self, out_dir):
        # ' BAA' (leading space): raw portal header, format_df_colnames not run yet
        assert fixture_df('RF_RESERVE_ZONE-202607041000.csv')[' BAA'].null_count() > 0
        tc = tc_for('7/4/2026 09:30:00')
        df = run_processor(
            dcim.get_process_rf_reserve_zone, 'RF_RESERVE_ZONE-202607041000.csv', tc, out_dir
        )
        assert df['BAA'].null_count() == 0
        assert set(df['ReserveZone'].unique()) == {1, 2, 3, 4, 5, 21}
        assert df.filter(pl.col('ReserveZone') == 21)['BAA'].unique().to_list() == ['SWPW']
        assert df['WindForecastMW'].dtype == pl.Float32

    def test_pre_launch_fills_spp(self, out_dir):
        tc = tc_for('7/1/2025 09:30:00')
        df = run_processor(
            dcim.get_process_rf_reserve_zone, 'RF_RESERVE_ZONE-202507010000.csv', tc, out_dir
        )
        assert df['BAA'].unique().to_list() == ['SPP']
        assert 21 not in df['ReserveZone'].unique().to_list()


# ============================================================
# Upsert
# ============================================================

class TestUpsertIm:

    def make_batch(self, out_dir, name, baa, mtlf, created):
        df = pl.DataFrame({
            'GMTIntervalEnd': [pd.Timestamp('2026-07-01 12:00:00')],
            'BAA': [baa],
            'MTLF': [mtlf],
            'file_create_time_utc': [pd.Timestamp(created)],
        })
        path = f'{out_dir}mtlf/{name}.parquet'
        df.write_parquet(path)
        return path

    def test_both_baas_survive_same_interval(self, out_dir):
        # regression: without BAA in the key, East and West rows clobber
        files = [
            self.make_batch(out_dir, 'east', 'SPP', 30000.0, '2026-07-01 12:05:00'),
            self.make_batch(out_dir, 'west', 'SWPW', 3000.0, '2026-07-01 12:05:00'),
        ]
        dcim.upsert_im(files, 'mtlf', base_path=out_dir)
        result = pl.read_parquet(f'{out_dir}mtlf.parquet')
        assert result.shape[0] == 2
        assert set(result['BAA'].unique()) == {'SPP', 'SWPW'}

    def test_latest_file_wins_on_same_key(self, out_dir):
        files = [
            self.make_batch(out_dir, 'old', 'SWPW', 1000.0, '2026-07-01 12:05:00'),
            self.make_batch(out_dir, 'new', 'SWPW', 2000.0, '2026-07-01 13:05:00'),
        ]
        dcim.upsert_im(files, 'mtlf', base_path=out_dir)
        result = pl.read_parquet(f'{out_dir}mtlf.parquet')
        assert result.shape[0] == 1
        assert result['MTLF'][0] == 2000.0

    def test_merges_with_existing_target(self, out_dir):
        dcim.upsert_im(
            [self.make_batch(out_dir, 'a', 'SPP', 1.0, '2026-07-01 12:05:00')],
            'mtlf', base_path=out_dir,
        )
        dcim.upsert_im(
            [self.make_batch(out_dir, 'b', 'SWPW', 2.0, '2026-07-01 12:05:00')],
            'mtlf', base_path=out_dir,
        )
        result = pl.read_parquet(f'{out_dir}mtlf.parquet')
        assert result.shape[0] == 2

    def test_merges_across_column_order_drift(self, out_dir):
        # regression: RF files place BAA mid-schema post-launch but appended
        # (via ensure_baa) pre-launch, so the stored table and new files can
        # differ in column order. upsert_im must align by name, not crash on
        # the positional vstack.
        dcim.upsert_im(
            [self.make_batch(out_dir, 'a', 'SPP', 1.0, '2026-07-01 12:05:00')],
            'mtlf', base_path=out_dir,
        )
        # second batch with the same columns in a different order
        reordered = pl.DataFrame({
            'BAA': ['SWPW'],
            'MTLF': [2.0],
            'file_create_time_utc': [pd.Timestamp('2026-07-01 12:05:00')],
            'GMTIntervalEnd': [pd.Timestamp('2026-07-01 12:00:00')],
        })
        path = f'{out_dir}mtlf/reordered.parquet'
        reordered.write_parquet(path)
        dcim.upsert_im([path], 'mtlf', base_path=out_dir)

        result = pl.read_parquet(f'{out_dir}mtlf.parquet')
        assert result.shape[0] == 2
        assert set(result['BAA'].unique()) == {'SPP', 'SWPW'}

    def test_unknown_target_raises(self, out_dir):
        with pytest.raises(ValueError):
            dcim.upsert_im([], 'gen_cap', base_path=out_dir)

    def test_keys_include_baa_everywhere(self):
        assert all('BAA' in keys for keys in dcim.UPSERT_KEYS.values())
        assert 'ReserveZone' in dcim.UPSERT_KEYS['rf_reserve_zone']


# ============================================================
# Range helpers
# ============================================================

class TestRangeHelpers:

    def test_daily_lmp_window_is_lag_adjusted(self):
        captured = {}

        def fake_range(end_ts, n_periods, freq, func, base_path=None):
            captured.update(end_ts=end_ts, freq=freq)
            return []

        with patch.object(dcim, 'get_range_data_im', side_effect=fake_range):
            dcim.get_range_data_daily_lmp(pd.Timestamp('2026-07-05'), 7)
        assert captured['end_ts'] == pd.Timestamp('2026-06-30')
        assert captured['freq'] == 'D'

"""
Unit tests for modeling.load_ensemble_from_dir

Tests cover:
- .pt file filtering (excludes .ckpt, .pkl)
- Model class dispatch via MODEL_CLASS_MAP
- NaiveEnsembleModel construction
- TRAIN_TIMESTAMP.pkl loading
"""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import modeling


def _make_model_mock(name):
    """Create a MagicMock with __name__ set so logging doesn't fail."""
    m = MagicMock()
    m.__name__ = name
    m.load.return_value = MagicMock(name=f'{name}_instance')
    return m


# ============================================================
# Test load_ensemble_from_dir
# ============================================================

class TestLoadEnsembleFromDir:
    """Tests for modeling.load_ensemble_from_dir function."""

    @patch('builtins.open', mock_open(read_data=b''))
    @patch('modeling.pickle.load')
    @patch('modeling.NaiveEnsembleModel')
    @patch('modeling.os.listdir')
    def test_loads_all_model_types(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """Dir has tsmixer_0.pt, tide_0.pt, tft_0.pt → all 3 model classes loaded."""
        mock_listdir.return_value = [
            'tsmixer_0.pt', 'tide_0.pt', 'tft_0.pt', 'TRAIN_TIMESTAMP.pkl',
        ]
        mock_tsmixer = _make_model_mock('TSMixerModel')
        mock_tide = _make_model_mock('TiDEModel')
        mock_tft = _make_model_mock('TFTModel')
        mock_pickle.return_value = pd.Timestamp('2026-03-01 12:00:00', tz='UTC')
        mock_ensemble.return_value = MagicMock(name='ensemble')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': mock_tsmixer,
            'tide_': mock_tide,
            'tft': mock_tft,
        }):
            ensemble, ts = modeling.load_ensemble_from_dir('/tmp/models')

        mock_tsmixer.load.assert_called_once()
        mock_tide.load.assert_called_once()
        mock_tft.load.assert_called_once()

    @patch('builtins.open', mock_open(read_data=b''))
    @patch('modeling.pickle.load')
    @patch('modeling.NaiveEnsembleModel')
    @patch('modeling.os.listdir')
    def test_excludes_ckpt_and_pkl(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """Only .pt files are loaded; .ckpt and .pkl are excluded."""
        mock_listdir.return_value = [
            'tsmixer_0.pt',
            'tsmixer_0.pt.ckpt',
            'TRAIN_TIMESTAMP.pkl',
        ]
        mock_tsmixer = _make_model_mock('TSMixerModel')
        mock_pickle.return_value = pd.Timestamp('2026-03-01', tz='UTC')
        mock_ensemble.return_value = MagicMock(name='ensemble')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': mock_tsmixer,
            'tide_': _make_model_mock('TiDEModel'),
            'tft': _make_model_mock('TFTModel'),
        }):
            modeling.load_ensemble_from_dir('/tmp/models')

        # Only the .pt file should be loaded, not the .ckpt
        mock_tsmixer.load.assert_called_once()

    @patch('builtins.open', mock_open(read_data=b''))
    @patch('modeling.pickle.load')
    @patch('modeling.NaiveEnsembleModel')
    @patch('modeling.os.listdir')
    def test_builds_ensemble(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """NaiveEnsembleModel is called with loaded models and train_forecasting_models=False."""
        mock_listdir.return_value = [
            'tsmixer_0.pt', 'tide_0.pt', 'TRAIN_TIMESTAMP.pkl',
        ]
        mock_tsmixer = _make_model_mock('TSMixerModel')
        mock_tide = _make_model_mock('TiDEModel')
        model_a = MagicMock(name='model_a')
        model_b = MagicMock(name='model_b')
        mock_tsmixer.load.return_value = model_a
        mock_tide.load.return_value = model_b
        mock_pickle.return_value = pd.Timestamp('2026-03-01', tz='UTC')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': mock_tsmixer,
            'tide_': mock_tide,
            'tft': _make_model_mock('TFTModel'),
        }):
            modeling.load_ensemble_from_dir('/tmp/models')

        mock_ensemble.assert_called_once()
        call_kwargs = mock_ensemble.call_args
        assert call_kwargs.kwargs['train_forecasting_models'] is False
        assert len(call_kwargs.kwargs['forecasting_models']) == 2

    @patch('builtins.open', mock_open(read_data=b''))
    @patch('modeling.pickle.load')
    @patch('modeling.NaiveEnsembleModel')
    @patch('modeling.os.listdir')
    def test_loads_timestamp(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """pickle loads TRAIN_TIMESTAMP.pkl and returns it as second tuple element."""
        mock_listdir.return_value = ['tide_0.pt', 'TRAIN_TIMESTAMP.pkl']
        expected_ts = pd.Timestamp('2026-02-28 08:30:00', tz='UTC')
        mock_pickle.return_value = expected_ts
        mock_ensemble.return_value = MagicMock(name='ensemble')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': _make_model_mock('TSMixerModel'),
            'tide_': _make_model_mock('TiDEModel'),
            'tft': _make_model_mock('TFTModel'),
        }):
            _, ts = modeling.load_ensemble_from_dir('/tmp/models')

        assert ts == expected_ts

    @patch('builtins.open', mock_open(read_data=b''))
    @patch('modeling.pickle.load')
    @patch('modeling.NaiveEnsembleModel')
    @patch('modeling.os.listdir')
    def test_empty_directory_raises(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """No .pt files → raise rather than serve a zero-model ensemble."""
        mock_listdir.return_value = ['TRAIN_TIMESTAMP.pkl']
        mock_pickle.return_value = pd.Timestamp('2026-03-01', tz='UTC')
        mock_ensemble.return_value = MagicMock(name='ensemble')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': _make_model_mock('TSMixerModel'),
            'tide_': _make_model_mock('TiDEModel'),
            'tft': _make_model_mock('TFTModel'),
        }):
            with pytest.raises(ValueError, match='no loadable model checkpoints'):
                modeling.load_ensemble_from_dir('/tmp/models')

    @patch('builtins.open', mock_open(read_data=b''))
    @patch('modeling.pickle.load')
    @patch('modeling.NaiveEnsembleModel')
    @patch('modeling.os.listdir')
    def test_unmatched_checkpoint_raises(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """A .pt file matching no class substring raises instead of being
        silently dropped into a smaller ensemble."""
        mock_listdir.return_value = ['mystery_0.pt', 'TRAIN_TIMESTAMP.pkl']
        mock_pickle.return_value = pd.Timestamp('2026-03-01', tz='UTC')
        mock_ensemble.return_value = MagicMock(name='ensemble')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': _make_model_mock('TSMixerModel'),
            'tide_': _make_model_mock('TiDEModel'),
            'tft': _make_model_mock('TFTModel'),
        }):
            with pytest.raises(ValueError, match='matches no known'):
                modeling.load_ensemble_from_dir('/tmp/models')

    @patch('builtins.open', mock_open(read_data=b''))
    @patch('modeling.pickle.load')
    @patch('modeling.NaiveEnsembleModel')
    @patch('modeling.os.listdir')
    def test_tide_pattern_does_not_match_tft(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """'tide_' pattern should not match 'tft_0.pt' — specificity check."""
        mock_listdir.return_value = ['tft_0.pt', 'TRAIN_TIMESTAMP.pkl']
        mock_tide = _make_model_mock('TiDEModel')
        mock_tft = _make_model_mock('TFTModel')
        mock_pickle.return_value = pd.Timestamp('2026-03-01', tz='UTC')
        mock_ensemble.return_value = MagicMock(name='ensemble')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': _make_model_mock('TSMixerModel'),
            'tide_': mock_tide,
            'tft': mock_tft,
        }):
            modeling.load_ensemble_from_dir('/tmp/models')

        # tide_ should NOT have been called for a tft file
        mock_tide.load.assert_not_called()
        mock_tft.load.assert_called_once()


class TestCoverageMetric:
    """Per-band coverage metric used to score every band off one backtest."""

    @staticmethod
    def _series(n=100, n_samples=1001):
        """Actuals at 0.0, predictions spread uniformly on [-1, 1].

        Every band straddles 0, so realized coverage is 1.0 and the deviation
        from nominal is exactly computable rather than sampled.
        """
        import numpy as np
        from darts import TimeSeries

        idx = pd.date_range('2026-01-01', periods=n, freq='h')
        actual = TimeSeries.from_times_and_values(
            idx, np.zeros((n, 1)), columns=['LMP']
        )
        samples = np.linspace(-1.0, 1.0, n_samples)
        pred = TimeSeries.from_times_and_values(
            idx, np.tile(samples, (n, 1, 1)), columns=['LMP']
        )
        return actual, pred

    def test_reports_raw_coverage_not_error(self):
        # Raw coverage is what makes the direction of miscalibration visible;
        # the error is derived from it and is unsigned.
        import numpy as np

        actual, pred = self._series()
        cov = float(np.mean(modeling.coverage_metric((0.1, 0.9))(actual, pred)))
        assert cov == pytest.approx(1.0, abs=0.01)

    def test_agrees_with_the_harness_coverage(self):
        # The study and the promote gate must measure calibration the same way;
        # both go through darts' mic.
        import numpy as np
        from darts.metrics import mic

        actual, pred = self._series()
        band = (0.05, 0.95)
        assert float(np.mean(modeling.coverage_metric(band)(actual, pred))) == (
            pytest.approx(float(np.mean(mic(actual, pred, q_interval=band))))
        )

    def test_each_band_gets_a_distinct_name(self):
        # Darts identifies metrics by name, so two bands sharing one would
        # collide in the backtest's metric columns.
        import selection

        names = [
            modeling.coverage_metric(b).__name__
            for b in selection.DIAGNOSTIC_BANDS
        ]
        assert names == [
            selection.coverage_label(b) for b in selection.DIAGNOSTIC_BANDS
        ]
        assert len(set(names)) == len(names)

    def test_keeps_the_wrapped_signature(self):
        # backtest passes metric_kwargs only for params in the signature.
        import inspect

        params = inspect.signature(modeling.coverage_metric((0.1, 0.9))).parameters
        assert 'actual_series' in params and 'pred_series' in params

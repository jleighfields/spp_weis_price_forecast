"""
Unit tests for modeling.load_ensemble_from_dir

Tests cover:
- .pt file filtering (excludes .ckpt, .pkl)
- Model class dispatch via MODEL_CLASS_MAP
- NaiveEnsembleModel construction
- TRAIN_TIMESTAMP.pkl loading
"""

import os
import pickle
import sys

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open, create_autospec

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
        mock_listdir.return_value = ['TRAIN_TIMESTAMP.pkl']
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
    def test_empty_directory(
        self, mock_listdir, mock_ensemble, mock_pickle,
    ):
        """No .pt files → ensemble built with empty list (no error)."""
        mock_listdir.return_value = ['TRAIN_TIMESTAMP.pkl']
        mock_pickle.return_value = pd.Timestamp('2026-03-01', tz='UTC')
        mock_ensemble.return_value = MagicMock(name='ensemble')

        with patch.dict(modeling.MODEL_CLASS_MAP, {
            'tsmixer': _make_model_mock('TSMixerModel'),
            'tide_': _make_model_mock('TiDEModel'),
            'tft': _make_model_mock('TFTModel'),
        }):
            modeling.load_ensemble_from_dir('/tmp/models')

        call_kwargs = mock_ensemble.call_args
        assert call_kwargs.kwargs['forecasting_models'] == []

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

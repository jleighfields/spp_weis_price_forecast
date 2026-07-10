"""
Unit tests for S3 download helpers in src/utils.py

Tests cover:
- download_checkpoints: downloading model files from S3 to local directory
- download_champion_checkpoints: reading champion.json and delegating to download_checkpoints
"""

import json
import os
import sys

import pytest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import utils


# ============================================================
# Test download_checkpoints
# ============================================================

class TestDownloadCheckpoints:
    """Tests for utils.download_checkpoints function."""

    @patch('utils.boto3.client')
    @patch('utils.get_loaded_models')
    @patch.dict(os.environ, {
        'AWS_S3_BUCKET': 'test-bucket',
        'S3_ENDPOINT_URL': 'https://s3.example.com',
    })
    def test_downloads_all_files(self, mock_get_models, mock_boto_client):
        """3 model keys → 3 download_file calls with correct args."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_get_models.return_value = [
            'folder/tsmixer_0.pt',
            'folder/tide_0.pt',
            'folder/tft_0.pt',
        ]

        utils.download_checkpoints('models/rt/retrains/2026-03-01/', '/tmp/models')

        assert mock_s3.download_file.call_count == 3
        mock_s3.download_file.assert_any_call(
            Bucket='test-bucket',
            Key='folder/tsmixer_0.pt',
            Filename='/tmp/models/tsmixer_0.pt',
        )
        mock_s3.download_file.assert_any_call(
            Bucket='test-bucket',
            Key='folder/tide_0.pt',
            Filename='/tmp/models/tide_0.pt',
        )
        mock_s3.download_file.assert_any_call(
            Bucket='test-bucket',
            Key='folder/tft_0.pt',
            Filename='/tmp/models/tft_0.pt',
        )

    @patch('utils.boto3.client')
    @patch('utils.get_loaded_models')
    @patch.dict(os.environ, {
        'AWS_S3_BUCKET': 'test-bucket',
        'S3_ENDPOINT_URL': 'https://s3.example.com',
    })
    def test_empty_file_list(self, mock_get_models, mock_boto_client):
        """get_loaded_models returns [] → no download_file calls."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_get_models.return_value = []

        utils.download_checkpoints('models/rt/retrains/empty/', '/tmp/models')

        mock_s3.download_file.assert_not_called()

    @patch('utils.boto3.client')
    @patch('utils.get_loaded_models')
    @patch.dict(os.environ, {
        'AWS_S3_BUCKET': 'test-bucket',
        'S3_ENDPOINT_URL': 'https://s3.example.com',
    })
    def test_extracts_filename_from_key(self, mock_get_models, mock_boto_client):
        """S3 key with nested path → local file uses only the basename."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_get_models.return_value = [
            'deep/nested/folder/sub/tsmixer_0.pt',
        ]

        utils.download_checkpoints('models/rt/retrains/nested/', '/tmp/dest')

        mock_s3.download_file.assert_called_once_with(
            Bucket='test-bucket',
            Key='deep/nested/folder/sub/tsmixer_0.pt',
            Filename='/tmp/dest/tsmixer_0.pt',
        )


# ============================================================
# Test download_champion_checkpoints
# ============================================================

class TestDownloadChampionCheckpoints:
    """Tests for utils.download_champion_checkpoints function."""

    @patch('utils.download_checkpoints')
    @patch('utils.boto3.client')
    @patch.dict(os.environ, {
        'AWS_S3_BUCKET': 'test-bucket',
        'AWS_S3_FOLDER': 'prod/',
        'S3_ENDPOINT_URL': 'https://s3.example.com',
    })
    def test_reads_champion_json_and_delegates(self, mock_boto_client, mock_dl):
        """Reads champion.json, extracts folder, delegates to download_checkpoints."""
        champion_config = {
            'champion_artifact_folder': 'models/rt/retrains/2026-02-28_10-00-00/',
        }
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(champion_config).encode('utf-8')),
        }

        utils.download_champion_checkpoints('/tmp/champ')

        mock_dl.assert_called_once_with(
            'models/rt/retrains/2026-02-28_10-00-00/',
            '/tmp/champ',
        )

    @patch('utils.download_checkpoints')
    @patch('utils.boto3.client')
    @patch.dict(os.environ, {
        'AWS_S3_BUCKET': 'test-bucket',
        'AWS_S3_FOLDER': 'staging/',
        'S3_ENDPOINT_URL': 'https://s3.example.com',
    })
    def test_uses_aws_folder_prefix(self, mock_boto_client, mock_dl):
        """Champion key is AWS_S3_FOLDER + 'models/rt/champion.json'."""
        champion_config = {'champion_artifact_folder': 'models/rt/retrains/latest/'}
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(champion_config).encode('utf-8')),
        }

        utils.download_champion_checkpoints('/tmp/champ')

        mock_s3.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='staging/models/rt/champion.json',
        )


# ============================================================
# Test per-target model namespace
# ============================================================

class TestTargetNamespace:
    """Each forecast target gets its own models/<target>/ storage namespace."""

    def test_prefix_and_key_are_target_scoped(self):
        assert utils.retrains_prefix('rt') == 'models/rt/retrains/'
        assert utils.retrains_prefix('da') == 'models/da/retrains/'
        assert utils.champion_key_suffix('rt') == 'models/rt/champion.json'
        assert utils.champion_key_suffix('da') == 'models/da/champion.json'

    def test_rt_default_constants_match_helpers(self):
        # the back-compat constants are the RT-default values
        assert utils.RETRAINS_PREFIX == utils.retrains_prefix('rt')
        assert utils.CHAMPION_KEY_SUFFIX == utils.champion_key_suffix('rt')

    @patch('utils.download_checkpoints')
    @patch('utils.boto3.client')
    @patch.dict(os.environ, {
        'AWS_S3_BUCKET': 'test-bucket',
        'AWS_S3_FOLDER': '',
        'S3_ENDPOINT_URL': 'https://s3.example.com',
    })
    def test_download_champion_reads_target_pointer(self, mock_boto_client, mock_dl):
        """target='da' reads models/da/champion.json, not the RT pointer."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(
                {'champion_artifact_folder': 'models/da/retrains/x/'}).encode('utf-8')),
        }

        utils.download_champion_checkpoints('/tmp/champ', target='da')

        mock_s3.get_object.assert_called_once_with(
            Bucket='test-bucket', Key='models/da/champion.json')
        mock_dl.assert_called_once_with('models/da/retrains/x/', '/tmp/champ')


# ============================================================
# Test training-config helpers
# ============================================================

def _sample_config():
    return utils.build_training_config(
        train_timestamp='2026-07-07T03:38:56',
        future_covariates=['MTLF', 're_ratio'],
        past_covariates=['lmp_diff'],
        nodes=['BHBA', 'BPA'],
        quantiles=[0.05, 0.5, 0.95],
        model_name='spp_west',
        model_types=['tide'],
        forecast_horizon=120,
        input_chunk_length=168,
        train_start='2026-04-15',
        train_end='2026-06-29',
        darts_version='0.45.0',
        torch_version='2.11.0+cu128',
    )


class TestTrainingConfig:
    def test_build_includes_covariate_and_provenance_fields(self):
        cfg = _sample_config()
        assert cfg['future_covariates'] == ['MTLF', 're_ratio']
        assert cfg['past_covariates'] == ['lmp_diff']
        assert cfg['nodes'] == ['BHBA', 'BPA']
        assert cfg['darts_version'] == '0.45.0'

    def test_validate_passes_on_match(self):
        # no exception when the serving covariates match the config
        utils.validate_model_covariates(_sample_config(), ['MTLF', 're_ratio'], ['lmp_diff'])

    def test_validate_raises_on_future_mismatch(self):
        with pytest.raises(ValueError, match='future covariate mismatch'):
            utils.validate_model_covariates(
                _sample_config(), ['MTLF', 're_ratio', 'break_indicator'], ['lmp_diff'])

    def test_validate_raises_on_reorder(self):
        # order matters — darts binds covariates positionally
        with pytest.raises(ValueError, match='future covariate mismatch'):
            utils.validate_model_covariates(_sample_config(), ['re_ratio', 'MTLF'], ['lmp_diff'])

    def test_validate_raises_on_past_mismatch(self):
        with pytest.raises(ValueError, match='past covariate mismatch'):
            utils.validate_model_covariates(_sample_config(), ['MTLF', 're_ratio'], ['lmp_diff', 'x'])

    def test_validate_skips_missing_fields(self):
        # legacy config without covariate keys -> no validation, no error
        utils.validate_model_covariates({}, ['anything'], ['anything'])

    def test_load_returns_none_when_absent(self, tmp_path):
        assert utils.load_training_config(str(tmp_path)) is None

    def test_load_round_trips(self, tmp_path):
        cfg = _sample_config()
        (tmp_path / utils.TRAINING_CONFIG_FILENAME).write_text(json.dumps(cfg))
        assert utils.load_training_config(str(tmp_path)) == cfg

    def test_active_model_types(self):
        assert utils.active_model_types(True, False, False) == ['tide']
        assert utils.active_model_types(True, True, True) == ['tide', 'tsmixer', 'tft']
        assert utils.active_model_types(False, False, False) == []


class TestGetLoadedModelsFilter:
    """get_loaded_models must download training_config.json with the checkpoints
    (else the app can't validate covariates), but must NOT match a bare
    champion.json pointer."""

    @staticmethod
    def _obj(key):
        m = MagicMock()
        m.key = key
        return m

    @patch.dict(os.environ, {'AWS_S3_BUCKET': 'b', 'AWS_S3_FOLDER': ''})
    @patch('utils.list_folder_contents_resource')
    def test_includes_config_and_checkpoints(self, mock_list):
        mock_list.return_value = [self._obj(k) for k in [
            'models/rt/retrains/ts/tide_0.pt',
            'models/rt/retrains/ts/tide_0.pt.ckpt',
            'models/rt/retrains/ts/TRAIN_TIMESTAMP.pkl',
            'models/rt/retrains/ts/training_config.json',
            'models/rt/retrains/ts/notes.txt',
        ]]
        keys = utils.get_loaded_models('models/rt/retrains/ts/')
        assert 'models/rt/retrains/ts/training_config.json' in keys
        assert 'models/rt/retrains/ts/tide_0.pt' in keys
        assert 'models/rt/retrains/ts/notes.txt' not in keys  # unrecognized file

    @patch.dict(os.environ, {'AWS_S3_BUCKET': 'b', 'AWS_S3_FOLDER': ''})
    @patch('utils.list_folder_contents_resource')
    def test_excludes_champion_json_pointer(self, mock_list):
        mock_list.return_value = [self._obj('models/rt/champion.json')]
        assert utils.get_loaded_models('models/') == []

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
from unittest.mock import patch, MagicMock, call

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

        utils.download_checkpoints('S3_models/2026-03-01/', '/tmp/models')

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

        utils.download_checkpoints('S3_models/empty/', '/tmp/models')

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

        utils.download_checkpoints('S3_models/nested/', '/tmp/dest')

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
            'champion_artifact_folder': 'S3_models/2026-02-28_10-00-00/',
        }
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(champion_config).encode('utf-8')),
        }

        utils.download_champion_checkpoints('/tmp/champ')

        mock_dl.assert_called_once_with(
            'S3_models/2026-02-28_10-00-00/',
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
        """Champion key is AWS_S3_FOLDER + 'S3_models/champion.json'."""
        champion_config = {'champion_artifact_folder': 'S3_models/latest/'}
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(champion_config).encode('utf-8')),
        }

        utils.download_champion_checkpoints('/tmp/champ')

        mock_s3.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='staging/S3_models/champion.json',
        )

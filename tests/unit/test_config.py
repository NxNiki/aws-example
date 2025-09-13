"""
Unit tests for the config module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bituslabs_ds.config import (
    DEFAULT_ATHENA_OUTPUT,
    DEFAULT_MAX_JOBS,
    IMAGE_URI,
    LOCAL_ROOT,
    REGION,
    S3_BUCKET,
    SAGEMAKER_ROLE,
    get_cpu_cores,
    setup_logging,
)


class TestGetCpuCores:
    """Test the get_cpu_cores function."""

    def test_get_cpu_cores_returns_positive_integer(self):
        """Test that get_cpu_cores returns a positive integer."""
        result = get_cpu_cores()
        assert isinstance(result, int)
        assert result > 0

    @patch("os.cpu_count")
    @patch("psutil.cpu_count")
    def test_get_cpu_cores_handles_exceptions(self, mock_psutil_cpu_count, mock_os_cpu_count):
        """Test that get_cpu_cores handles exceptions gracefully."""
        mock_os_cpu_count.side_effect = Exception("Test error")
        mock_psutil_cpu_count.side_effect = ImportError("No psutil")

        result = get_cpu_cores()
        assert result == 1  # Default value

    @patch("os.cpu_count")
    @patch("psutil.cpu_count")
    def test_get_cpu_cores_custom_default(self, mock_psutil_cpu_count, mock_os_cpu_count):
        """Test custom default value."""
        mock_os_cpu_count.side_effect = Exception("Test error")
        mock_psutil_cpu_count.side_effect = ImportError("No psutil")

        result = get_cpu_cores(default=4)
        assert result == 4


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_setup_logging_with_custom_path(self, temp_dir):
        """Test setting up logging with custom path."""
        log_dir = temp_dir / "logs"
        log_filename = "test.log"

        setup_logging(str(log_dir), log_filename)

        log_file = log_dir / log_filename
        assert log_file.exists()

    def test_setup_logging_with_dot_path(self, temp_dir):
        """Test setting up logging with '.' path."""
        with patch("os.getcwd", return_value=str(temp_dir)):
            setup_logging(".", "test.log")

            log_file = temp_dir / ".log" / "test.log"
            assert log_file.exists()

    def test_setup_logging_auto_filename(self, temp_dir):
        """Test setting up logging with auto-generated filename."""
        log_dir = temp_dir / "logs"

        setup_logging(str(log_dir), "")

        # Should create a file with timestamp
        log_files = list(log_dir.glob("job_*.log"))
        assert len(log_files) == 1

    def test_setup_logging_creates_directory(self, temp_dir):
        """Test that setup_logging creates the directory if it doesn't exist."""
        log_dir = temp_dir / "new_logs"

        setup_logging(str(log_dir), "test.log")

        assert log_dir.exists()
        assert (log_dir / "test.log").exists()


class TestConfigConstants:
    """Test configuration constants."""

    def test_region_constant(self):
        """Test REGION constant."""
        assert REGION == "us-west-2"
        assert isinstance(REGION, str)

    def test_s3_bucket_constant(self):
        """Test S3_BUCKET constant."""
        assert S3_BUCKET == "bituslabs-team-ai"
        assert isinstance(S3_BUCKET, str)

    def test_sagemaker_role_constant(self):
        """Test SAGEMAKER_ROLE constant."""
        assert SAGEMAKER_ROLE.startswith("arn:aws:iam::")
        assert "SageMakerExecutionRole" in SAGEMAKER_ROLE

    def test_image_uri_constant(self):
        """Test IMAGE_URI constant."""
        assert IMAGE_URI.startswith("338568447110.dkr.ecr.")
        assert "bituslabs-ds-sagemaker" in IMAGE_URI

    def test_default_max_jobs_constant(self):
        """Test DEFAULT_MAX_JOBS constant."""
        assert DEFAULT_MAX_JOBS == 4
        assert isinstance(DEFAULT_MAX_JOBS, int)
        assert DEFAULT_MAX_JOBS > 0

    def test_default_athena_output_constant(self):
        """Test DEFAULT_ATHENA_OUTPUT constant."""
        assert DEFAULT_ATHENA_OUTPUT.startswith("s3://")
        assert S3_BUCKET in DEFAULT_ATHENA_OUTPUT
        assert "athena-results" in DEFAULT_ATHENA_OUTPUT

    def test_local_root_constant(self):
        """Test LOCAL_ROOT constant."""
        assert isinstance(LOCAL_ROOT, Path)
        assert LOCAL_ROOT.exists()  # Should point to project root

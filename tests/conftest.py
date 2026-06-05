"""
Pytest configuration and shared fixtures for the test suite.
"""

import os
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import boto3
import numpy as np
import pandas as pd
import pytest
from moto import mock_athena, mock_s3


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(100),
            "value": np.random.normal(0, 1, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="H"),
            "profit": np.random.normal(10, 5, 100),
            "bet_amount": np.random.exponential(2, 100),
            "slot_type": np.random.choice([1, 2, 3], 100),
        }
    )


@pytest.fixture
def sample_numeric_dataframe():
    """Create a sample DataFrame with only numeric columns."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 50),
            "feature_2": np.random.exponential(1, 50),
            "feature_3": np.random.uniform(-5, 5, 50),
            "target": np.random.choice([0, 1], 50),
        }
    )


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=1000, freq="H")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "value": np.random.normal(0, 1, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 24),
            "group": np.random.choice(["A", "B"], 1000),
        }
    )


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    with mock_s3():
        client = boto3.client("s3", region_name="us-west-2")
        client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        yield client


@pytest.fixture
def mock_athena_client():
    """Mock Athena client for testing."""
    with mock_athena():
        client = boto3.client("athena", region_name="us-west-2")
        yield client


@pytest.fixture
def sample_s3_files():
    """Create sample S3 file paths for testing."""
    return [
        "s3://test-bucket/data/file1.csv",
        "s3://test-bucket/data/file2.csv",
        "s3://test-bucket/data/file3.csv",
    ]


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return dedent(
        """id,value,category
        1,10.5,A
        2,20.3,B
        3,15.7,A
        4,8.2,C
        5,12.1,B"""
    )


@pytest.fixture
def mock_spark_session():
    """Mock Spark session for testing."""
    with patch("pyspark.sql.SparkSession") as mock_spark:
        mock_session = MagicMock()
        mock_spark.builder.appName.return_value.master.return_value.getOrCreate.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials for testing."""
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_DEFAULT_REGION": "us-west-2",
        },
    ):
        yield


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "region": "us-west-2",
        "s3_bucket": "test-bucket",
        "sagemaker_role": "arn:aws:iam::123456789012:role/TestRole",
        "max_jobs": 4,
        "athena_output": "s3://test-bucket/athena-results",
    }


@pytest.fixture
def sample_transitions_data():
    """Sample transitions data for GAIL testing."""
    np.random.seed(42)
    n_samples = 100
    obs_dim = 10
    action_dim = 5

    return {
        "obs": np.random.randn(n_samples, obs_dim).astype(np.float32),
        "acts": np.random.randint(0, action_dim, n_samples).astype(np.float32),
        "next_obs": np.random.randn(n_samples, obs_dim).astype(np.float32),
        "dones": np.random.choice([True, False], n_samples),
        "rews": np.random.randn(n_samples).astype(np.float32),
        "infos": [{"step": i} for i in range(n_samples)],
    }


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    with patch("torch.device") as mock_device:
        mock_device.return_value = "cpu"
        yield mock_device


@pytest.fixture
def sample_bet_dictionary():
    """Sample bet dictionary for slot machine testing."""
    return {
        "USD": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        "EUR": [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        "CNY": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    }


@pytest.fixture
def sample_paylines():
    """Sample paylines for slot machine testing."""
    return [
        [0, 1, 2, 3, 4],  # Line 1: top row
        [5, 6, 7, 8, 9],  # Line 2: middle row
        [10, 11, 12, 13, 14],  # Line 3: bottom row
    ]


@pytest.fixture
def sample_payout_table():
    """Sample payout table for slot machine testing."""
    return {
        1: [100, 200, 500],
        2: [80, 150, 400],
        3: [60, 200, 300],
        4: [40, 100, 250],
        5: [30, 80, 200],
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "aws: marks tests that require AWS services")
    config.addinivalue_line("markers", "spark: marks tests that require Spark")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to tests that take longer
        if "integration" in item.nodeid or "aws" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add aws marker to AWS-related tests
        if "aws" in item.nodeid or "s3" in item.nodeid or "athena" in item.nodeid:
            item.add_marker(pytest.mark.aws)

        # Add spark marker to Spark-related tests
        if "spark" in item.nodeid or "pyspark" in item.nodeid:
            item.add_marker(pytest.mark.spark)

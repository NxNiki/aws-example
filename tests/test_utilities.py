"""
Test utilities and helper functions for the test suite.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def create_sample_dataframe(
        n_rows: int = 100,
        n_numeric_cols: int = 5,
        n_categorical_cols: int = 2,
        include_missing: bool = False,
        include_outliers: bool = False,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        np.random.seed(seed)

        data = {}

        # Add numeric columns
        for i in range(n_numeric_cols):
            col_name = f"numeric_{i}"
            values = np.random.normal(0, 1, n_rows)

            if include_outliers and i == 0:  # Add outliers to first numeric column
                outlier_indices = np.random.choice(n_rows, size=5, replace=False)
                values[outlier_indices] = np.random.normal(0, 10, 5)

            if include_missing and i == 1:  # Add missing values to second numeric column
                missing_indices = np.random.choice(n_rows, size=10, replace=False)
                values[missing_indices] = np.nan

            data[col_name] = values

        # Add categorical columns
        for i in range(n_categorical_cols):
            col_name = f"categorical_{i}"
            categories = [f"cat_{j}" for j in range(3 + i)]
            data[col_name] = np.random.choice(categories, n_rows)

        # Add datetime column
        data["timestamp"] = pd.date_range("2024-01-01", periods=n_rows, freq="H")

        # Add target column for ML testing
        data["target"] = np.random.choice([0, 1], n_rows)

        return pd.DataFrame(data)

    @staticmethod
    def create_time_series_data(
        n_points: int = 1000,
        n_series: int = 3,
        include_trend: bool = True,
        include_seasonality: bool = True,
        include_noise: bool = True,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create time series data for testing."""
        np.random.seed(seed)

        dates = pd.date_range("2024-01-01", periods=n_points, freq="H")

        data = {"timestamp": dates}

        for i in range(n_series):
            series_name = f"series_{i}"
            values = np.zeros(n_points)

            # Add trend
            if include_trend:
                trend = np.linspace(0, i + 1, n_points)
                values += trend

            # Add seasonality
            if include_seasonality:
                seasonal = np.sin(2 * np.pi * np.arange(n_points) / 24) * (i + 1)
                values += seasonal

            # Add noise
            if include_noise:
                noise = np.random.normal(0, 0.5, n_points)
                values += noise

            data[series_name] = values

        return pd.DataFrame(data)

    @staticmethod
    def create_correlated_data(
        n_rows: int = 100, n_features: int = 10, correlation_strength: float = 0.8, seed: int = 42
    ) -> pd.DataFrame:
        """Create correlated data for testing correlation analysis."""
        np.random.seed(seed)

        # Create base features
        base_features = np.random.normal(0, 1, (n_rows, n_features))

        # Create correlated features
        correlated_features = []
        for i in range(n_features):
            if i == 0:
                # First feature is independent
                correlated_features.append(base_features[:, i])
            else:
                # Other features are correlated with the first one
                correlated = (
                    correlation_strength * base_features[:, 0] + (1 - correlation_strength) * base_features[:, i]
                )
                correlated_features.append(correlated)

        # Create DataFrame
        data = {f"feature_{i}": correlated_features[i] for i in range(n_features)}

        return pd.DataFrame(data)


class MockAWSClient:
    """Mock AWS client for testing."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._calls: List[Dict] = []
        self._responses: Dict[str, List[Any]] = {}

    def add_response(self, method: str, response: Any):
        """Add a mock response for a method."""
        if method not in self._responses:
            self._responses[method] = []
        self._responses[method].append(response)

    def get_calls(self, method: str) -> List[Dict]:
        """Get all calls made to a method."""
        return [call for call in self._calls if call["method"] == method]

    def reset(self):
        """Reset the mock client."""
        self._calls = []
        self._responses = {}


class TestFileManager:
    """Utility class for managing test files."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.created_files: List[Union[str, Path]] = []

    def create_csv_file(self, filename: str, data: pd.DataFrame, **kwargs) -> Path:
        """Create a CSV file with given data."""
        file_path = self.temp_dir / filename
        data.to_csv(file_path, index=False, **kwargs)
        self.created_files.append(file_path)
        return file_path

    def create_json_file(self, filename: str, data: Dict, **kwargs) -> Path:
        """Create a JSON file with given data."""
        import json

        file_path = self.temp_dir / filename
        with open(file_path, "w") as f:
            json.dump(data, f, **kwargs)
        self.created_files.append(file_path)
        return file_path

    def create_text_file(self, filename: str, content: str) -> Path:
        """Create a text file with given content."""
        file_path = self.temp_dir / filename
        file_path.write_text(content)
        self.created_files.append(file_path)
        return file_path

    def cleanup(self):
        """Clean up all created files."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()

        if self.temp_dir.exists():
            self.temp_dir.rmdir()


class PerformanceTimer:
    """Utility class for measuring test performance."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start timing."""
        import time

        self.start_time = time.time()

    def stop(self):
        """Stop timing."""
        import time

        self.end_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer not started or stopped")
        return self.end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class TestAssertions:
    """Custom assertions for testing."""

    @staticmethod
    def assert_dataframe_equals_approximately(
        df1: pd.DataFrame, df2: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8
    ):
        """Assert that two DataFrames are approximately equal."""
        assert df1.shape == df2.shape, f"Shapes differ: {df1.shape} vs {df2.shape}"
        assert list(df1.columns) == list(df2.columns), "Columns differ"

        for col in df1.columns:
            if df1[col].dtype in [np.float64, np.float32]:
                np.testing.assert_allclose(
                    df1[col].values, df2[col].values, rtol=rtol, atol=atol, err_msg=f"Column {col} differs"
                )
            else:
                pd.testing.assert_series_equal(df1[col], df2[col], check_dtype=False, err_msg=f"Column {col} differs")

    @staticmethod
    def assert_dataframe_contains_columns(df: pd.DataFrame, required_columns: List[str]):
        """Assert that DataFrame contains all required columns."""
        missing_columns = set(required_columns) - set(df.columns)
        assert not missing_columns, f"Missing columns: {missing_columns}"

    @staticmethod
    def assert_dataframe_has_no_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None):
        """Assert that DataFrame has no missing values in specified columns."""
        if columns is None:
            columns = df.columns

        for col in columns:
            missing_count = df[col].isnull().sum()
            assert missing_count == 0, f"Column {col} has {missing_count} missing values"

    @staticmethod
    def assert_dataframe_has_expected_dtypes(df: pd.DataFrame, expected_dtypes: Dict[str, Any]):
        """Assert that DataFrame has expected data types."""
        for col, expected_dtype in expected_dtypes.items():
            assert col in df.columns, f"Column {col} not found"
            actual_dtype = df[col].dtype
            assert actual_dtype == expected_dtype, f"Column {col} has dtype {actual_dtype}, expected {expected_dtype}"


# Pytest fixtures for test utilities
@pytest.fixture
def test_data_generator():
    """Fixture for test data generator."""
    return TestDataGenerator()


@pytest.fixture
def test_file_manager(temp_dir):
    """Fixture for test file manager."""
    return TestFileManager(temp_dir)


@pytest.fixture
def performance_timer():
    """Fixture for performance timer."""
    return PerformanceTimer()


@pytest.fixture
def mock_aws_s3_client():
    """Fixture for mock S3 client."""
    return MockAWSClient("s3")


@pytest.fixture
def mock_aws_athena_client():
    """Fixture for mock Athena client."""
    return MockAWSClient("athena")


# Utility functions for common test patterns
def skip_if_no_aws_credentials():
    """Skip test if AWS credentials are not available."""
    try:
        import boto3

        boto3.client("s3").list_buckets()
    except Exception:
        pytest.skip("AWS credentials not available")


def skip_if_no_spark():
    """Skip test if Spark is not available."""
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("test").getOrCreate()
        spark.stop()
    except Exception:
        pytest.skip("Spark not available")


def create_mock_spark_dataframe(data: Dict) -> MagicMock:
    """Create a mock Spark DataFrame."""
    mock_df = MagicMock()
    mock_df.collect.return_value = [MagicMock() for _ in range(len(list(data.values())[0]))]
    mock_df.count.return_value = len(list(data.values())[0])
    mock_df.columns = list(data.keys())
    return mock_df


def assert_logs_contain(logs, expected_messages):
    """Assert that logs contain expected messages."""
    log_text = "\n".join(str(record) for record in logs)
    for message in expected_messages:
        assert message in log_text, f"Expected message '{message}' not found in logs"

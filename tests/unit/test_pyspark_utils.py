"""
Unit tests for the pyspark_utils module.
"""

import re
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bituslabs_ds.pyspark_utils import (
    create_stat_aggregations,
    display_df_rows,
    encode_label,
    read_data_with_partition,
    read_files_to_spark,
)


class TestReadDataWithPartition:
    """Test the read_data_with_partition function."""

    def test_read_data_with_partition_success(self, mock_spark_session):
        """Test successful data reading with partition extraction."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_spark_session.read.format.return_value.options.return_value.load.return_value = mock_df

        # Mock withColumn and drop methods
        mock_df.withColumn.return_value = mock_df
        mock_df.drop.return_value = mock_df

        # Test data
        path_pattern = "s3://test-bucket/data/*/*.csv"
        regex_pattern = r"(?P<year>\d{4})/(?P<month>\d{2})"

        result = read_data_with_partition(
            mock_spark_session, path_pattern, regex_pattern, format="csv", read_opts={"header": "true"}
        )

        # Verify calls
        mock_spark_session.read.format.assert_called_once_with("csv")
        mock_spark_session.read.format.return_value.options.assert_called_once_with(header="true")
        mock_spark_session.read.format.return_value.options.return_value.load.assert_called_once_with(path_pattern)

        # Verify withColumn was called for each partition column
        assert mock_df.withColumn.call_count >= 2  # year and month columns
        mock_df.drop.assert_called_once_with("_filepath")

    def test_read_data_with_partition_parquet_format(self, mock_spark_session):
        """Test reading parquet format with partition extraction."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_spark_session.read.format.return_value.options.return_value.load.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        mock_df.drop.return_value = mock_df

        # Test data
        path_pattern = "s3://test-bucket/data/*/*.parquet"
        regex_pattern = r"(?P<year>\d{4})"

        result = read_data_with_partition(mock_spark_session, path_pattern, regex_pattern, format="parquet")

        # Verify format was set to parquet
        mock_spark_session.read.format.assert_called_once_with("parquet")

    def test_read_data_with_partition_no_read_opts(self, mock_spark_session):
        """Test reading data without read options."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_spark_session.read.format.return_value.options.return_value.load.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        mock_df.drop.return_value = mock_df

        # Test data
        path_pattern = "s3://test-bucket/data/*/*.csv"
        regex_pattern = r"(?P<year>\d{4})"

        result = read_data_with_partition(mock_spark_session, path_pattern, regex_pattern)

        # Verify options was called with empty dict
        mock_spark_session.read.format.return_value.options.assert_called_once_with()


class TestReadFilesToSpark:
    """Test the read_files_to_spark function."""

    def test_read_files_to_spark_basic(self, mock_spark_session):
        """Test basic file reading to Spark."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_spark_session.read.option.return_value.csv.return_value = mock_df
        mock_df.withColumnRenamed.return_value = mock_df
        mock_df.select.return_value = mock_df

        # Test data
        s3_files = ["s3://test-bucket/file1.csv", "s3://test-bucket/file2.csv"]
        column_names = ["col1", "col2", "col3"]
        keep_columns = ["col1", "col2"]

        result = read_files_to_spark(mock_spark_session, s3_files, column_names, keep_columns)

        # Verify calls
        mock_spark_session.read.option.assert_called_once_with("header", "false")
        mock_spark_session.read.option.return_value.csv.assert_called_once_with(s3_files)

        # Verify column renaming
        assert mock_df.withColumnRenamed.call_count == 3  # for each column name

        # Verify column selection
        mock_df.select.assert_called_once_with(*keep_columns)

    def test_read_files_to_spark_no_column_names(self, mock_spark_session):
        """Test reading files without column names."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_spark_session.read.option.return_value.csv.return_value = mock_df
        mock_df.select.return_value = mock_df

        # Test data
        s3_files = ["s3://test-bucket/file1.csv"]
        keep_columns = ["_c0", "_c1"]

        result = read_files_to_spark(mock_spark_session, s3_files, column_names=None, keep_columns=keep_columns)

        # Verify no column renaming occurred
        mock_df.withColumnRenamed.assert_not_called()

        # Verify column selection
        mock_df.select.assert_called_once_with(*keep_columns)

    def test_read_files_to_spark_no_keep_columns(self, mock_spark_session):
        """Test reading files without keep_columns."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_spark_session.read.option.return_value.csv.return_value = mock_df
        mock_df.withColumnRenamed.return_value = mock_df
        mock_df.select.return_value = mock_df

        # Test data
        s3_files = ["s3://test-bucket/file1.csv"]
        column_names = ["col1", "col2"]

        result = read_files_to_spark(mock_spark_session, s3_files, column_names, keep_columns=None)

        # Verify column renaming occurred
        assert mock_df.withColumnRenamed.call_count == 2

        # Verify no column selection
        mock_df.select.assert_not_called()


class TestDisplayDfRows:
    """Test the display_df_rows function."""

    def test_display_df_rows_basic(self, mock_spark_session):
        """Test basic DataFrame row display."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_df.take.return_value = [MagicMock(), MagicMock(), MagicMock()]  # Mock row objects

        display_df_rows(mock_df, "Test message")

        # Verify take was called with default n_rows=5
        mock_df.take.assert_called_once_with(5)

    def test_display_df_rows_custom_n_rows(self, mock_spark_session):
        """Test DataFrame row display with custom n_rows."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_df.take.return_value = [MagicMock(), MagicMock()]

        display_df_rows(mock_df, "Test message", n_rows=2)

        # Verify take was called with custom n_rows
        mock_df.take.assert_called_once_with(2)

    def test_display_df_rows_empty_dataframe(self, mock_spark_session):
        """Test DataFrame row display with empty DataFrame."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_df.take.return_value = []

        # Should not raise an exception
        display_df_rows(mock_df, "Test message")

        mock_df.take.assert_called_once_with(5)


class TestEncodeLabel:
    """Test the encode_label function."""

    def test_encode_label_success(self, mock_spark_session):
        """Test successful label encoding."""
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_df.select.return_value = mock_df

        # Mock StringIndexer
        with patch("bituslabs_ds.pyspark_utils.StringIndexer") as mock_indexer:
            mock_indexer_instance = MagicMock()
            mock_indexer.return_value = mock_indexer_instance

            # Mock fit and transform
            mock_indexer_instance.fit.return_value = mock_indexer_instance
            mock_indexer_instance.transform.return_value = mock_df

            result = encode_label(mock_df, "category", "category_index")

            # Verify StringIndexer was created correctly
            mock_indexer.assert_called_once_with(inputCol="category", outputCol="category_index")

            # Verify fit and transform were called
            mock_indexer_instance.fit.assert_called_once_with(mock_df)
            mock_indexer_instance.transform.assert_called_once()

            assert result == mock_df

    def test_encode_label_different_columns(self, mock_spark_session):
        """Test label encoding with different input/output columns."""
        # Mock Spark DataFrame
        mock_df = MagicMock()

        # Mock StringIndexer
        with patch("bituslabs_ds.pyspark_utils.StringIndexer") as mock_indexer:
            mock_indexer_instance = MagicMock()
            mock_indexer.return_value = mock_indexer_instance
            mock_indexer_instance.fit.return_value = mock_indexer_instance
            mock_indexer_instance.transform.return_value = mock_df

            result = encode_label(mock_df, "input_col", "output_col")

            # Verify StringIndexer was created with correct columns
            mock_indexer.assert_called_once_with(inputCol="input_col", outputCol="output_col")


class TestCreateStatAggregations:
    """Test the create_stat_aggregations function."""

    def test_create_stat_aggregations_basic(self):
        """Test basic statistical aggregations creation."""
        column_name = "value"

        result = create_stat_aggregations(column_name)

        assert len(result) == 6  # min, max, mean, p25, median, p75

        # Check that all aggregations are Column objects
        for agg in result:
            assert hasattr(agg, "alias")

        # Check column names
        expected_names = ["value_min", "value_max", "value_mean", "value_p25", "value_median", "value_p75"]
        actual_names = [agg.alias() for agg in result]
        assert actual_names == expected_names

    def test_create_stat_aggregations_with_rename(self):
        """Test statistical aggregations with custom rename."""
        column_name = "value"
        rename = "custom_name"

        result = create_stat_aggregations(column_name, rename)

        assert len(result) == 6

        # Check column names with custom rename
        expected_names = [
            "custom_name_min",
            "custom_name_max",
            "custom_name_mean",
            "custom_name_p25",
            "custom_name_median",
            "custom_name_p75",
        ]
        actual_names = [agg.alias() for agg in result]
        assert actual_names == expected_names

    def test_create_stat_aggregations_no_rename(self):
        """Test statistical aggregations without rename (should use original name)."""
        column_name = "test_column"

        result = create_stat_aggregations(column_name, rename=None)

        assert len(result) == 6

        # Check that original column name is used
        expected_names = [
            "test_column_min",
            "test_column_max",
            "test_column_mean",
            "test_column_p25",
            "test_column_median",
            "test_column_p75",
        ]
        actual_names = [agg.alias() for agg in result]
        assert actual_names == expected_names

    def test_create_stat_aggregations_empty_rename(self):
        """Test statistical aggregations with empty rename string."""
        column_name = "value"
        rename = ""

        result = create_stat_aggregations(column_name, rename)

        assert len(result) == 6

        # Check that empty rename is used
        expected_names = ["_min", "_max", "_mean", "_p25", "_median", "_p75"]
        actual_names = [agg.alias() for agg in result]
        assert actual_names == expected_names

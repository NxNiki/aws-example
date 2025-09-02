"""
Unit tests for the athena_utils module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pandas as pd
import pytest
from moto import mock_athena

from bituslabs_ds.athena_utils import (
    check_query_status,
    execute_query,
    get_query_result,
    submit_query,
    wait_query_finish,
)


class TestSubmitQuery:
    """Test the submit_query function."""

    @mock_athena
    def test_submit_query_success(self):
        """Test successful query submission."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the response
        with patch.object(athena_client, "start_query_execution") as mock_start:
            mock_start.return_value = {"QueryExecutionId": "test-query-id"}

            result = submit_query("SELECT * FROM test_table", "test_database")

            assert result == "test-query-id"
            mock_start.assert_called_once()

    @mock_athena
    def test_submit_query_with_custom_output(self):
        """Test query submission with custom output location."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the response
        with patch.object(athena_client, "start_query_execution") as mock_start:
            mock_start.return_value = {"QueryExecutionId": "test-query-id"}

            result = submit_query("SELECT * FROM test_table", "test_database")

            # Verify the call was made with correct parameters
            call_args = mock_start.call_args
            assert call_args[1]["QueryString"] == "SELECT * FROM test_table"
            assert call_args[1]["QueryExecutionContext"]["Database"] == "test_database"


class TestCheckQueryStatus:
    """Test the check_query_status function."""

    @mock_athena
    def test_check_query_status_success(self):
        """Test checking query status."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the response
        with patch.object(athena_client, "get_query_execution") as mock_get:
            mock_get.return_value = {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

            result = check_query_status("test-query-id")

            assert result == "SUCCEEDED"
            mock_get.assert_called_once_with(QueryExecutionId="test-query-id")

    @mock_athena
    def test_check_query_status_failed(self):
        """Test checking failed query status."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the response
        with patch.object(athena_client, "get_query_execution") as mock_get:
            mock_get.return_value = {"QueryExecution": {"Status": {"State": "FAILED"}}}

            result = check_query_status("test-query-id")

            assert result == "FAILED"


class TestWaitQueryFinish:
    """Test the wait_query_finish function."""

    @mock_athena
    def test_wait_query_finish_success(self):
        """Test waiting for successful query completion."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the response sequence
        with patch.object(athena_client, "get_query_execution") as mock_get:
            mock_get.side_effect = [
                {"QueryExecution": {"Status": {"State": "RUNNING"}}},
                {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}},
            ]

            # Should not raise an exception
            wait_query_finish("test-query-id", interval=0.1)

            assert mock_get.call_count == 2

    @mock_athena
    def test_wait_query_finish_failed(self):
        """Test waiting for failed query completion."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the response
        with patch.object(athena_client, "get_query_execution") as mock_get:
            mock_get.return_value = {"QueryExecution": {"Status": {"State": "FAILED"}}}

            # Should raise an exception
            with pytest.raises(Exception, match="Query failed with status: FAILED"):
                wait_query_finish("test-query-id")

    @mock_athena
    def test_wait_query_finish_cancelled(self):
        """Test waiting for cancelled query completion."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the response
        with patch.object(athena_client, "get_query_execution") as mock_get:
            mock_get.return_value = {"QueryExecution": {"Status": {"State": "CANCELLED"}}}

            # Should raise an exception
            with pytest.raises(Exception, match="Query failed with status: CANCELLED"):
                wait_query_finish("test-query-id")


class TestGetQueryResult:
    """Test the get_query_result function."""

    @mock_athena
    def test_get_query_result_with_s3_path(self):
        """Test getting query result from S3 path."""
        # Mock the wait_query_finish function
        with patch("bituslabs_ds.athena_utils.wait_query_finish"):
            # Mock the read_to_pandas_df function
            with patch("bituslabs_ds.athena_utils.read_to_pandas_df") as mock_read:
                mock_read.return_value = pd.DataFrame({"col1": [1, 2, 3]})

                result = get_query_result("test-query-id", "s3://test-bucket/results/")

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 3
                mock_read.assert_called_once()

    @mock_athena
    def test_get_query_result_without_s3_path(self):
        """Test getting query result without S3 path."""
        # Mock the wait_query_finish function
        with patch("bituslabs_ds.athena_utils.wait_query_finish"):
            # Mock the Athena client
            athena_client = boto3.client("athena", region_name="us-west-2")

            with patch.object(athena_client, "get_paginator") as mock_paginator:
                # Mock paginator response
                mock_paginator.return_value.paginate.return_value = [
                    {
                        "ResultSet": {
                            "Rows": [
                                {"Data": [{"VarCharValue": "col1"}, {"VarCharValue": "col2"}]},
                                {"Data": [{"VarCharValue": "1"}, {"VarCharValue": "2"}]},
                                {"Data": [{"VarCharValue": "3"}, {"VarCharValue": "4"}]},
                            ]
                        }
                    }
                ]

                result = get_query_result("test-query-id")

                assert isinstance(result, pd.DataFrame)
                assert list(result.columns) == ["col1", "col2"]
                assert len(result) == 2


class TestExecuteQuery:
    """Test the execute_query function."""

    @mock_athena
    def test_execute_query_with_local_cache(self, temp_dir):
        """Test executing query with local cache."""
        cache_file = temp_dir / "cache.csv"
        cache_file.write_text("col1,col2\n1,2\n3,4")

        result = execute_query("SELECT * FROM test_table", "test_database", local_cache=str(cache_file))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2"]

    @mock_athena
    def test_execute_query_without_return_result(self):
        """Test executing query without returning result."""
        with patch("bituslabs_ds.athena_utils.submit_query") as mock_submit:
            mock_submit.return_value = "test-query-id"

            result = execute_query("SELECT * FROM test_table", "test_database", return_result=False)

            assert result is None
            mock_submit.assert_called_once()

    @mock_athena
    def test_execute_query_with_s3_output(self):
        """Test executing query with S3 output."""
        with patch("bituslabs_ds.athena_utils.submit_query") as mock_submit:
            with patch("bituslabs_ds.athena_utils.get_query_result") as mock_get_result:
                with patch("bituslabs_ds.athena_utils.s3") as mock_s3:
                    mock_submit.return_value = "test-query-id"
                    mock_get_result.return_value = pd.DataFrame({"col1": [1, 2, 3]})

                    result = execute_query(
                        "SELECT * FROM test_table",
                        "test_database",
                        s3_file_path="s3://test-bucket/custom-output.csv",
                        return_result=True,
                    )

                    assert isinstance(result, pd.DataFrame)
                    # Verify S3 operations were called
                    assert mock_s3.copy_object.called
                    assert mock_s3.delete_object.called

    @mock_athena
    def test_execute_query_overwrite_cache(self, temp_dir):
        """Test executing query with cache overwrite."""
        cache_file = temp_dir / "cache.csv"
        cache_file.write_text("old,data\n1,2")

        with patch("bituslabs_ds.athena_utils.submit_query") as mock_submit:
            with patch("bituslabs_ds.athena_utils.get_query_result") as mock_get_result:
                mock_submit.return_value = "test-query-id"
                mock_get_result.return_value = pd.DataFrame({"new": [3, 4], "data": [5, 6]})

                result = execute_query(
                    "SELECT * FROM test_table",
                    "test_database",
                    local_cache=str(cache_file),
                    overwrite_cache=True,
                    return_result=True,
                )

                assert isinstance(result, pd.DataFrame)
                assert list(result.columns) == ["new", "data"]

                # Verify cache was overwritten
                cached_data = pd.read_csv(cache_file)
                assert list(cached_data.columns) == ["new", "data"]

    @mock_athena
    def test_execute_query_different_buckets_error(self):
        """Test error when copying across different buckets."""
        with patch("bituslabs_ds.athena_utils.submit_query") as mock_submit:
            with patch("bituslabs_ds.athena_utils.get_query_result") as mock_get_result:
                mock_submit.return_value = "test-query-id"
                mock_get_result.return_value = pd.DataFrame({"col1": [1, 2, 3]})

                with pytest.raises(Exception, match="cannot copy output across buckets"):
                    execute_query(
                        "SELECT * FROM test_table",
                        "test_database",
                        s3_file_path="s3://different-bucket/output.csv",
                        return_result=True,
                    )

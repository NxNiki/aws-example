"""
Integration tests for AWS services.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import boto3
import pandas as pd
import pytest
from moto import mock_athena, mock_s3

from bituslabs_ds.athena_utils import execute_query, get_query_result, submit_query
from bituslabs_ds.s3_utils import list_s3_files, read_files, upload_file_to_s3, write_pandas_to_s3


@pytest.mark.integration
@pytest.mark.aws
class TestS3Integration:
    """Integration tests for S3 operations."""

    @mock_s3
    def test_s3_read_write_cycle(self, sample_dataframe):
        """Test complete S3 read-write cycle."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Write DataFrame to S3
        write_pandas_to_s3(sample_dataframe, "test-bucket", "data/test.csv")

        # Read DataFrame from S3
        result = read_files("s3://test-bucket/data/test.csv")

        # Verify data integrity
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)

        # Check that data matches (allowing for type differences)
        pd.testing.assert_frame_equal(result, sample_dataframe, check_dtype=False)

    @mock_s3
    def test_s3_multiple_files_upload_download(self, temp_dir):
        """Test uploading and downloading multiple files."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Create test files
        test_files = []
        for i in range(3):
            test_file = temp_dir / f"test_{i}.csv"
            test_file.write_text(f"id,value\n{i},{i*10}")
            test_files.append(test_file)

        # Upload files
        uploaded_uris = []
        for i, test_file in enumerate(test_files):
            uri = upload_file_to_s3(test_file, "test-bucket", f"data/test_{i}.csv")
            uploaded_uris.append(uri)

        # List files
        files = list_s3_files("test-bucket", "data/")

        # Verify uploads
        assert len(files) == 3
        assert all("test_" in file for file in files)

        # Download and verify content
        for i, file_uri in enumerate(files):
            result = read_files(file_uri)
            assert len(result) == 1
            assert result["id"].iloc[0] == i
            assert result["value"].iloc[0] == i * 10

    @mock_s3
    def test_s3_large_dataframe_handling(self):
        """Test handling of large DataFrames."""
        # Create large DataFrame
        large_df = pd.DataFrame(
            {
                "id": range(10000),
                "value": [f"value_{i}" for i in range(10000)],
                "category": [f"cat_{i % 10}" for i in range(10000)],
            }
        )

        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Write large DataFrame
        write_pandas_to_s3(large_df, "test-bucket", "data/large.csv")

        # Read large DataFrame
        result = read_files("s3://test-bucket/data/large.csv")

        # Verify data integrity
        assert len(result) == 10000
        assert list(result.columns) == ["id", "value", "category"]
        assert result["id"].iloc[0] == 0
        assert result["id"].iloc[-1] == 9999

    @mock_s3
    def test_s3_parallel_file_processing(self, temp_dir):
        """Test parallel processing of multiple files."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Create multiple test files
        file_uris = []
        for i in range(5):
            test_file = temp_dir / f"parallel_test_{i}.csv"
            test_file.write_text(f"id,value,category\n{i},{i*10},cat_{i}")

            # Upload file
            uri = upload_file_to_s3(test_file, "test-bucket", f"data/parallel_test_{i}.csv")
            file_uris.append(uri)

        # Read all files in parallel
        result = read_files(file_uris, parallel_mode="thread", max_workers=3)

        # Verify results
        assert len(result) == 5
        assert "id" in result.columns
        assert "value" in result.columns
        assert "category" in result.columns

        # Check that all data is present
        assert set(result["id"]) == {0, 1, 2, 3, 4}
        assert set(result["value"]) == {0, 10, 20, 30, 40}


@pytest.mark.integration
@pytest.mark.aws
class TestAthenaIntegration:
    """Integration tests for Athena operations."""

    @mock_athena
    def test_athena_query_execution_cycle(self):
        """Test complete Athena query execution cycle."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Mock the query execution flow
        with patch.object(athena_client, "start_query_execution") as mock_start:
            with patch.object(athena_client, "get_query_execution") as mock_get:
                with patch.object(athena_client, "get_paginator") as mock_paginator:
                    # Mock query submission
                    mock_start.return_value = {"QueryExecutionId": "test-query-id"}

                    # Mock query status checks
                    mock_get.side_effect = [
                        {"QueryExecution": {"Status": {"State": "RUNNING"}}},
                        {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}},
                    ]

                    # Mock query results
                    mock_paginator.return_value.paginate.return_value = [
                        {
                            "ResultSet": {
                                "Rows": [
                                    {"Data": [{"VarCharValue": "id"}, {"VarCharValue": "name"}]},
                                    {"Data": [{"VarCharValue": "1"}, {"VarCharValue": "Alice"}]},
                                    {"Data": [{"VarCharValue": "2"}, {"VarCharValue": "Bob"}]},
                                ]
                            }
                        }
                    ]

                    # Execute query
                    result = execute_query("SELECT id, name FROM test_table", "test_database", return_result=True)

                    # Verify results
                    assert isinstance(result, pd.DataFrame)
                    assert len(result) == 2
                    assert list(result.columns) == ["id", "name"]
                    assert result["id"].iloc[0] == "1"
                    assert result["name"].iloc[0] == "Alice"

    @mock_athena
    def test_athena_query_with_s3_output(self):
        """Test Athena query with S3 output."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        with patch.object(athena_client, "start_query_execution") as mock_start:
            with patch.object(athena_client, "get_query_execution") as mock_get:
                with patch("bituslabs_ds.athena_utils.read_to_pandas_df") as mock_read:
                    # Mock query submission
                    mock_start.return_value = {"QueryExecutionId": "test-query-id"}

                    # Mock query completion
                    mock_get.return_value = {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

                    # Mock S3 read
                    mock_read.return_value = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

                    # Execute query with S3 output
                    result = execute_query(
                        "SELECT id, value FROM test_table",
                        "test_database",
                        s3_file_path="s3://test-bucket/custom-output.csv",
                        return_result=True,
                    )

                    # Verify results
                    assert isinstance(result, pd.DataFrame)
                    assert len(result) == 3
                    assert list(result.columns) == ["id", "value"]

    @mock_athena
    def test_athena_query_failure_handling(self):
        """Test Athena query failure handling."""
        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        with patch.object(athena_client, "start_query_execution") as mock_start:
            with patch.object(athena_client, "get_query_execution") as mock_get:
                # Mock query submission
                mock_start.return_value = {"QueryExecutionId": "test-query-id"}

                # Mock query failure
                mock_get.return_value = {"QueryExecution": {"Status": {"State": "FAILED"}}}

                # Execute query and expect failure
                with pytest.raises(Exception, match="Query failed with status: FAILED"):
                    execute_query("INVALID SQL QUERY", "test_database", return_result=True)


@pytest.mark.integration
@pytest.mark.aws
class TestAWSDataPipeline:
    """Integration tests for complete AWS data pipeline."""

    @mock_s3
    @mock_athena
    def test_complete_data_pipeline(self, sample_dataframe):
        """Test complete data pipeline from S3 to Athena and back."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Setup Athena
        athena_client = boto3.client("athena", region_name="us-west-2")

        # Step 1: Upload data to S3
        write_pandas_to_s3(sample_dataframe, "test-bucket", "input/data.csv")

        # Step 2: Verify data in S3
        files = list_s3_files("test-bucket", "input/")
        assert len(files) == 1
        assert "data.csv" in files[0]

        # Step 3: Read data from S3
        downloaded_data = read_files(files[0])
        assert len(downloaded_data) == len(sample_dataframe)

        # Step 4: Process data and upload results
        processed_data = downloaded_data.copy()
        processed_data["processed_value"] = processed_data["value"] * 2

        write_pandas_to_s3(processed_data, "test-bucket", "output/processed_data.csv")

        # Step 5: Verify processed data
        output_files = list_s3_files("test-bucket", "output/")
        assert len(output_files) == 1

        final_data = read_files(output_files[0])
        assert "processed_value" in final_data.columns
        assert len(final_data) == len(sample_dataframe)

    @mock_s3
    def test_data_validation_pipeline(self, sample_dataframe):
        """Test data validation pipeline."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload original data
        write_pandas_to_s3(sample_dataframe, "test-bucket", "data/original.csv")

        # Create corrupted data (missing some rows)
        corrupted_data = sample_dataframe.iloc[:-10].copy()
        write_pandas_to_s3(corrupted_data, "test-bucket", "data/corrupted.csv")

        # Read both datasets
        original = read_files("s3://test-bucket/data/original.csv")
        corrupted = read_files("s3://test-bucket/data/corrupted.csv")

        # Validate data integrity
        assert len(original) > len(corrupted)
        assert len(original) - len(corrupted) == 10

        # Check that columns match
        assert list(original.columns) == list(corrupted.columns)

        # Check that data types are preserved
        for col in original.columns:
            if original[col].dtype != corrupted[col].dtype:
                # Allow for minor type differences due to CSV serialization
                assert str(original[col].dtype).split(".")[0] == str(corrupted[col].dtype).split(".")[0]

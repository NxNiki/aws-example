"""
Unit tests for the s3_utils module.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import boto3
import pandas as pd
import pytest
from moto import mock_s3

from bituslabs_ds.s3_utils import (
    list_s3_files,
    parse_bucket_name,
    parse_s3_path,
    read_dataset,
    read_files,
    read_to_pandas_df,
    upload_file_to_s3,
    upload_folder_to_s3,
    write_df_to_s3,
    write_pandas_to_s3,
    write_spark_to_s3,
)


class TestParseBucketName:
    """Test the parse_bucket_name function."""

    def test_parse_bucket_name_basic(self):
        """Test basic bucket name parsing."""
        result = parse_bucket_name("my-bucket")
        assert result == "my-bucket"

    def test_parse_bucket_name_with_slash(self):
        """Test bucket name with trailing slash."""
        result = parse_bucket_name("my-bucket/")
        assert result == "my-bucket"

    def test_parse_bucket_name_s3_prefix(self):
        """Test bucket name with s3:// prefix."""
        result = parse_bucket_name("s3://my-bucket")
        assert result == "my-bucket"

    def test_parse_bucket_name_s3a_prefix(self):
        """Test bucket name with s3a:// prefix."""
        result = parse_bucket_name("s3a://my-bucket")
        assert result == "my-bucket"

    def test_parse_bucket_name_s3n_prefix(self):
        """Test bucket name with s3n:// prefix."""
        result = parse_bucket_name("s3n://my-bucket")
        assert result == "my-bucket"


class TestParseS3Path:
    """Test the parse_s3_path function."""

    def test_parse_s3_path_basic(self):
        """Test basic S3 path parsing."""
        bucket, key = parse_s3_path("s3://my-bucket/path/to/file.csv")
        assert bucket == "my-bucket"
        assert key == "path/to/file.csv"

    def test_parse_s3_path_s3a_scheme(self):
        """Test parsing s3a:// scheme."""
        bucket, key = parse_s3_path("s3a://my-bucket/path/to/file.csv")
        assert bucket == "my-bucket"
        assert key == "path/to/file.csv"

    def test_parse_s3_path_s3n_scheme(self):
        """Test parsing s3n:// scheme."""
        bucket, key = parse_s3_path("s3n://my-bucket/path/to/file.csv")
        assert bucket == "my-bucket"
        assert key == "path/to/file.csv"

    def test_parse_s3_path_root_key(self):
        """Test parsing S3 path with root key."""
        bucket, key = parse_s3_path("s3://my-bucket/file.csv")
        assert bucket == "my-bucket"
        assert key == "file.csv"

    def test_parse_s3_path_invalid_scheme(self):
        """Test parsing invalid scheme."""
        with pytest.raises(ValueError, match="Unsupported S3 URI scheme"):
            parse_s3_path("https://my-bucket/file.csv")

    def test_parse_s3_path_missing_bucket(self):
        """Test parsing S3 path with missing bucket."""
        with pytest.raises(ValueError, match="Missing bucket in S3 URI"):
            parse_s3_path("s3:///path/to/file.csv")


class TestReadToPandasDf:
    """Test the read_to_pandas_df function."""

    @mock_s3
    def test_read_to_pandas_df_success(self, sample_csv_content):
        """Test successful reading of CSV from S3."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload CSV content
        s3_client.put_object(Bucket="test-bucket", Key="data/test.csv", Body=sample_csv_content)

        # Read the data
        result = read_to_pandas_df("test-bucket", "data/test.csv")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "id" in result.columns
        assert "value" in result.columns

    @mock_s3
    def test_read_to_pandas_df_with_columns(self, sample_csv_content):
        """Test reading CSV with specific columns."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload CSV content
        s3_client.put_object(Bucket="test-bucket", Key="data/test.csv", Body=sample_csv_content)

        # Read specific columns
        result = read_to_pandas_df("test-bucket", "data/test.csv", columns=["id", "value"])

        assert list(result.columns) == ["id", "value"]
        assert len(result) == 5


class TestWritePandasToS3:
    """Test the write_pandas_to_s3 function."""

    @mock_s3
    def test_write_pandas_to_s3_success(self, sample_dataframe):
        """Test successful writing of DataFrame to S3."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Write DataFrame
        write_pandas_to_s3(sample_dataframe, "test-bucket", "data/output.csv")

        # Verify the file was uploaded
        response = s3_client.get_object(Bucket="test-bucket", Key="data/output.csv")
        content = response["Body"].read().decode("utf-8")

        # Check that content contains CSV data
        assert "id" in content
        assert "value" in content


class TestWriteDfToS3:
    """Test the write_df_to_s3 function."""

    @mock_s3
    def test_write_pandas_dataframe(self, sample_dataframe):
        """Test writing pandas DataFrame."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Write DataFrame
        write_df_to_s3(sample_dataframe, "test-bucket", "data/output.csv")

        # Verify the file was uploaded
        response = s3_client.get_object(Bucket="test-bucket", Key="data/output.csv")
        assert response is not None

    def test_write_unsupported_dataframe_type(self):
        """Test writing unsupported DataFrame type."""
        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            write_df_to_s3("not_a_dataframe", "test-bucket", "data/output.csv")


class TestUploadFileToS3:
    """Test the upload_file_to_s3 function."""

    @mock_s3
    def test_upload_file_to_s3_success(self, temp_dir, sample_csv_content):
        """Test successful file upload to S3."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Create test file
        test_file = temp_dir / "test.csv"
        test_file.write_text(sample_csv_content)

        # Upload file
        result = upload_file_to_s3(test_file, "test-bucket", "data/test.csv")

        assert result == "s3://test-bucket/data/test.csv"

        # Verify the file was uploaded
        response = s3_client.get_object(Bucket="test-bucket", Key="data/test.csv")
        assert response is not None

    def test_upload_file_to_s3_file_not_found(self):
        """Test uploading non-existent file."""
        result = upload_file_to_s3("nonexistent.csv", "test-bucket", "data/test.csv")
        assert result is None

    @mock_s3
    def test_upload_file_to_s3_content_type_detection(self, temp_dir):
        """Test content type detection for different file types."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Test JSON file
        json_file = temp_dir / "test.json"
        json_file.write_text('{"key": "value"}')

        result = upload_file_to_s3(json_file, "test-bucket", "data/test.json")
        assert result == "s3://test-bucket/data/test.json"

        # Test HTML file
        html_file = temp_dir / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        result = upload_file_to_s3(html_file, "test-bucket", "data/test.html")
        assert result == "s3://test-bucket/data/test.html"


class TestUploadFolderToS3:
    """Test the upload_folder_to_s3 function."""

    @mock_s3
    def test_upload_folder_to_s3_success(self, temp_dir, sample_csv_content):
        """Test successful folder upload to S3."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Create test folder structure
        test_folder = temp_dir / "test_folder"
        test_folder.mkdir()

        # Create files in folder
        (test_folder / "file1.csv").write_text(sample_csv_content)
        (test_folder / "file2.csv").write_text(sample_csv_content)

        # Upload folder
        s3_prefix, uploaded_uris = upload_folder_to_s3(test_folder, "test-bucket", "data/")

        assert s3_prefix == "s3://test-bucket/data/"
        assert len(uploaded_uris) == 2
        assert all(uri.startswith("s3://test-bucket/data/") for uri in uploaded_uris)

    def test_upload_folder_to_s3_invalid_directory(self):
        """Test uploading invalid directory."""
        s3_prefix, uploaded_uris = upload_folder_to_s3("nonexistent_folder", "test-bucket", "data/")

        assert s3_prefix == "s3://test-bucket/data/"
        assert uploaded_uris == []


class TestListS3Files:
    """Test the list_s3_files function."""

    @mock_s3
    def test_list_s3_files_basic(self):
        """Test basic S3 file listing."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload test files
        s3_client.put_object(Bucket="test-bucket", Key="data/file1.csv", Body="content1")
        s3_client.put_object(Bucket="test-bucket", Key="data/file2.csv", Body="content2")
        s3_client.put_object(Bucket="test-bucket", Key="other/file3.csv", Body="content3")

        # List files
        files = list_s3_files("test-bucket", "data/")

        assert len(files) == 2
        assert all("data/" in file for file in files)
        assert all(file.startswith("s3://test-bucket/") for file in files)

    @mock_s3
    def test_list_s3_files_with_pattern(self):
        """Test S3 file listing with pattern."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload test files
        s3_client.put_object(Bucket="test-bucket", Key="data/file1.csv", Body="content1")
        s3_client.put_object(Bucket="test-bucket", Key="data/file2.txt", Body="content2")
        s3_client.put_object(Bucket="test-bucket", Key="data/file3.csv", Body="content3")

        # List files with pattern
        files = list_s3_files("test-bucket", "data/", pattern=r"\.csv$")

        assert len(files) == 2
        assert all("file1.csv" in file or "file3.csv" in file for file in files)


class TestReadFiles:
    """Test the read_files function."""

    @mock_s3
    def test_read_files_single_file(self, sample_csv_content):
        """Test reading single file from S3."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload CSV content
        s3_client.put_object(Bucket="test-bucket", Key="data/test.csv", Body=sample_csv_content)

        # Read file
        result = read_files("s3://test-bucket/data/test.csv")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "id" in result.columns

    @mock_s3
    def test_read_files_multiple_files(self, sample_csv_content):
        """Test reading multiple files from S3."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload multiple CSV files
        for i in range(3):
            s3_client.put_object(Bucket="test-bucket", Key=f"data/file{i}.csv", Body=sample_csv_content)

        # Read files
        files = [
            "s3://test-bucket/data/file0.csv",
            "s3://test-bucket/data/file1.csv",
            "s3://test-bucket/data/file2.csv",
        ]
        result = read_files(files)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 15  # 5 rows * 3 files

    @mock_s3
    def test_read_files_with_local_cache(self, temp_dir, sample_csv_content):
        """Test reading files with local cache."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload CSV content
        s3_client.put_object(Bucket="test-bucket", Key="data/test.csv", Body=sample_csv_content)

        # Create local cache file
        cache_file = temp_dir / "cache.csv"
        cache_file.write_text(sample_csv_content)

        # Read with cache
        result = read_files("s3://test-bucket/data/test.csv", local_cache_path=str(cache_file))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    @mock_s3
    def test_read_files_with_columns(self, sample_csv_content):
        """Test reading files with specific columns."""
        # Setup S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        # Upload CSV content
        s3_client.put_object(Bucket="test-bucket", Key="data/test.csv", Body=sample_csv_content)

        # Read specific columns
        result = read_files("s3://test-bucket/data/test.csv", columns=["id", "value"])

        assert list(result.columns) == ["id", "value"]
        assert len(result) == 5


class TestReadDataset:
    """Test the read_dataset function."""

    @patch("bituslabs_ds.s3_utils.dataset")
    @patch("bituslabs_ds.s3_utils.fs.S3FileSystem")
    def test_read_dataset_success(self, mock_s3fs, mock_dataset):
        """Test successful dataset reading."""
        # Mock the dataset
        mock_dataset.return_value = MagicMock()

        # Call function
        result = read_dataset("s3://test-bucket/data/", region="us-west-2")

        # Verify calls
        mock_s3fs.assert_called_once_with(region="us-west-2")
        mock_dataset.assert_called_once()

        assert result is not None

    @patch("bituslabs_ds.s3_utils.dataset")
    @patch("bituslabs_ds.s3_utils.fs.S3FileSystem")
    def test_read_dataset_with_format(self, mock_s3fs, mock_dataset):
        """Test dataset reading with specific format."""
        # Mock the dataset
        mock_dataset.return_value = MagicMock()

        # Call function with format
        result = read_dataset("s3://test-bucket/data/", region="us-west-2", data_format="csv")

        # Verify calls
        mock_dataset.assert_called_once()
        call_args = mock_dataset.call_args
        assert call_args[1]["format"] == "csv"

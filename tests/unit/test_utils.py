"""
Unit tests for the utils module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import zscore

from bituslabs_ds.utils import (
    add_event_group_by_gap,
    batch_iterator,
    check_consecutive_event,
    column_iterator,
    convert_to_list,
    df_power_transform,
    get_data_from_url,
    group_iterator,
    keep_numeric_columns,
    remove_outliers,
    save_list,
)


class TestGetDataFromUrl:
    """Test the get_data_from_url function."""

    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_download_and_extract_success(self, mock_listdir, mock_exists, mock_run):
        """Test successful download and extraction."""
        mock_exists.return_value = False
        mock_listdir.return_value = []
        mock_run.return_value = MagicMock()

        result = get_data_from_url("https://example.com/data.zip", "data.zip", "extracted", overwrite=False)

        assert result is True
        assert mock_run.call_count == 2  # wget and unzip

    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_skip_existing_directory(self, mock_listdir, mock_exists, mock_isdir):
        """Test skipping download when directory exists and is not empty."""
        mock_exists.return_value = True
        # The skip-path checks both os.path.exists AND os.path.isdir on extraction_dir.
        mock_isdir.return_value = True
        mock_listdir.return_value = ["file1.txt", "file2.txt"]

        result = get_data_from_url("https://example.com/data.zip", "data.zip", "extracted", overwrite=False)

        assert result is True

    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_wget_not_found(self, mock_listdir, mock_exists, mock_run):
        """Test handling when wget command is not found."""
        mock_exists.return_value = False
        mock_listdir.return_value = []
        mock_run.side_effect = FileNotFoundError()

        result = get_data_from_url("https://example.com/data.zip", "data.zip", "extracted", overwrite=False)

        assert result is False


class TestSaveList:
    """Test the save_list function."""

    def test_save_json_format(self, temp_dir):
        """Test saving list in JSON format."""
        test_list = [1, 2, 3, "test", True]
        filepath = temp_dir / "test.json"

        save_list(test_list, str(filepath), "json")

        assert filepath.exists()
        with open(filepath, "r") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_list

    def test_save_python_format(self, temp_dir):
        """Test saving list in Python format."""
        test_list = [1, 2, 3, "test", True]
        filepath = temp_dir / "test.py"

        save_list(test_list, str(filepath), "python")

        assert filepath.exists()
        with open(filepath, "r") as f:
            content = f.read()
        assert "my_list = [" in content
        assert "1," in content
        # save_list("python") uses repr(), which renders strings with single quotes.
        assert "'test'," in content

    def test_invalid_format(self):
        """Test error handling for invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            save_list([1, 2, 3], "test.txt", "invalid")


class TestRemoveOutliers:
    """Test the remove_outliers function."""

    def test_remove_outliers_numpy_array(self):
        """Test removing outliers from numpy array."""
        # With only 5 points, the [100,200] outlier inflates the std enough that
        # a 2-sigma threshold lets it through. 1.5 σ is enough to flag it.
        data = np.array([[1, 2], [2, 3], [100, 200], [3, 4], [4, 5]])

        filtered_data, filtered_indices = remove_outliers(data, z_thresh=1.5)

        assert len(filtered_data) < len(data)
        assert len(filtered_indices) == len(data)
        assert np.all(filtered_indices == [True, True, False, True, True])

    def test_remove_outliers_dataframe(self):
        """Test removing outliers from pandas DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 100, 3, 4], "col2": [2, 3, 200, 4, 5]})

        filtered_df, filtered_indices = remove_outliers(df, z_thresh=1.5)

        assert len(filtered_df) < len(df)
        assert len(filtered_indices) == len(df)
        assert filtered_indices.sum() == 4  # 4 rows should remain


class TestKeepNumericColumns:
    """Test the keep_numeric_columns function."""

    def test_keep_only_numeric_columns(self):
        """Test keeping only numeric columns."""
        df = pd.DataFrame(
            {
                "numeric1": [1, 2, 3],
                "numeric2": [1.1, 2.2, 3.3],
                "string": ["a", "b", "c"],
                "boolean": [True, False, True],
            }
        )

        result = keep_numeric_columns(df)

        assert "numeric1" in result.columns
        assert "numeric2" in result.columns
        assert "string" not in result.columns
        assert "boolean" not in result.columns

    def test_all_numeric_columns(self):
        """Test when all columns are numeric."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [1.1, 2.2, 3.3]})

        result = keep_numeric_columns(df)

        assert len(result.columns) == 2
        assert result.equals(df)


class TestDfPowerTransform:
    """Test the df_power_transform function."""

    def test_power_transform_single_column(self, sample_numeric_dataframe):
        """Test power transformation on a single column."""
        df = sample_numeric_dataframe.copy()
        original_col = "feature_1"

        result = df_power_transform(df, col_names=original_col, suffix="_transformed")

        assert f"{original_col}_transformed" in result.columns
        assert original_col in result.columns

    def test_power_transform_multiple_columns(self, sample_numeric_dataframe):
        """Test power transformation on multiple columns."""
        df = sample_numeric_dataframe.copy()
        cols = ["feature_1", "feature_2"]

        result = df_power_transform(df, col_names=cols, suffix="_log")

        for col in cols:
            assert f"{col}_log" in result.columns

    def test_power_transform_all_columns(self, sample_numeric_dataframe):
        """Test power transformation on all columns."""
        df = sample_numeric_dataframe.copy()

        result = df_power_transform(df, suffix="_transformed")

        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                assert f"{col}_transformed" in result.columns


class TestCheckConsecutiveEvent:
    """Test the check_consecutive_event function."""

    def test_consecutive_events_both_directions(self):
        """Test checking consecutive events in both directions."""
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="30s")})

        result = check_consecutive_event(df, "timestamp", threshold=40, direction="both")

        assert "is_consecutive" in result.columns
        assert result["is_consecutive"].dtype == bool

    def test_consecutive_events_before_only(self):
        """Test checking consecutive events before only."""
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="30s")})

        result = check_consecutive_event(df, "timestamp", threshold=40, direction="before")

        assert "is_consecutive" in result.columns

    def test_consecutive_events_after_only(self):
        """Test checking consecutive events after only."""
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="30s")})

        result = check_consecutive_event(df, "timestamp", threshold=40, direction="after")

        assert "is_consecutive" in result.columns


class TestAddEventGroupByGap:
    """Test the add_event_group_by_gap function."""

    def test_add_event_group_by_gap(self):
        """Test adding event groups based on gaps."""
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="30s")})

        result = add_event_group_by_gap(df, "timestamp", threshold=40)

        assert "gap_group" in result.columns
        assert result["gap_group"].dtype in [np.int64, np.int32]


class TestColumnIterator:
    """Test the column_iterator function."""

    def test_column_iterator_single_n(self, sample_dataframe):
        """Test column iterator with single n value."""
        ordered_cols = ["id", "value", "category"]
        n_columns = 2

        results = list(column_iterator(sample_dataframe, ordered_cols, n_columns))

        assert len(results) == 1
        df_subset, n = results[0]
        assert n == 2
        assert list(df_subset.columns) == ["id", "value"]

    def test_column_iterator_multiple_n(self, sample_dataframe):
        """Test column iterator with multiple n values."""
        ordered_cols = ["id", "value", "category"]
        n_columns = [1, 2, 3]

        results = list(column_iterator(sample_dataframe, ordered_cols, n_columns))

        assert len(results) == 3
        for i, (df_subset, n) in enumerate(results):
            assert n == i + 1
            assert len(df_subset.columns) == i + 1


class TestGroupIterator:
    """Test the group_iterator function."""

    def test_group_iterator_with_group_col(self, sample_dataframe):
        """Test group iterator with group column."""
        group_col = "category"
        count_thresh = 1

        results = list(group_iterator(sample_dataframe, group_col, count_thresh))

        assert len(results) > 0
        for df_subset, group, index in results:
            assert isinstance(df_subset, pd.DataFrame)
            assert group in sample_dataframe[group_col].unique()
            assert isinstance(index, int)

    def test_group_iterator_no_group_col(self, sample_dataframe):
        """Test group iterator without group column."""
        results = list(group_iterator(sample_dataframe, None))

        assert len(results) == 1
        df_subset, group, index = results[0]
        assert df_subset.equals(sample_dataframe)
        assert group is None
        assert index == 0


class TestBatchIterator:
    """Test the batch_iterator function."""

    def test_batch_iterator(self):
        """Test batch iterator functionality."""
        data = list(range(10))
        chunk_size = 3

        results = list(batch_iterator(data, chunk_size))

        assert len(results) == 4  # 10 items / 3 = 4 chunks
        assert results[0][0] == [0, 1, 2]
        assert results[1][0] == [3, 4, 5]
        assert results[2][0] == [6, 7, 8]
        assert results[3][0] == [9]

    def test_batch_iterator_exact_chunks(self):
        """Test batch iterator with exact chunk division."""
        data = list(range(9))
        chunk_size = 3

        results = list(batch_iterator(data, chunk_size))

        assert len(results) == 3
        for i, (chunk, chunk_num, total_chunks) in enumerate(results):
            assert chunk_num == i + 1
            assert total_chunks == 3


class TestConvertToList:
    """Test the convert_to_list function."""

    def test_convert_string_to_list(self):
        """Test converting string to list."""
        result = convert_to_list("test")
        assert result == ["test"]

    def test_convert_int_to_list(self):
        """Test converting integer to list."""
        result = convert_to_list(42)
        assert result == [42]

    def test_convert_list_to_list(self):
        """Test converting list to list."""
        original = [1, 2, 3]
        result = convert_to_list(original)
        assert result == original

    def test_convert_numpy_array_to_list(self):
        """Test converting numpy array to list."""
        arr = np.array([1, 2, 3])
        result = convert_to_list(arr)
        assert result == [1, 2, 3]

    def test_convert_empty_value(self):
        """Test converting empty/None value."""
        result = convert_to_list(None)
        assert result == []

        result = convert_to_list("")
        assert result == []

    def test_convert_invalid_type(self):
        """Test converting invalid type."""
        with pytest.raises(TypeError):
            convert_to_list(object())

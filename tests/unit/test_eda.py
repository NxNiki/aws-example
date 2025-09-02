"""
Unit tests for the eda module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bituslabs_ds.eda import (
    Anova,
    DataProfiler,
    DataVisualizer,
    read_csv_cols,
    read_excel,
    split_column_by_multiple_separators,
    split_column_by_threshold,
)


class TestReadCsvCols:
    """Test the read_csv_cols function."""

    def test_read_csv_cols_basic(self, temp_dir, sample_csv_content):
        """Test basic CSV reading functionality."""
        # Create test CSV file
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(sample_csv_content)

        result = read_csv_cols([str(csv_file)], ["id", "value"])

        assert len(result) == 5
        assert "id" in result.columns
        assert "value" in result.columns
        assert result["id"].iloc[0] == 1

    def test_read_csv_cols_with_filters(self, temp_dir):
        """Test CSV reading with filters."""
        csv_content = """id,value,category
1,10.5,A
2,20.3,B
3,15.7,A
4,8.2,C
5,12.1,B"""

        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = read_csv_cols([str(csv_file)], ["id", "value"], filters={"category": "A"})

        assert len(result) == 2
        assert all(result["category"] == "A")

    def test_read_csv_cols_with_sampling(self, temp_dir):
        """Test CSV reading with sampling."""
        csv_content = "\n".join([f"{i},{i*2},A" for i in range(100)])
        csv_content = "id,value,category\n" + csv_content

        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = read_csv_cols([str(csv_file)], ["id", "value"], sampling=10)

        assert len(result) == 10

    def test_read_csv_cols_multiple_files(self, temp_dir):
        """Test reading from multiple CSV files."""
        # Create two CSV files
        csv1 = temp_dir / "test1.csv"
        csv1.write_text("id,value\n1,10\n2,20")

        csv2 = temp_dir / "test2.csv"
        csv2.write_text("id,value\n3,30\n4,40")

        result = read_csv_cols([str(csv1), str(csv2)], ["id", "value"])

        assert len(result) == 4
        assert set(result["id"]) == {1, 2, 3, 4}


class TestReadExcel:
    """Test the read_excel function."""

    def test_read_excel_single_sheet(self, temp_dir):
        """Test reading Excel file with single sheet."""
        # Create test data
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        excel_file = temp_dir / "test.xlsx"
        df.to_excel(excel_file, index=False)

        result = read_excel(str(excel_file))

        assert len(result) == 3
        assert "A" in result.columns
        assert "B" in result.columns

    def test_read_excel_multiple_sheets(self, temp_dir):
        """Test reading Excel file with multiple sheets."""
        # Create test data for multiple sheets
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"C": [5, 6], "D": [7, 8]})

        excel_file = temp_dir / "test.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            df1.to_excel(writer, sheet_name="Sheet1", index=False)
            df2.to_excel(writer, sheet_name="Sheet2", index=False)

        result = read_excel(str(excel_file))

        assert len(result) == 4
        assert "sheet_name" in result.columns
        assert "A" in result.columns
        assert "C" in result.columns


class TestSplitColumnByThreshold:
    """Test the split_column_by_threshold function."""

    def test_split_single_column(self, sample_dataframe):
        """Test splitting a single column by threshold."""
        df = sample_dataframe.copy()

        result = split_column_by_threshold(df, "value", threshold=0.0)

        assert "value_below_0.0" in result.columns
        assert "value_above_0.0" in result.columns

        # Check that values are properly split
        below_mask = df["value"] <= 0.0
        above_mask = df["value"] > 0.0

        assert result.loc[below_mask, "value_below_0.0"].equals(df.loc[below_mask, "value"])
        assert result.loc[above_mask, "value_above_0.0"].equals(df.loc[above_mask, "value"])

    def test_split_multiple_columns(self, sample_dataframe):
        """Test splitting multiple columns by different thresholds."""
        df = sample_dataframe.copy()

        result = split_column_by_threshold(df, ["value", "profit"], threshold=[0.0, 10.0])

        assert "value_below_0.0" in result.columns
        assert "value_above_0.0" in result.columns
        assert "profit_below_10.0" in result.columns
        assert "profit_above_10.0" in result.columns


class TestSplitColumnByMultipleSeparators:
    """Test the split_column_by_multiple_separators function."""

    def test_split_column_by_separator(self):
        """Test splitting column by separator."""
        df = pd.DataFrame({"items": ["a;b;c", "d;e", "f;g;h;i", "single"]})

        result = split_column_by_multiple_separators(df, "items", sep=";")

        assert "items_1" in result.columns
        assert "items_2" in result.columns
        assert "items_3" in result.columns
        assert "items_4" in result.columns

        # Check first row
        assert result.loc[0, "items_1"] == "a"
        assert result.loc[0, "items_2"] == "b"
        assert result.loc[0, "items_3"] == "c"
        assert pd.isna(result.loc[0, "items_4"])


class TestDataProfiler:
    """Test the DataProfiler class."""

    def test_data_profiler_initialization(self, sample_dataframe):
        """Test DataProfiler initialization."""
        profiler = DataProfiler(sample_dataframe)

        assert profiler.df.equals(sample_dataframe)
        assert profiler.skewness_threshold == 1.5
        assert hasattr(profiler, "_original_numeric_columns")

    def test_check_numeric_columns(self, sample_dataframe):
        """Test checking numeric columns."""
        profiler = DataProfiler(sample_dataframe)

        numeric_cols = profiler.check_numeric_columns()

        expected_numeric = ["id", "value", "profit", "bet_amount", "slot_type"]
        assert set(numeric_cols) == set(expected_numeric)

    def test_check_numeric_columns_exclude_boolean(self, sample_dataframe):
        """Test checking numeric columns excluding boolean."""
        profiler = DataProfiler(sample_dataframe)

        numeric_cols = profiler.check_numeric_columns(include_boolean=False)

        # Should not include boolean columns if any exist
        assert all(not profiler.df[col].dtype == "bool" for col in numeric_cols)

    def test_count_missing_columns(self, sample_dataframe):
        """Test counting missing values in columns."""
        # Add some missing values
        df = sample_dataframe.copy()
        df.loc[0:2, "value"] = np.nan
        df.loc[5:7, "category"] = np.nan

        profiler = DataProfiler(df)
        missing_info = profiler.count_missing_columns(verbose=False)

        assert missing_info["column"] == list(df.columns)
        assert missing_info["missing_count"][df.columns.get_loc("value")] == 3
        assert missing_info["missing_count"][df.columns.get_loc("category")] == 3

    def test_get_skewed_columns(self, sample_numeric_dataframe):
        """Test getting skewed columns."""
        profiler = DataProfiler(sample_numeric_dataframe)

        pos_skewed, neg_skewed = profiler.get_skewed_columns()

        assert isinstance(pos_skewed, list)
        assert isinstance(neg_skewed, list)
        assert all(col in profiler.df.columns for col in pos_skewed + neg_skewed)

    def test_transform_skewed_columns(self, sample_numeric_dataframe):
        """Test transforming skewed columns."""
        profiler = DataProfiler(sample_numeric_dataframe)

        profiler.transform_skewed_columns()

        # Check if transformed columns were added
        original_cols = set(profiler.df.columns)
        transformed_cols = set(profiler.df.columns)

        # Should have at least the original columns
        assert original_cols.issubset(transformed_cols)

    def test_get_correlation_matrix(self, sample_numeric_dataframe):
        """Test getting correlation matrix."""
        profiler = DataProfiler(sample_numeric_dataframe)

        corr_matrix = profiler.get_correlation_matrix()

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
        assert all(col in profiler.df.columns for col in corr_matrix.columns)


class TestDataVisualizer:
    """Test the DataVisualizer class."""

    def test_data_visualizer_initialization(self, sample_dataframe):
        """Test DataVisualizer initialization."""
        viz = DataVisualizer(sample_dataframe)

        assert viz.data.equals(sample_dataframe)
        assert hasattr(viz, "data_profiler")
        assert isinstance(viz.data_profiler, DataProfiler)

    def test_data_visualizer_with_profiler(self, sample_dataframe):
        """Test DataVisualizer initialization with existing profiler."""
        profiler = DataProfiler(sample_dataframe)
        viz = DataVisualizer(profiler)

        assert viz.data_profiler is profiler
        assert viz.data.equals(sample_dataframe)

    def test_create_figure_basic(self, sample_dataframe):
        """Test basic figure creation."""
        viz = DataVisualizer(sample_dataframe)

        fig, axes = viz.create_figure(layout_cols=["value", "profit"], n_cols=2, fig_title="Test Figure")

        assert fig is not None
        assert len(axes) == 2
        assert viz.figure is fig

    def test_create_figure_with_group(self, sample_dataframe):
        """Test figure creation with grouping."""
        viz = DataVisualizer(sample_dataframe)

        fig, axes = viz.create_figure(layout_cols=["value"], group_col="category", n_cols=1)

        assert fig is not None
        assert len(axes) == 1
        assert viz.group_col == "category"


class TestAnova:
    """Test the Anova class."""

    def test_anova_initialization(self, sample_dataframe):
        """Test Anova initialization."""
        anova = Anova(sample_dataframe, between_vars="category", var_columns=["value", "profit"])

        assert anova.between_vars == ["category"]
        assert anova.var_columns == ["value", "profit"]
        assert hasattr(anova, "anova_report")
        assert hasattr(anova, "post_hoc_report")

    def test_anova_with_multiple_between_vars(self, sample_dataframe):
        """Test Anova with multiple between variables."""
        # Add another categorical variable
        df = sample_dataframe.copy()
        df["group"] = np.random.choice(["X", "Y"], len(df))

        anova = Anova(df, between_vars=["category", "group"], var_columns=["value", "profit"])

        assert len(anova.between_vars) == 2
        assert "category" in anova.between_vars
        assert "group" in anova.between_vars

    def test_anova_table_property(self, sample_dataframe):
        """Test anova_table property."""
        anova = Anova(sample_dataframe, between_vars="category", var_columns=["value", "profit"])

        # Run ANOVA first
        anova.run_anova()

        table = anova.anova_table
        assert isinstance(table, pd.DataFrame)

    def test_tuple_to_string_conversion(self):
        """Test tuple to string conversion utility."""
        # Test with tuple
        result = Anova.tuple_to_string(("A", "B", "C"))
        assert result == "A,B,C"

        # Test with string
        result = Anova.tuple_to_string("single")
        assert result == "single"

    def test_string_to_tuple_conversion(self):
        """Test string to tuple conversion utility."""
        # Test with comma-separated string
        result = Anova.string_to_tuple("A,B,C")
        assert result == ("A", "B", "C")

        # Test with single value
        result = Anova.string_to_tuple("single")
        assert result == ("single",)

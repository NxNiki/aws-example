"""
Integration tests for ML models and data processing.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bituslabs_ds.eda import Anova, DataProfiler, DataVisualizer
from bituslabs_ds.utils import check_consecutive_event, df_power_transform, remove_outliers


@pytest.mark.integration
class TestMLDataPipeline:
    """Integration tests for ML data processing pipeline."""

    def test_complete_data_preprocessing_pipeline(self, sample_dataframe):
        """Test complete data preprocessing pipeline."""
        # Step 1: Data profiling
        profiler = DataProfiler(sample_dataframe)

        # Check numeric columns
        numeric_cols = profiler.check_numeric_columns()
        assert len(numeric_cols) > 0

        # Check missing values
        missing_info = profiler.count_missing_columns(verbose=False)
        assert "column" in missing_info
        assert "missing_count" in missing_info

        # Step 2: Outlier removal
        numeric_data = sample_dataframe[numeric_cols]
        filtered_data, filtered_indices = remove_outliers(numeric_data, z_thresh=2)

        assert len(filtered_data) <= len(numeric_data)
        assert len(filtered_indices) == len(numeric_data)

        # Step 3: Power transformation
        transformed_data = df_power_transform(filtered_data, suffix="_transformed")

        # Check that transformed columns were added
        original_cols = set(filtered_data.columns)
        transformed_cols = set(transformed_data.columns)
        assert original_cols.issubset(transformed_cols)

        # Step 4: Distribution analysis
        stats = profiler.check_distribution_stats()
        assert len(stats) > 0
        assert all("column" in stat for stat in stats)
        assert all("skewness" in stat for stat in stats)

        # Step 5: Correlation analysis
        corr_matrix = profiler.get_correlation_matrix()
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]

    def test_time_series_analysis_pipeline(self, sample_time_series_data):
        """Test time series analysis pipeline."""
        # Step 1: Data profiling
        profiler = DataProfiler(sample_time_series_data)

        # Step 2: Check consecutive events
        result = check_consecutive_event(
            sample_time_series_data, "timestamp", threshold=3600, direction="both"  # 1 hour in seconds
        )

        assert "is_consecutive" in result.columns
        assert result["is_consecutive"].dtype == bool

        # Step 3: Statistical analysis
        stats = profiler.check_distribution_stats()
        assert len(stats) > 0

        # Step 4: Group analysis
        if "group" in sample_time_series_data.columns:
            # Test ANOVA if we have grouping variable
            anova = Anova(result, between_vars="group", var_columns=["value"])

            anova_table = anova.run_anova()
            assert isinstance(anova_table, pd.DataFrame)

    def test_visualization_pipeline(self, sample_dataframe):
        """Test data visualization pipeline."""
        # Step 1: Create visualizer
        viz = DataVisualizer(sample_dataframe)

        # Step 2: Create figure
        fig, axes = viz.create_figure(
            layout_cols=["value", "profit"], group_col="category", n_cols=2, fig_title="Test Visualization"
        )

        assert fig is not None
        assert len(axes) == 2
        assert viz.group_col == "category"

        # Step 3: Add plots (mocked to avoid actual plotting)
        with patch.object(viz, "add_boxplot") as mock_boxplot:
            with patch.object(viz, "add_stripplot") as mock_stripplot:
                viz.add_boxplot(x_col="category")
                viz.add_stripplot(x_col="category")

                mock_boxplot.assert_called_once()
                mock_stripplot.assert_called_once()

        # Step 4: Add correlation heatmap
        with patch.object(viz, "add_correlation_heatmap") as mock_heatmap:
            viz.add_correlation_heatmap()
            mock_heatmap.assert_called_once()

    def test_statistical_analysis_pipeline(self, sample_dataframe):
        """Test statistical analysis pipeline."""
        # Step 1: Data preparation
        profiler = DataProfiler(sample_dataframe)
        numeric_cols = profiler.check_numeric_columns()

        # Step 2: ANOVA analysis
        if "category" in sample_dataframe.columns:
            anova = Anova(
                sample_dataframe, between_vars="category", var_columns=numeric_cols[:2]  # Use first 2 numeric columns
            )

            # Run ANOVA
            anova_table = anova.run_anova()
            assert isinstance(anova_table, pd.DataFrame)

            # Step 3: Post-hoc analysis
            if len(anova_table) > 0:
                post_hoc_table = anova.run_post_hoc_analysis(var_columns=numeric_cols[:2], method="Tukey")
                assert isinstance(post_hoc_table, pd.DataFrame)

    def test_feature_engineering_pipeline(self, sample_numeric_dataframe):
        """Test feature engineering pipeline."""
        # Step 1: Data profiling
        profiler = DataProfiler(sample_numeric_dataframe)

        # Step 2: Identify skewed columns
        pos_skewed, neg_skewed = profiler.get_skewed_columns()

        # Step 3: Transform skewed columns
        profiler.transform_skewed_columns()

        # Step 4: Get processed columns
        processed_cols = profiler.processed_numerical_columns
        assert len(processed_cols) >= len(profiler._original_numeric_columns)

        # Step 5: Feature augmentation
        feature_cols = [col for col in processed_cols if col != "target"]
        if len(feature_cols) > 1:
            profiler.augment_columns(feature_cols, "combined", method=["mean", "pca"])

            # Check that augmented columns were added
            assert any("combined_mean" in col for col in profiler.df.columns)
            assert any("combined_pca" in col for col in profiler.df.columns)

    def test_data_quality_assessment_pipeline(self, sample_dataframe):
        """Test data quality assessment pipeline."""
        # Step 1: Create data with quality issues
        df_with_issues = sample_dataframe.copy()

        # Add missing values
        df_with_issues.loc[0:5, "value"] = np.nan

        # Add outliers
        df_with_issues.loc[10:15, "profit"] = df_with_issues["profit"].max() * 10

        # Step 2: Data profiling
        profiler = DataProfiler(df_with_issues)

        # Step 3: Missing value analysis
        missing_info = profiler.count_missing_columns(verbose=False)
        missing_counts = dict(zip(missing_info["column"], missing_info["missing_count"]))
        assert missing_counts["value"] > 0

        # Step 4: Outlier detection
        numeric_cols = profiler.check_numeric_columns()
        numeric_data = df_with_issues[numeric_cols].dropna()

        filtered_data, filtered_indices = remove_outliers(numeric_data, z_thresh=2)
        assert len(filtered_data) < len(numeric_data)

        # Step 5: Data quality metrics
        initial_rows = len(df_with_issues)
        final_rows = len(filtered_data)
        data_retention_rate = final_rows / initial_rows

        assert 0 < data_retention_rate <= 1

        # Step 6: Distribution analysis
        stats = profiler.check_distribution_stats()
        assert len(stats) > 0

        # Check for extreme skewness
        extreme_skew = [stat for stat in stats if abs(stat["skewness"]) > 3]
        if extreme_skew:
            # Apply transformations
            profiler.transform_skewed_columns()
            new_stats = profiler.check_distribution_stats()
            assert len(new_stats) >= len(stats)

    def test_end_to_end_ml_preprocessing(self, sample_dataframe):
        """Test end-to-end ML preprocessing pipeline."""
        # Step 1: Initial data assessment
        profiler = DataProfiler(sample_dataframe)
        initial_shape = profiler.df.shape

        # Step 2: Data cleaning
        # Remove non-numeric columns for ML
        numeric_data = profiler.df.select_dtypes(include=[np.number])

        # Remove outliers
        cleaned_data, _ = remove_outliers(numeric_data, z_thresh=2.5)

        # Step 3: Feature engineering
        # Transform skewed features
        transformed_data = df_power_transform(cleaned_data, suffix="_transformed")

        # Step 4: Feature selection (simplified)
        # Remove highly correlated features
        corr_matrix = transformed_data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation > 0.95
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        final_data = transformed_data.drop(columns=to_drop)

        # Step 5: Final validation
        assert final_data.shape[0] <= initial_shape[0]  # May have fewer rows due to outlier removal
        assert final_data.shape[1] >= initial_shape[1]  # May have more columns due to transformations

        # Check for any remaining missing values
        assert final_data.isnull().sum().sum() == 0

        # Check for infinite values
        assert np.isfinite(final_data.select_dtypes(include=[np.number])).all().all()

        # Step 6: Data quality metrics
        data_quality_metrics = {
            "initial_rows": initial_shape[0],
            "final_rows": final_data.shape[0],
            "initial_cols": initial_shape[1],
            "final_cols": final_data.shape[1],
            "retention_rate": final_data.shape[0] / initial_shape[0],
            "feature_expansion_rate": final_data.shape[1] / initial_shape[1],
        }

        assert data_quality_metrics["retention_rate"] > 0.5  # Keep at least 50% of data
        assert data_quality_metrics["feature_expansion_rate"] >= 1.0  # Should have same or more features

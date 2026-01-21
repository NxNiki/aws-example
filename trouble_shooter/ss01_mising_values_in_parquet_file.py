"""
in ss01 feature engineering etl, we replace rare NULL values (caused by dividing by 0) to 0, but we still see 
a small proportation of missing values in the parquet file.
"""

import pandas as pd

data = pd.read_parquet("/Users/niuxin/Documents/aws-example/jobs/output_ss01_wucaishen/ss01_features_grouped.parquet")

# Show columns with NaN or NA values and how many there are
missing_counts = data.isnull().sum()
cols_with_missing = missing_counts[missing_counts > 0]
print("Columns with NaN/NA values and count:")
print(cols_with_missing)

print("\nRows with NaN/NA values:")
missing_rows = data[data.isnull().any(axis=1)]
print(missing_rows)

# Show all columns for rows where only 'profit_rate' is NaN or NA (other columns are not)
if "profit_rate" in data.columns:
    profit_rate_nan_only = data[data["profit_rate"].isnull()]
    print("\nRows where ONLY 'profit_rate' is NaN/NA (showing all columns):")
    print(profit_rate_nan_only[["user_id", ""]])
else:
    print("'profit_rate' column not found in the dataframe.")

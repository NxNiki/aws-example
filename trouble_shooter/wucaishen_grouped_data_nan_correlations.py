"""
wucaishen data has empty values in the correlation map (in user segmentation)
for the following variables:
    payout_min
    payout_p25
    win_streak_min
    win_streak_p25
    win_streak_median

This is unexpected as add_correlation_heatmap in DataVisualizer (eda.py) removes
rows with nan values.


conclusion: power transform and scaler does not cause NaN correations. It is confirmed
that remove_outlier() cause the issue!
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, power_transform

file = "/Users/niuxin/Documents/aws-example/jobs/output_wucaishen/wucaishen_grouped_data_2024.parquet"
data = pd.read_parquet(
    file, columns=["payout_min", "payout_p25", "win_streak_min", "win_streak_p25", "win_streak_median"]
)

print(data.describe())

print((data > 0).sum())

print(data.corr())
data = pd.DataFrame(power_transform(data), columns=data.columns)
print(data.corr())

scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print(data.corr())

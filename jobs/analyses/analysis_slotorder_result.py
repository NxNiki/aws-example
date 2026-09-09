"""
check the values of columns in slotmachine and convert them to values according to a LUT.
"""

from bituslabs_ds.s3_utils import read_to_pandas_df

bucket = "bituslabs-team-ai"
data = "ds-data-slotorders/slot_orders_wucaishen_for_mathtable.csv"

df = read_to_pandas_df(bucket, data)

# Set to store unique integers
unique_numbers = set()

# Loop through columns 1 to 8 (0-indexed columns 0 to 7)
for col in ["column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8"]:
    for cell in df[col].dropna():  # Ignore NaN values
        numbers = cell.split(",")
        for num in numbers:
            num = num.strip()
            if num.isdigit():
                unique_numbers.add(int(num))

# Print sorted list of unique values
print("Unique integer values in columns 1-8:")
print(sorted(unique_numbers))

import pandas as pd

# Load the data
data_file = "/Users/niuxin/Downloads/slot_orders_wucaishen_for_mathtable.csv"
df = pd.read_csv(data_file)

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

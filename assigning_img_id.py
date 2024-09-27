import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'your_file.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Add an 'id' column with sequential numbers starting from 1
df['id'] = range(1, len(df) + 1)
df['agreement'] = 3

# Save the modified DataFrame back to a new CSV file
df.to_csv('your_new_file.csv', index=False)
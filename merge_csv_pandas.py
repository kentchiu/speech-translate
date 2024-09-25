import pandas as pd

# Read the CSV file
df = pd.read_csv('temp.csv')

# Merge rows with the same 'Filename'
merged_df = df.groupby('Filename').agg(lambda x: ' | '.join(x.dropna().astype(str))).reset_index()

# Write the merged rows back to the CSV file
merged_df.to_csv('temp.csv', index=False)

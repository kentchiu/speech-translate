import pandas as pd

df = pd.read_csv("test-data/compoare-0925.csv").sort_values(by="Filename")

# Group the data by 'Filename' and concatenate the other columns with newline as the separator
grouped_df = (
    df.groupby("Filename").agg(lambda x: "<br>".join(x.astype(str))).reset_index()
)

# Create a markdown table format
markdown_table = "| Filename | Model | Lang | Translation Time | Transcription Time | Translation | Transcription |\n"
markdown_table += "|----------|-------|------|------------------|--------------------|-------------|---------------|\n"

# Add rows to the markdown table
for _, row in grouped_df.iterrows():
    markdown_table += f"| {row['Filename']} | {row['Model']} | {row['Lang']} | {row['Translation Time']} | {row['Transcription Time']} | {row['Translation']} | {row['Transcription']} |\n"


print(markdown_table)

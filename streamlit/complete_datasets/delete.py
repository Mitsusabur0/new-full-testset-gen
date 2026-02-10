import pandas as pd

# Read the parquet file
df = pd.read_parquet('streamlit/complete_datasets/testset_results.parquet')

# Delete the first 20 rows
df = df.iloc[20:]

# Save back to parquet
df.to_parquet('streamlit/complete_datasets/smaller.parquet', index=False)

print(f"Original rows: {len(pd.read_parquet('streamlit/complete_datasets/testset_results.parquet'))}")
print(f"Rows after deletion: {len(df)}")
print("File saved as 'output.parquet'")
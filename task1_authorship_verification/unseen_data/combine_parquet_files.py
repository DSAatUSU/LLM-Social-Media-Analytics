import pandas as pd
import glob
import os

parquet_folder = 'parquet_files'
output_file = 'combined_users_original.parquet'

# List all parquet files
parquet_files = glob.glob(os.path.join(parquet_folder, '*.parquet'))

# Combine all files into a single DataFrame
combined_df = pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)

# Save to parquet file
combined_df.to_parquet(output_file, index=False)

print(f"Combined data successfully saved to '{output_file}'")

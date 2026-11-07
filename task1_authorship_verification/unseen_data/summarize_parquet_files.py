import pandas as pd
import glob
import os

# Folder containing the parquet files
parquet_folder = 'parquet_files'

# List all parquet files in the specified folder
parquet_files = glob.glob(os.path.join(parquet_folder, '*.parquet'))

# List to hold the result for each user
summary_list = []

# Iterate through each file
for file in parquet_files:
    # Extract user_id from filename
    filename = os.path.basename(file)
    user_id = filename.split('_')[0]

    # Load parquet file into a DataFrame
    df = pd.read_parquet(file)

    # Check if 'text' column exists
    if 'text' in df.columns:
        total_posts = len(df)
        rt_posts = df['text'].str.startswith('RT ').sum()
        original_posts = total_posts - rt_posts
    else:
        total_posts = rt_posts = original_posts = 0

    # Append results to the summary list
    summary_list.append({
        'user_id': user_id,
        'total_posts': total_posts,
        'RT_posts': rt_posts,
        'original_posts': original_posts,
        'file_path': file  # include file path for later use
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_list)

# Save summary to CSV file
summary_df.to_csv('user_posts_summary.csv', index=False)

print("Summary saved successfully to 'user_posts_summary.csv'")
print(summary_df.head(50))

# Clearly identify users with fewer than 40 original tweets
insufficient_tweets_df = summary_df[summary_df['original_posts'] < 40]

if not insufficient_tweets_df.empty:
    print("\nUsers with fewer than 40 original tweets:")
    for idx, row in insufficient_tweets_df.iterrows():
        print(f"User ID: {row['user_id']} with {row['original_posts']} original tweets.")

    # Pick the first user to output their data as JSON
    selected_user_row = insufficient_tweets_df.iloc[0]
    selected_df = pd.read_parquet(selected_user_row['file_path'])

    # Save to JSON file
    json_filename = f"{selected_user_row['user_id']}_tweets.json"
    selected_df.to_json(json_filename, orient='records', lines=True)

    print(f"\nJSON file generated for user '{selected_user_row['user_id']}': {json_filename}")
else:
    print("\nAll users have 40 or more original tweets.")

import pandas as pd
import os

def load_combined_data(parquet_path, description_path, cache_path="preprocessed_combined_data.pkl"):
    CACHE_VERSION = "2.2"  # Updated clearly for this change
    regenerate = True

    if os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        regenerate = cached.get("version") != CACHE_VERSION

    if regenerate:
        print("Loading combined tweets dataset...")
        df = pd.read_parquet(parquet_path)

        print("Loading user descriptions from Excel file clearly...")
        desc_df = pd.concat(pd.read_excel(description_path, sheet_name=None), ignore_index=True)
        desc_dict = desc_df.set_index("user_id")["description"].to_dict()

        tweets_data = {}
        for user_id, group in df.groupby('user_id'):
            tweets = group[['text', 'tweet_created_at']].sort_values('tweet_created_at').to_dict('records')
            tweets_data[user_id] = tweets

        data = {
            "version": CACHE_VERSION,
            "user_descriptions": desc_dict,
            "tweets_data": tweets_data,
            "original_dataset": df
        }

        pd.to_pickle(data, cache_path)
        print("Data loaded and cached clearly.")
    else:
        print("Loading data from cache clearly...")
        desc_dict = cached["user_descriptions"]
        tweets_data = cached["tweets_data"]
        df = cached["original_dataset"]

    return desc_dict, tweets_data, df

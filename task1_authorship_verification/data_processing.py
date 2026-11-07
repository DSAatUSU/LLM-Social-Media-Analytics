import pandas as pd
import os
import openpyxl

def load_data(description_path, tweets_path, cache_path="preprocessed_data.pkl", chunksize=10_000):
    """
    Load and organize description and tweet datasets with caching.

    Args:
        description_path (str): Path to the description Excel file.
        tweets_path (str): Path to the tweets CSV file.
        cache_path (str): Path to the cached preprocessed data file.
        chunksize (int): Number of rows to process at a time for the tweets file.

    Returns:
        tuple: A dictionary of user descriptions, a dictionary of tweets grouped by user ID, and the original dataset.
    """
    CACHE_VERSION = "1.1"  # Update this version whenever preprocessing logic changes

    # Check for existing cache
    regenerate_data = False
    if os.path.exists(cache_path):
        cached_data = pd.read_pickle(cache_path)
        if cached_data.get("version") != CACHE_VERSION:
            print("Cache version mismatch. Regenerating data...")
            regenerate_data = True
    else:
        regenerate_data = True

    if regenerate_data:
        print("Loading and preprocessing datasets...")

        # Load description data
        description_data = pd.concat(
            pd.read_excel(description_path, sheet_name=None, engine="openpyxl"), ignore_index=True
        )
        user_descriptions = description_data.set_index("user_id")["description"].to_dict()

        # Preload tweets in chunks
        tweets_data = {}
        original_dataset_chunks = []  # To store the original dataset
        for chunk in pd.read_csv(
            tweets_path,
            usecols=["user_id", "text", "tweet_created_at", "retweeted_status_text", "Round"],
            chunksize=chunksize,
        ):
            # Append each chunk to reconstruct the original dataset
            original_dataset_chunks.append(chunk)

            # Filter out retweeted tweets
            filtered_chunk = chunk[chunk["retweeted_status_text"].isna()]

            for user_id, group in filtered_chunk.groupby("user_id"):
                if user_id not in tweets_data:
                    tweets_data[user_id] = []
                # Store both the text and created_at in a dictionary
                tweets_data[user_id].extend(
                    group.sort_values("tweet_created_at")[["text", "tweet_created_at", "Round"]]
                    .to_dict("records")
                )

        # Combine chunks into a single DataFrame for the original dataset
        original_dataset = pd.concat(original_dataset_chunks, ignore_index=True)

        # Save preprocessed data and original dataset to cache with the version
        data = {
            "version": CACHE_VERSION,
            "user_descriptions": user_descriptions,
            "tweets_data": tweets_data,
            "original_dataset": original_dataset
        }
        pd.to_pickle(data, cache_path)
        print("Data loaded and preprocessed successfully. Cached for future use.")

    else:
        print("Loading preprocessed data from cache...")
        user_descriptions = cached_data["user_descriptions"]
        tweets_data = cached_data["tweets_data"]
        original_dataset = cached_data["original_dataset"]

    return user_descriptions, tweets_data, original_dataset

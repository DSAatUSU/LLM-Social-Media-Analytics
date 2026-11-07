"""Load the VAPOR user descriptions and tweets for Task II (post generation).

NOTE: this module is a faithful re-implementation of the loader that was used on
the lab server (`content_generation/data_processing.py`), reconstructed from how
`user_selection.py` and `generate_and_evaluate.py` consume its output:

    users_data, tweets_data = load_data(description_path, tweets_path, cache_path)

    users_data[user_id]  -> dict with the user-profile columns of the descriptions
                            file (description, followers_count, following, verified, ...)
    tweets_data[user_id] -> list of {"text": ..., "tweet_created_at": ...} sorted by time
                            (retweets are NOT removed here; the callers drop "RT @" posts)

The preprocessed data is cached in a pickle so the large CSV is only parsed once.
"""
import os

import pandas as pd

CACHE_VERSION = "1.0"


def load_data(description_path, tweets_path, cache_path="preprocessed_users.pkl", chunksize=10_000):
    if os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        if cached.get("version") == CACHE_VERSION:
            print("Loading preprocessed data from cache...")
            return cached["users_data"], cached["tweets_data"]
        print("Cache version mismatch. Regenerating data...")

    print("Loading and preprocessing datasets...")
    description_data = pd.concat(
        pd.read_excel(description_path, sheet_name=None, engine="openpyxl"), ignore_index=True
    )
    description_data = description_data.drop_duplicates(subset="user_id").set_index("user_id")
    users_data = description_data.to_dict(orient="index")

    tweets_data = {}
    for chunk in pd.read_csv(tweets_path, usecols=["user_id", "text", "tweet_created_at"], chunksize=chunksize):
        chunk = chunk.dropna(subset=["text"])
        for user_id, group in chunk.groupby("user_id"):
            tweets_data.setdefault(user_id, []).extend(
                group[["text", "tweet_created_at"]].to_dict("records")
            )

    # keep each user's posts in chronological order
    for user_id, tweets in tweets_data.items():
        tweets.sort(key=lambda t: str(t["tweet_created_at"]))

    pd.to_pickle({"version": CACHE_VERSION, "users_data": users_data, "tweets_data": tweets_data}, cache_path)
    print("Data loaded and preprocessed successfully. Cached for future use.")
    return users_data, tweets_data

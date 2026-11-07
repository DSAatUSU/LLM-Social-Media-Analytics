import os
import json
import random
import pandas as pd
import emoji
import re
import requests
from data_processing import load_combined_data

def expand_url(short_url):
    try:
        response = requests.head(short_url, allow_redirects=True, timeout=5)
        return response.url
    except requests.RequestException:
        return short_url

def preprocess_tweets(tweets):
    preprocessed_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        normalized_tweet = re.sub(r'\s+', ' ', emoji.emojize(emoji.demojize(tweet)).strip())
        url_pattern = re.compile(r'https?://\S+')
        expanded_tweet = url_pattern.sub(lambda match: expand_url(match.group(0)), normalized_tweet)
        preprocessed_tweets.append(expanded_tweet)
    return preprocessed_tweets

def save_dataset(dataset, output_path):
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to '{output_path}'.")

def compute_class_distribution(dataset):
    class_0_count = sum(len(entry["others_tweets"]) for entry in dataset)
    class_1_count = sum(len(entry["future_tweets"]) for entry in dataset)
    return {"class_0": class_0_count, "class_1": class_1_count}

if __name__ == "__main__":
    config = {
        "parquet_path": "combined_users_original.parquet",
        "description_file_path": os.environ.get("VAPOR_DESCRIPTIONS", "data/raw/user_descriptions_rounds.xlsx"),
        "cache_path": "preprocessed_combined_data.pkl",
        "K": 20,
        "M1": 20,
        "M2": 20,
        "output_dataset_path": "datasets/classification_dataset.json",
        "class_distribution_file": "class_distributions.json"
    }

    os.makedirs("datasets", exist_ok=True)

    user_descriptions, tweets_data, _ = load_combined_data(
        parquet_path=config["parquet_path"],
        description_path=config["description_file_path"],
        cache_path=config["cache_path"]
    )

    for user_id in tweets_data:
        tweets_data[user_id] = [
            tweet for tweet in tweets_data[user_id] if not tweet["text"].startswith("RT @")
        ]

    dataset = []
    for idx, user_id in enumerate(tweets_data, start=1):
        print(f"Processing user {idx}/{len(tweets_data)}: User ID {user_id}")

        tweets_sorted = sorted(tweets_data[user_id], key=lambda x: pd.to_datetime(x['tweet_created_at']))

        if len(tweets_sorted) < (config["K"] + config["M1"]):
            print(f"User {user_id} skipped due to insufficient tweets.")
            continue

        previous_tweets_raw = [tweet["text"] for tweet in tweets_sorted[:config["K"]]]
        future_tweets_raw = [tweet["text"] for tweet in tweets_sorted[-config["M1"]:]]

        previous_tweets = preprocess_tweets(previous_tweets_raw)
        future_tweets = preprocess_tweets(future_tweets_raw)

        other_users_tweets_raw = [
            tweet["text"] for uid, tweets in tweets_data.items() if uid != user_id for tweet in tweets
        ]

        if len(other_users_tweets_raw) < config["M2"]:
            print(f"Insufficient other tweets for user {user_id}. Skipping.")
            continue

        random_other_tweets_raw = random.sample(other_users_tweets_raw, config["M2"])
        others_tweets = preprocess_tweets(random_other_tweets_raw)

        dataset_entry = {
            "user_id": user_id,
            "description": user_descriptions.get(user_id, ""),
            "previous_tweets": previous_tweets,
            "future_tweets": [{"tweet": tweet, "ground_truth_label": 1} for tweet in future_tweets],
            "others_tweets": [{"tweet": tweet, "ground_truth_label": 0} for tweet in others_tweets],
        }

        dataset.append(dataset_entry)

    save_dataset(dataset, config["output_dataset_path"])

    class_distribution = compute_class_distribution(dataset)
    print(f"Class distribution: {class_distribution}")

    if os.path.exists(config["class_distribution_file"]):
        with open(config["class_distribution_file"], "r") as f:
            all_distributions = json.load(f)
    else:
        all_distributions = {}

    all_distributions["current_run"] = class_distribution
    with open(config["class_distribution_file"], "w") as f:
        json.dump(all_distributions, f, indent=4)

    print(f"Class distribution saved to '{config['class_distribution_file']}'")

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # task1 modules
import json
from difflib import get_close_matches
from data_processing import load_data
from sampling import get_others_tweets
from datetime import datetime

# Define paths
INPUT_FOLDER = "../../datasets"   # datasets produced by task1 generate_datasets.py
OUTPUT_FOLDER = "datasets"         # same datasets + others_tweets_training

def save_dataset(dataset, output_path):
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"✅ Dataset saved to '{output_path}'")


def extract_timeframe(previous_tweets, tweets_data):
    """
    Extract the minimum and maximum timestamps for the previous tweets.
    If an exact match is not found, perform fuzzy matching.

    Args:
        previous_tweets (list): List of previous tweets.
        tweets_data (dict): Dictionary containing all tweets grouped by user_id.

    Returns:
        tuple: (min_time, max_time)
    """
    tweet_dates = []

    for tweet_text in previous_tweets:
        # Ensure tweet_text is a string (avoid None/NaN issues)
        if not isinstance(tweet_text, str) or tweet_text.strip() == "":
            continue  # Skip invalid entries

        found = False

        for user_id, user_tweets in tweets_data.items():
            for tweet in user_tweets:
                if not isinstance(tweet["text"], str):  # Ensure tweet text is a string
                    continue
                
                # Direct match
                if tweet["text"] == tweet_text:
                    tweet_dates.append(datetime.strptime(tweet["tweet_created_at"], "%a %b %d %H:%M:%S %z %Y"))
                    found = True
                    break

            if found:
                break  # Stop searching once a match is found

        # If no exact match is found, use fuzzy matching
        if not found:
            for user_id, user_tweets in tweets_data.items():
                valid_texts = [t["text"] for t in user_tweets if isinstance(t["text"], str)]  # Ensure valid strings
                
                if valid_texts:
                    close_match = get_close_matches(tweet_text, valid_texts, n=1, cutoff=0.8)
                    if close_match:
                        for tweet in user_tweets:
                            if tweet["text"] == close_match[0]:  # Found closest match
                                tweet_dates.append(datetime.strptime(tweet["tweet_created_at"], "%a %b %d %H:%M:%S %z %Y"))
                                break

    if not tweet_dates:
        raise ValueError("❌ No valid timestamps found for previous tweets.")

    return min(tweet_dates), max(tweet_dates)  # Return timeframe

if __name__ == "__main__":
    # Load tweets data
    _, tweets_data, _ = load_data(
        os.environ.get("VAPOR_DESCRIPTIONS", "data/raw/user_descriptions_rounds.xlsx"),
        os.environ.get("VAPOR_TWEETS", "data/raw/tweets_no_duplicates.csv"),
        cache_path="preprocessed_data.pkl"
    )

    # Process each dataset in INPUT_FOLDER
    for file_name in os.listdir(INPUT_FOLDER):
        if not file_name.endswith(".json"):
            continue
        
        input_path = os.path.join(INPUT_FOLDER, file_name)
        output_path = os.path.join(OUTPUT_FOLDER, file_name)
        
        if os.path.exists(output_path):
            print(f"⚠️ {file_name} already processed. Skipping.")
            continue
        
        print(f"🔄 Processing {file_name}...")
        
        # Load dataset
        with open(input_path, "r") as f:
            dataset = json.load(f)
        
        # Determine the other_tweets_sampling_method from file name
        if "random" in file_name:
            other_tweets_sampling_method = "random"
        elif "similar_topics" in file_name:
            other_tweets_sampling_method = "similar_topics"
        elif "followees" in file_name:
            other_tweets_sampling_method = "followees"
        elif "followers" in file_name:
            other_tweets_sampling_method = "followers"
        elif "both" in file_name:
            other_tweets_sampling_method = "both"
        else:
            print(f"⚠️ Unable to determine other_tweets_sampling_method for {file_name}. Skipping.")
            continue
        
        # Add others_tweets_training
        for entry in dataset:
            user_id = entry["user_id"]
            previous_tweets = entry["previous_tweets"]
            
            if not previous_tweets:
                print(f"⚠️ No previous tweets for user {user_id}. Skipping others_tweets_training.")
                entry["others_tweets_training"] = []
                continue
            
            # Get timeframe
            min_time, max_time = extract_timeframe(previous_tweets, tweets_data)
            if not min_time or not max_time:
                print(f"⚠️ Could not determine timeframe for user {user_id}. Skipping others_tweets_training.")
                entry["others_tweets_training"] = []
                continue
            
            print(f"🔍 Sampling others_tweets_training for user {user_id} using method '{other_tweets_sampling_method}'...")
            others_tweets_training = get_others_tweets(
                tweets_data, user_id, len(previous_tweets),
                method=other_tweets_sampling_method, user_tweets=previous_tweets
            )
            
            entry["others_tweets_training"] = [{"tweet": tweet, "ground_truth_label": 0} for tweet in others_tweets_training]
        
        # Save updated dataset
        save_dataset(dataset, output_path)
        
    print("✅ All datasets processed successfully!")

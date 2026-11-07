import os
import json
import pandas as pd

# Define paths
INPUT_FOLDER = "datasets"
OUTPUT_FOLDER = "testing_sets"  # Folder to store generated testing CSVs
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure output directory exists
dataset_filenames = [
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'both', 'user_sampling_method': 'random'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'both', 'user_sampling_method': 'recent_activity'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'both', 'user_sampling_method': 'top_active'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'followees', 'user_sampling_method': 'random'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'followees', 'user_sampling_method': 'recent_activity'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'followees', 'user_sampling_method': 'top_active'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'followers', 'user_sampling_method': 'random'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'followers', 'user_sampling_method': 'recent_activity'}.json",
    "dataset_{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': 'followers', 'user_sampling_method': 'top_active'}.json"
]

# Process each dataset
for dataset_file in dataset_filenames:
    input_path = os.path.join(INPUT_FOLDER, dataset_file)

    # Skip processing if dataset file does not exist
    if not os.path.exists(input_path):
        print(f"⚠️ Skipping {dataset_file} (File not found)")
        continue

    print(f"📄 Processing dataset: {dataset_file}")

    # Load the dataset
    with open(input_path, "r") as f:
        dataset = json.load(f)

    # Initialize a list to store the dataset records
    test_data_records = []

    # Process each user entry
    for entry in dataset:
        user_id = entry["user_id"]

        # ✅ Extract `previous_tweets` (it's a list of raw strings)
        previous_tweets = entry.get("previous_tweets", [])

        # ✅ Extract `future_tweets` and `others_tweets` (tweets are inside dictionaries)
        future_tweets = [
            (tweet["tweet"], 1) for tweet in entry.get("future_tweets", [])
        ]  # Label = 1
        others_tweets = [
            (tweet["tweet"], 0) for tweet in entry.get("others_tweets", [])
        ]  # Label = 0

        # Combine `future_tweets` and `others_tweets` for selection
        fixed_tweets = future_tweets + others_tweets

        # Skip users without previous tweets or fixed_tweets
        if not previous_tweets or not fixed_tweets:
            continue

        # **Generate pairs (previous_tweet, fixed_tweet)**
        for fixed_tweet, label in fixed_tweets:
            for tweet1 in previous_tweets:
                test_data_records.append([tweet1, fixed_tweet, label])

    # Convert the collected data into a DataFrame
    test_df = pd.DataFrame(test_data_records, columns=["Tweet1", "Tweet2", "Label"])

    # Save to CSV with a filename corresponding to the dataset
    output_csv_filename = f"testing_dataset_{dataset_file.replace('.json', '.csv')}"
    output_csv_path = os.path.join(OUTPUT_FOLDER, output_csv_filename)

    test_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"✅ Testing dataset saved: {output_csv_path}")

print("\n🎉 All testing datasets have been generated successfully!")

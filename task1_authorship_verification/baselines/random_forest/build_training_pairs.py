import os
import json
import pandas as pd

# Define paths
INPUT_FOLDER = "datasets"
OUTPUT_FOLDER = "training_sets"  # Folder to store generated training CSVs
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
    data_records = []

    # Process each user entry
    for entry in dataset:
        user_id = entry["user_id"]

        # ✅ Extract `previous_tweets` (it's a list of raw strings)
        previous_tweets = entry.get("previous_tweets", [])

        # ✅ Extract `others_tweets_training` (tweets are inside dictionaries)
        others_tweets_training = [
            tweet["tweet"] for tweet in entry.get("others_tweets_training", [])
        ]

        # If there are no previous_tweets, skip this user
        if not previous_tweets:
            continue

        # **Pair every previous_tweet with every other previous_tweet (label = 1)**
        for i in range(len(previous_tweets)):
            for j in range(len(previous_tweets)):
                if i != j:  # Prevent self-pairing
                    data_records.append([previous_tweets[i], previous_tweets[j], 1])

        # **Pair every previous_tweet with every tweet in others_tweets_training (label = 0)**
        for tweet1 in previous_tweets:
            for tweet2 in others_tweets_training:
                data_records.append([tweet1, tweet2, 0])

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(data_records, columns=["Tweet1", "Tweet2", "Label"])

    # Save to CSV with a filename corresponding to the dataset
    output_csv_filename = f"training_dataset_{dataset_file.replace('.json', '.csv')}"
    output_csv_path = os.path.join(OUTPUT_FOLDER, output_csv_filename)

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"✅ Training dataset saved: {output_csv_path}")

print("\n🎉 All training datasets have been generated successfully!")

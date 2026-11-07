import os
import itertools
import json
from data_processing import load_data
from sampling import sample_users, get_user_tweets, get_others_tweets

def save_dataset(dataset, output_path):
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to '{output_path}'.")

def compute_class_distribution(dataset):
    """
    Compute the distribution of classes (0 and 1) in the dataset.

    Args:
        dataset (list): The generated dataset.

    Returns:
        dict: Counts of class 0 and class 1 samples.
    """
    class_0_count = 0
    class_1_count = 0

    for entry in dataset:
        for tweet in entry["future_tweets"]:
            if tweet["ground_truth_label"] == 1:
                class_1_count += 1
        for tweet in entry["others_tweets"]:
            if tweet["ground_truth_label"] == 0:
                class_0_count += 1

    return {"class_0": class_0_count, "class_1": class_1_count}

if __name__ == "__main__":
    # Configuration
    config = {
        "description_file_path": os.environ.get("VAPOR_DESCRIPTIONS", "data/raw/user_descriptions_rounds.xlsx"),
        "tweets_file_path": os.environ.get("VAPOR_TWEETS", "data/raw/tweets_no_duplicates.csv"),
        "cache_path": "preprocessed_data.pkl",  # Path to cached preprocessed data
    }

    # Parameter grid for grid search
    #param_grid = {
    #    "K": [20],
    #    "M1": [20],
    #    "M2": [20],
    #    "desired_users": [50],
    #    "other_tweets_sampling_method": ["random", "similar_topics", "followees", "followers", "both"],  # Other tweets sampling methods
    #    "user_sampling_method": ["random", "top_active", "recent_activity"],  # User sampling methods
    #}

    param_grid = {
        "K": [20],
        "M1": [20],
        "M2": [20],
        "desired_users": [50],
        "other_tweets_sampling_method": ["both"],  # Other tweets sampling methods, ["random", "similar_topics", "followees", "followers", "both"]
        "user_sampling_method": ["recent_activity"],  # User sampling methods, ["random", "top_active", "recent_activity"]
    }

    # Load data
    user_descriptions, tweets_data, _ = load_data(
        config["description_file_path"], config["tweets_file_path"], cache_path=config["cache_path"]
    )

    # Iterate over parameter combinations
    for params in itertools.product(*param_grid.values()):
        current_config = dict(zip(param_grid.keys(), params))
        output_file = f"datasets/dataset_{current_config}.json"

        if os.path.exists(output_file):
            print(f"Dataset for {current_config} already exists. Skipping.")
            continue

        print(f"\nCurrent Configuration: {current_config}")
        print(f"User Sampling Method: '{current_config['user_sampling_method']}', "
              f"Other Tweets Sampling Method: '{current_config['other_tweets_sampling_method']}'")

        # Sample users
        try:
            sampled_users = sample_users(
                tweets_data,
                user_descriptions,
                K=current_config["K"],
                M1=current_config["M1"],
                desired_users=current_config["desired_users"],
                method=current_config["user_sampling_method"],
            )
        except ValueError as e:
            print(f"Skipping configuration {current_config} due to sampling issue: {e}")
            continue

        print(f"Sampled {len(sampled_users)} users using the '{current_config['user_sampling_method']}' method.")

        dataset = []
        for user_id in sampled_users:
            try:
                # Get user tweets
                previous_tweets, future_tweets, future_tweets_data = get_user_tweets(
                    tweets_data, user_id, current_config["K"], current_config["M1"]
                )
                # Get others' tweets
                print(f"Sampling 'other tweets' using the '{current_config['other_tweets_sampling_method']}' method...")
                others_tweets = get_others_tweets(
                    tweets_data,
                    user_id,
                    current_config["M2"],
                    method=current_config["other_tweets_sampling_method"],
                    user_tweets=future_tweets_data,
                )

                # Combine and store data
                dataset.append({
                    "user_id": user_id,
                    "description": user_descriptions.get(user_id, ""),
                    "previous_tweets": [tweet["text"] for tweet in previous_tweets],  # Keep as a list of strings
                    "future_tweets": [{"tweet": tweet["text"], "ground_truth_label": 1} for tweet in future_tweets],
                    "others_tweets": [{"tweet": tweet, "ground_truth_label": 0} for tweet in others_tweets],  # Directly use strings
                })
            except ValueError as e:
                print(f"Skipping user {user_id}: {e}")
                continue

        # Save dataset
        save_dataset(dataset, output_file)

        # Compute and save class distribution
        class_distribution = compute_class_distribution(dataset)
        print(f"Class distribution for {current_config}: {class_distribution}")

        # Save class distributions to a file
        dist_file = "class_distributions.json"
        if os.path.exists(dist_file):
            with open(dist_file, "r") as f:
                all_distributions = json.load(f)
        else:
            all_distributions = {}

        all_distributions[str(current_config)] = class_distribution
        with open(dist_file, "w") as f:
            json.dump(all_distributions, f, indent=4)

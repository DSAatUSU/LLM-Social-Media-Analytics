import os
import pandas as pd
import json
from data_processing import load_data

# ✅ Paths
DESCRIPTION_FILE = os.environ.get("VAPOR_DESCRIPTIONS", "data/raw/user_descriptions_rounds.xlsx")
TWEETS_FILE = os.environ.get("VAPOR_TWEETS", "data/raw/tweets_no_duplicates.csv")
CACHE_PATH = "preprocessed_users.pkl"
USER_ANALYSIS_JSON = "user_analysis_nonbatch.json"
SELECTED_USERS_JSON = "selected_users2.json"
OUTPUT_JSON = "selected_users2_tweets_added.json"

# ✅ Load data
users_data, tweets_data = load_data(DESCRIPTION_FILE, TWEETS_FILE, cache_path=CACHE_PATH)

# ✅ Load final selected user IDs from analysis results (as strings)
with open(USER_ANALYSIS_JSON, "r") as f:
    user_analysis_results = json.load(f)

final_user_ids = {str(user["user_id"]) for user in user_analysis_results["user_results"]}
print(f"✅ Loaded final user IDs from analysis: {len(final_user_ids)}")

# ✅ Load selected users details explicitly as strings
selected_users_df = pd.read_json(SELECTED_USERS_JSON, dtype={"user_id": str})
selected_users_df = selected_users_df[selected_users_df["user_id"].isin(final_user_ids)]
print(f"✅ Matched selected user details: {len(selected_users_df)} users")

# ✅ Add tweets data to each user (excluding retweets starting with 'RT @')
selected_users_df["tweets"] = selected_users_df["user_id"].apply(
    lambda uid: [tweet["text"] for tweet in tweets_data.get(uid, [])
                 if not tweet["text"].startswith("RT @")]
)

# ✅ Save to new JSON file
selected_users_df.to_json(OUTPUT_JSON, orient="records", indent=4, force_ascii=False)

print(f"🎉 Final JSON saved successfully at '{OUTPUT_JSON}' with tweets (excluding retweets) for {len(selected_users_df)} users.")

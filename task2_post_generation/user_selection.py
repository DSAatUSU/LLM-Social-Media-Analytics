import pandas as pd
import os
import json
from data_processing import load_data

# ✅ File paths
DESCRIPTION_FILE = os.environ.get("VAPOR_DESCRIPTIONS", "data/raw/user_descriptions_rounds.xlsx")
TWEETS_FILE = os.environ.get("VAPOR_TWEETS", "data/raw/tweets_no_duplicates.csv")
CACHE_PATH = "preprocessed_users.pkl"
OUTPUT_FILE = "selected_users2.json"

# ✅ Load preprocessed user and tweet data
data = load_data(DESCRIPTION_FILE, TWEETS_FILE, cache_path=CACHE_PATH)
users_data, tweets_data = data

print(f"🔍 Total users loaded: {len(users_data)}")
print(f"🔍 Total users with tweets: {len(tweets_data)}")

# ✅ Convert users data to DataFrame
user_df = pd.DataFrame.from_dict(users_data, orient="index")
user_df.index.name = "user_id"
user_df.reset_index(inplace=True)

# ✅ Extract years from tweet timestamps
all_tweets = []
for uid, tweets in tweets_data.items():
    for tweet in tweets:
        year = None
        try:
            if isinstance(tweet["tweet_created_at"], str):
                year = int(tweet["tweet_created_at"].split()[-1])
        except:
            pass
        all_tweets.append({"user_id": uid, "year": year})

activity_df = pd.DataFrame(all_tweets)
print(f"📅 Total tweets parsed for year filtering: {len(activity_df)}")
print("📅 Sample extracted years:", activity_df["year"].dropna().unique()[:10])

# ✅ Filter users active between 2018–2020
active_users = activity_df[activity_df["year"].between(2018, 2020)]["user_id"].unique()
print(f"✅ Users active between 2018–2020: {len(active_users)}")

# ✅ Initial filter based on activity
filtered_users = user_df[user_df["user_id"].isin(active_users)]
print(f"📊 Users after year filtering: {len(filtered_users)}")

# ✅ Apply additional filters
filtered_users = filtered_users[
    (filtered_users["followers_count"] > 500) &
    (filtered_users["following"] > 100) &
    (filtered_users["description"].str.len() > 20) &
    (filtered_users["verified"] == True)
]
print(f"📊 Users after applying all filters (before tweet count): {len(filtered_users)}")

# ✅ Add tweet count
filtered_users["total_tweets"] = filtered_users["user_id"].apply(lambda uid: len(tweets_data.get(uid, [])))
filtered_users = filtered_users[filtered_users["total_tweets"] > 200]
print(f"✅ Final number of users after tweet count filter: {len(filtered_users)}")

# ✅ Sample users if more than 100, else keep all
# if len(filtered_users) >= 100:
#     selected_users = filtered_users.sample(n=100, random_state=42)
# else:
#     selected_users = filtered_users
#     print("⚠️ Less than 100 users available — using all available.")

# ✅ Use all available users after filtering
selected_users = filtered_users
print(f"✅ Using all {len(selected_users)} available users.")

# ✅ Save final users to JSON
selected_users.to_json(OUTPUT_FILE, orient="records", indent=4)
print(f"✅ Saved {len(selected_users)} users to {OUTPUT_FILE}")

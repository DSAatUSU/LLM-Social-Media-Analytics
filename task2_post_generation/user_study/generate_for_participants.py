import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # task2 model_api.py
from model_api import initialize_model, generate_tweets
from model_api_batch import generate_tweets_batch, retrieve_batch_results  # user-study variant: 8 posts per user

# Parameters
num_generated_tweets = 8
temperature = 0.7

# Your model API keys
gemini_api_key = os.environ.get("GEMINI_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Participants file (not distributed): one record per consenting participant with
# user_id, description, followers_count, following_count and their recent posts in "tweets"
with open("user_study_participants.json", "r") as f:
    real_users = json.load(f)

# Models setup
llm_models = {
    "gemini": ["gemini-1.5-pro-002"],
    "ollama": ["llama3.2"],
    "deepseek": ["deepseek"]
}

batch_model_name = "gpt-4o"

# Generate tweets and append directly to existing user data
for user in real_users:
    user_context = {
        "description": user["description"],
        "followers_count": user["followers_count"],
        "following_count": user["following_count"]
    }

    input_data = {
        "sampled_tweets": user["tweets"],
        "user_context": user_context
    }

    user["generated_tweets"] = {}

    for model_type, model_names in llm_models.items():
        for model_name in model_names:
            print(f"Generating tweets for user {user['user_id']} using {model_name}...")

            model_resources = initialize_model(
                model_type=model_type,
                model_name=model_name,
                gemini_api_key=gemini_api_key,
                deepseek_api_key=deepseek_api_key
            )

            generated_tweets = generate_tweets(
                input_data=input_data,
                model_name=model_name,
                model_type=model_type,
                num_generated_tweets=num_generated_tweets,
                temperature=temperature,
                model_resources=model_resources,
                sampling_number=len(user["tweets"])
            )

            user["generated_tweets"][model_name] = generated_tweets

# GPT-4o Batch Processing
batch_requests = []
for user in real_users:
    batch_requests.append({
        "custom_id": user["user_id"],
        "input_data": {
            "sampled_tweets": user["tweets"],
            "user_context": {
                "description": user["description"],
                "followers_count": user["followers_count"],
                "following_count": user["following_count"]
            }
        }
    })

batch_id = generate_tweets_batch(batch_requests, batch_model_name, openai_api_key)
print(f"Batch ID for GPT-4o: {batch_id}")

batch_results = retrieve_batch_results(batch_id, openai_api_key)

# Adding batch results
gpt4o_results = {
    res["custom_id"]: [tweet.strip() for tweet in res["response"]["body"]["choices"][0]["message"]["content"].split("\n") if tweet.strip()] 
    for res in batch_results if res["response"]["status_code"] == 200
}

for user in real_users:
    user["generated_tweets"][batch_model_name] = gpt4o_results.get(user["user_id"], [])

# Save results to a new JSON file
with open("user_study_participants_with_generated_tweets.json", "w", encoding="utf-8") as f:
    json.dump(real_users, f, indent=4)

print("Tweets generation completed and results saved to user_study_participants_with_generated_tweets.json.")

import json
import time
import requests
from openai import OpenAI

def generate_tweets_batch(batch_requests, model_name, api_key):
    client = OpenAI(api_key=api_key)

    # Prepare batch prompts
    batch_prompts = []
    for req in batch_requests:
        sampled_tweets = req["input_data"]["sampled_tweets"]
        user_context = req["input_data"]["user_context"]
        prompt = (
            f"We have collected {len(sampled_tweets)} tweets from a user on Twitter, included below:\n\n"
            f"{chr(10).join(sampled_tweets)}\n\n"
            "Below, you can also find the user-written bio (description) as well as the number of followings and followers of the user:\n\n"
            f"User Bio: {user_context['description']}\n"
            f"Number of Followings: {user_context['following_count']}\n"
            f"Number of Followers: {user_context['followers_count']}\n\n"
            f"Now, generate exactly 10 tweets that the user could have posted on their Twitter account. "
            "Each tweet should not exceed 280 characters and must be formatted as:\n"
            "1. <tweet_content>\n"
            "2. <tweet_content>\n"
            "...\n"
            "Ensure all 10 tweets are included, without extra spacing or missing tweets."
        )
        batch_prompts.append({
            "custom_id": req["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(10 * 70, 16000),  # Adjust as necessary
                "temperature": 0.7
            }
        })

    # Write batch request to a jsonl file
    batch_filename = "batch_requests.jsonl"
    with open(batch_filename, "w") as f:
        for prompt in batch_prompts:
            f.write(json.dumps(prompt) + "\n")

    # Submit batch request
    with open(batch_filename, "rb") as batch_file:
        batch_response = client.files.create(file=batch_file, purpose="batch")

    file_id = batch_response.id

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch.id  # Return the batch ID to retrieve later


def retrieve_batch_results(batch_id, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    url = f"https://api.openai.com/v1/batches/{batch_id}"

    while True:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        batch_info = response.json()
        status = batch_info["status"]
        print(f"Batch status: {status}")

        if status == "completed":
            output_file_id = batch_info.get("output_file_id")
            if output_file_id:
                break
            else:
                error_file_id = batch_info.get("error_file_id")
                if error_file_id:
                    print(f"Retrieving error details from file ID: {error_file_id}")

                    error_file_url = f"https://api.openai.com/v1/files/{error_file_id}/content"
                    error_response = requests.get(error_file_url, headers=headers)
                    error_response.raise_for_status()

                    error_details = [json.loads(line) for line in error_response.text.strip().split("\n")]

                    print("\n⚠️ Error Details from Batch Processing:")
                    for detail in error_details:
                        print(json.dumps(detail, indent=2))

                    raise RuntimeError("Batch failed. See error details above.")
                else:
                    raise RuntimeError("Batch completed but no output or error file ID returned.")
        elif status in ["failed", "cancelled"]:
            raise RuntimeError(f"Batch {batch_id} {status}. Details: {batch_info.get('errors')}")

        time.sleep(30)  # check again every 30 seconds

    file_url = f"https://api.openai.com/v1/files/{output_file_id}/content"
    file_response = requests.get(file_url, headers=headers)
    file_response.raise_for_status()

    with open("batch_results.jsonl", "w", encoding="utf-8") as f:
        f.write(file_response.text)

    print("✅ Saved batch_results.jsonl successfully!")

    results = [json.loads(line) for line in file_response.text.strip().split("\n")]

    return results

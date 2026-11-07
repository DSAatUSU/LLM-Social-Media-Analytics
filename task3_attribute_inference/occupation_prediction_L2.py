import json
import random
import re
import os
import concurrent.futures
from sklearn.metrics import accuracy_score, f1_score
from model_api_occupation import initialize_model, classify_occupation

def get_unique_occupations(json_file):
    with open(json_file, 'r') as f:
        users = json.load(f)
    return sorted(set(user["occupation_L2"] for user in users if user["occupation_L2"]))

def create_prompt(user, occupations, sampled_tweets):
    occupations_str = ', '.join(occupations)
    return (
        f"Your task is to infer the primary occupational category of a Twitter/X user based on their tweets.\n\n"
        f"You will be provided with exactly {len(sampled_tweets)} randomly selected tweets from this user.\n\n"
        f"Assign this user to exactly one of the following occupational categories, based on the content of their tweets:\n\n"
        f"{occupations_str}\n\n"
        f"Here are the user's sampled tweets:\n{sampled_tweets}\n\n"
        "Respond strictly and exclusively in the following JSON format:\n"
        "{\n"
        "  \"occupation_category\": \"<one category from the above list>\"\n"
        "}\n\n"
        "Do not include any other text, explanation, or formatting."
    )

def calculate_metrics(results, models_info, categories):
    y_true = [r["ground_truth"] for r in results]
    metrics = {}
    for model in models_info:
        y_pred = [r["predictions"].get(model, "Uncategorized") for r in results]
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(
            y_true, y_pred, average='weighted', labels=categories, zero_division=0
        )
        metrics[model] = {"accuracy": accuracy, "f1_score": f1}
    return metrics

def extract_prediction(prediction_raw):
    match = re.search(r'```json\s*(\{.*?\})\s*```', prediction_raw, re.DOTALL)
    json_content = match.group(1) if match else prediction_raw
    try:
        parsed_json = json.loads(json_content)
        if isinstance(parsed_json, dict):
            prediction = parsed_json.get('occupation_category', "Uncategorized")
        elif isinstance(parsed_json, list) and parsed_json:
            prediction = parsed_json[0]
        else:
            prediction = "Uncategorized"
        if isinstance(prediction, dict):
            prediction = prediction.get('occupation_category', "Uncategorized")
    except (json.JSONDecodeError, KeyError, TypeError):
        prediction = "Uncategorized"
    return prediction

def main(input_json, output_json, num_tweets=50, num_users=None, models_info=None, timeout_seconds=120):
    occupations = get_unique_occupations(input_json)

    with open(input_json, 'r') as f:
        users = json.load(f)

    if num_users:
        users = users[:num_users]

    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            results = json.load(f).get('results', [])
    else:
        results = []

    processed_users = {r["user_id"]: r for r in results}
    any_processing_done = False

    for user in users:
        user_id = user["user_id"]
        tweets_sample = random.sample(user["tweets"], min(num_tweets, len(user["tweets"])))
        prompt = create_prompt(user, occupations, tweets_sample)

        for model_type, details in models_info.items():
            if user_id in processed_users and model_type in processed_users[user_id]["predictions"]:
                print(f"⏭️ Skipping user {user_id} ({model_type})")
                continue

            model_resources = initialize_model(model_type, **details['init_args'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(classify_occupation, prompt, details['model_name'], model_type, model_resources)
                try:
                    prediction_raw = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    print(f"⚠️ Timeout for user {user_id} ({model_type}), assigning 'Timeout'.")
                    prediction_raw = json.dumps({"occupation_category": "Timeout"})

            prediction = extract_prediction(prediction_raw)

            processed_users.setdefault(user_id, {
                "user_id": user_id,
                "name": user.get("name"),
                "description": user.get("description"),
                "used_tweets": tweets_sample,
                "ground_truth": user["occupation_L2"],
                "predictions": {}
            })
            processed_users[user_id]["predictions"][model_type] = prediction
            any_processing_done = True

            results = list(processed_users.values())
            metrics = calculate_metrics(results, models_info, occupations)

            with open(output_json, 'w') as f:
                json.dump({"results": results, "metrics": metrics}, f, indent=2)

            print(f"✅ Processed {model_type} for user {user_id}: {prediction}")

    if not any_processing_done:
        print("🚩 No new processing was necessary.")

    metrics = calculate_metrics(list(processed_users.values()), models_info, occupations)
    with open(output_json, 'w') as f:
        json.dump({"results": list(processed_users.values()), "metrics": metrics}, f, indent=2)

    print(f"🎉 Final metrics: {metrics}")

if __name__ == "__main__":
    categories = get_unique_occupations("selected_users2_fully_labeled.json")
    models_info = {
        "openai": {"model_name": "gpt-4o", "init_args": {"api_key": os.environ.get("OPENAI_API_KEY")}},
        "gemini": {"model_name": "gemini-1.5-pro-002", "init_args": {"gemini_api_key": os.environ.get("GEMINI_API_KEY")}},
        "ollama": {"model_name": "llama3.2", "init_args": {"model_name": "llama3.2"}},
        "deepseek": {"model_name": "deepseek-chat","init_args": {"deepseek_api_key": os.environ.get("DEEPSEEK_API_KEY")}},
        #"bert": {"model_name": "fine_tuned_bert_demographic", "init_args": {"fine_tuned_path": "fine_tuned_bert_demographic"}}
    }

    main(
        input_json="selected_users2_fully_labeled.json",
        output_json="selected_users_occupations_predicted_L2.json",
        num_tweets=50,
        num_users=None,
        models_info=models_info,
        timeout_seconds=120
    )

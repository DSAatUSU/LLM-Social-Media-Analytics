"""Task I - Authorship verification with LLMs (and BERT).

For every dataset JSON in --datasets and every model in --models, each candidate
post (the user's own future posts + posts from other users) is classified with
Prompt 1 of the paper ("was this post written by the user?"). Results are written
per (model, dataset) to --results and aggregate metrics are appended to
--metrics. Progress is checkpointed so an interrupted run can be resumed.

Example:
    export OPENAI_API_KEY=... GEMINI_API_KEY=... DEEPSEEK_API_KEY=...
    python run_classification.py --models gpt-4 gemini-1.5-pro-002 deepseek llama3.2 bert

API keys are read from the environment (see .env.example in the repo root).
"""
import argparse
import json
import os

import pandas as pd

from evaluation import evaluate_results
from model_api import classify_tweets, initialize_model
from prompt import prepare_prompt


def model_type_for(model_name):
    """Map a model name to the backend used in model_api.py."""
    name = model_name.lower()
    if name.startswith("llama"):
        return "ollama"
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith("deepseek"):
        return "deepseek"
    if name == "bert":
        return "bert"
    return "openai"  # gpt-4, gpt-3.5-turbo, ...


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="datasets", help="folder with dataset_*.json files (from generate_datasets.py)")
    ap.add_argument("--results", default="results", help="output folder for per-(model, dataset) predictions")
    ap.add_argument("--metrics", default="evaluation_metrics.csv", help="CSV to append aggregate metrics to")
    ap.add_argument("--checkpoint", default="progress_checkpoint.json")
    ap.add_argument("--models", nargs="+", default=["gpt-4"],
                    help="any of: gpt-4 gpt-3.5-turbo gemini-1.5-pro-002 deepseek llama3.2 bert")
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    progress = load_json(args.checkpoint) if os.path.exists(args.checkpoint) else {}

    dataset_files = sorted(f for f in os.listdir(args.datasets) if f.endswith(".json"))
    if not dataset_files:
        raise SystemExit(f"No dataset JSON files found in '{args.datasets}'.")

    for dataset_file in dataset_files:
        dataset = load_json(os.path.join(args.datasets, dataset_file))

        for model_name in args.models:
            model_type = model_type_for(model_name)
            print(f"\n=== {model_name} ({model_type}) on {dataset_file} ===")

            model_resources = initialize_model(
                model_type=model_type,
                model_name=model_name if model_type == "ollama" else None,
                api_key=os.environ.get("OPENAI_API_KEY") if model_type == "openai" else None,
                gemini_api_key=os.environ.get("GEMINI_API_KEY") if model_type == "gemini" else None,
                deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY") if model_type == "deepseek" else None,
            )

            key = f"{model_name}_{dataset_file}"
            progress.setdefault(key, {"completed_users": []})

            output_file = os.path.join(args.results, f"results_{model_name}_{dataset_file}")
            results = load_json(output_file) if os.path.exists(output_file) else []

            for entry in dataset:
                user_id = entry["user_id"]
                if user_id in progress[key]["completed_users"]:
                    continue

                description = entry["description"]
                previous_tweets = entry["previous_tweets"]
                tweets_to_classify = entry["future_tweets"] + entry["others_tweets"]

                user_results, success = [], True
                for tweet_data in tweets_to_classify:
                    tweet_text = tweet_data["tweet"]
                    prompt = prepare_prompt(description, previous_tweets, tweet_text)
                    response = classify_tweets(prompt, tweet_text, model_name, model_type, model_resources)
                    if response is None:
                        success = False
                        print(f"  user {user_id}: API error, will be retried on the next run")
                        break
                    user_results.append({
                        "tweet": tweet_text,
                        "ground_truth_label": tweet_data["ground_truth_label"],
                        "model_prediction": response["prediction"],
                    })

                if success:
                    results.append({"user_id": user_id, "results": user_results})
                    progress[key]["completed_users"].append(user_id)
                    save_json(args.checkpoint, progress)
                    save_json(output_file, results)

            missing = {e["user_id"] for e in dataset} - {r["user_id"] for r in results}
            if missing:
                print(f"  {len(missing)} users still missing for {model_name}; rerun to retry them.")

            metrics = evaluate_results(output_file)
            row = {
                "model_name": model_name,
                "dataset_file": dataset_file,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "classification_report": metrics["classification_report"].replace("\n", " "),
            }
            pd.DataFrame([row]).to_csv(args.metrics, mode="a", header=not os.path.exists(args.metrics), index=False)
            print(f"  weighted metrics appended to {args.metrics}")

    print("\nDone. Run clean_metrics.py to flatten the classification reports.")


if __name__ == "__main__":
    main()

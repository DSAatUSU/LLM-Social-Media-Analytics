# use_authorship_baseline.py
import os
import json
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow_text  # Required for newer USE models

# Load Google's Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

def load_json_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_embeddings(text_list):
    return embed(text_list).numpy()

def authorship_verification(entry, similarity_threshold=0.7):
    previous_tweets = entry.get("previous_tweets", [])
    future_tweets = entry.get("future_tweets", [])
    others_tweets = entry.get("others_tweets", [])

    if not previous_tweets or not future_tweets:
        return None, None

    author_embedding = np.mean(compute_embeddings(previous_tweets), axis=0).reshape(1, -1)

    test_tweets = [tweet["tweet"] for tweet in future_tweets + others_tweets]
    true_labels = [1] * len(future_tweets) + [0] * len(others_tweets)

    test_embeddings = compute_embeddings(test_tweets)

    similarities = cosine_similarity(test_embeddings, author_embedding).flatten()

    pred_labels = (similarities >= similarity_threshold).astype(int)

    return true_labels, pred_labels

def evaluate_dataset(json_path, tweet_method, user_method, model_name="use_cosine"):
    data = load_json_dataset(json_path)
    all_true, all_pred = [], []

    for entry in data:
        result = authorship_verification(entry)
        if result is None:
            continue

        true_labels, pred_labels = result

        if len(true_labels) == 0 or len(pred_labels) == 0:
            continue

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

    if all_true:
        accuracy = accuracy_score(all_true, all_pred)
        precision, recall, f1_score, support = precision_recall_fscore_support(all_true, all_pred, average='binary', zero_division=0)
        precision_0, recall_0, f1_score_0, support_0 = precision_recall_fscore_support(all_true, all_pred, average='binary', pos_label=0, zero_division=0)
        macro_avg_precision, macro_avg_recall, macro_avg_f1, _ = precision_recall_fscore_support(all_true, all_pred, average='macro', zero_division=0)
        weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, _ = precision_recall_fscore_support(all_true, all_pred, average='weighted', zero_division=0)

        result = {
            "model_name": model_name,
            "other_tweets_sampling_method": tweet_method,
            "user_sampling_method": user_method,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "precision_0": precision_0,
            "recall_0": recall_0,
            "f1_score_0": f1_score_0,
            "precision_1": precision,
            "recall_1": recall,
            "f1_score_1": f1_score,
            "support_1": support,
            "macro_avg_precision": macro_avg_precision,
            "macro_avg_recall": macro_avg_recall,
            "macro_avg_f1": macro_avg_f1,
            "weighted_avg_precision": weighted_avg_precision,
            "weighted_avg_recall": weighted_avg_recall,
            "weighted_avg_f1": weighted_avg_f1
        }

        return result
    else:
        return None

if __name__ == "__main__":
    dataset_dir = "../datasets"
    sampling_methods = ['both', 'followees', 'followers', 'random', 'similar_topics']
    user_sampling_methods = ['random', 'recent_activity', 'top_active']

    results = []

    for tweet_method in sampling_methods:
        for user_method in user_sampling_methods:
            filename = f"dataset_{{'K': 20, 'M1': 20, 'M2': 20, 'desired_users': 50, 'other_tweets_sampling_method': '{tweet_method}', 'user_sampling_method': '{user_method}'}}.json"
            json_path = os.path.join(dataset_dir, filename)
            if os.path.exists(json_path):
                result = evaluate_dataset(json_path, tweet_method, user_method)
                if result:
                    results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv("use_cosine_baseline_evaluation_metrics.csv", index=False)
    print("CSV file saved: use_cosine_baseline_evaluation_metrics.csv")

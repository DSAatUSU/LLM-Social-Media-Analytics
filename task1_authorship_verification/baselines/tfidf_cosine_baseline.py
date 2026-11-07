# tfidf_cosine_baseline.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_json_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_with_cosine_similarity(user_profile, test_tweets):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=1, max_df=0.95)

    combined = [user_profile] + test_tweets
    tfidf_matrix = vectorizer.fit_transform(combined)

    user_vector = tfidf_matrix[0]
    tweet_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(tweet_vectors, user_vector)

    threshold = np.mean(similarities)
    preds = (similarities.flatten() >= threshold).astype(int)

    return preds

def evaluate_dataset(json_path, tweet_method, user_method, model_name="tfidf_cosine"):
    data = load_json_dataset(json_path)
    all_labels, all_preds = [], []

    for entry in data:
        previous_tweets = entry.get("previous_tweets", [])
        future_tweets = entry.get("future_tweets", [])
        others_tweets = entry.get("others_tweets", [])

        if not previous_tweets or not future_tweets:
            continue

        user_profile = " ".join(previous_tweets)

        test_tweets = [tweet["tweet"] for tweet in future_tweets + others_tweets]
        labels = [1] * len(future_tweets) + [0] * len(others_tweets)

        preds = predict_with_cosine_similarity(user_profile, test_tweets)

        all_labels.extend(labels)
        all_preds.extend(preds)

    if all_labels:
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_score, support = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        precision_0, recall_0, f1_score_0, support_0 = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=0, zero_division=0)
        macro_avg_precision, macro_avg_recall, macro_avg_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

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
    df_results.to_csv("tfidf_cosine_baseline_evaluation_metrics.csv", index=False)
    print("CSV file saved: tfidf_cosine_baseline_evaluation_metrics.csv")

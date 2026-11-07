# stylometry_baseline.py (robust tokenisation, safe handling, confusion matrix)
import os
import json
import random
import numpy as np
import pandas as pd
from faststylometry.corpus import Corpus
from faststylometry.burrows_delta import calculate_burrows_delta
from faststylometry.probability import calibrate, predict_proba
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re

def load_json_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_global_negative_pool(dataset):
    pool = []
    for entry in dataset:
        pool.extend(tweet["tweet"] for tweet in entry.get("others_tweets", []))
    return pool

def custom_tokenise(text):
    pattern = re.compile(r'(https?://\S+|[@#]\w+|\w+|[\U00010000-\U0010FFFF])', re.UNICODE)
    try:
        return pattern.findall(text.lower())
    except:
        return []

def corpus_has_valid_token_overlap(train_corpus):
    author_tokens = [set(tokens) for tokens in train_corpus.tokens if tokens]
    if len(author_tokens) < 2:
        return False
    overlap = set.intersection(*author_tokens)
    return len(overlap) > 0

def build_corpora(entry, global_negative_pool, num_negatives=3):
    train_corpus = Corpus()
    test_corpus = Corpus()

    user_id = str(entry["user_id"])
    previous_tweets = entry.get("previous_tweets", [])
    future_tweets = entry.get("future_tweets", [])
    others_tweets = entry.get("others_tweets", [])

    if not previous_tweets or not future_tweets:
        return None, None, None

    train_corpus.add_book(user_id, f"{user_id}_known", " ".join(previous_tweets))

    if not others_tweets:
        others_tweets = [{"tweet": tweet} for tweet in random.sample(global_negative_pool, min(num_negatives, len(global_negative_pool)))]

    for idx, tweet in enumerate(others_tweets):
        train_corpus.add_book(f"other_{idx}", f"other_{idx}_text", tweet["tweet"])

    labels = []

    for idx, tweet in enumerate(future_tweets):
        book_id = f"{user_id}_future_{idx}"
        test_corpus.add_book(book_id, book_id, tweet["tweet"])
        labels.append(1)

    for idx, tweet in enumerate(others_tweets):
        book_id = f"other_{user_id}_{idx}"
        test_corpus.add_book(book_id, book_id, tweet["tweet"])
        labels.append(0)

    return train_corpus, test_corpus, labels

def fallback_predict(train_corpus, test_corpus, user_id):
    try:
        delta_df = calculate_burrows_delta(train_corpus, test_corpus)
        if delta_df.empty or np.isnan(delta_df.values).all():
            return None
        threshold = np.nanmean(delta_df.values)
        preds = (np.nanmin(delta_df.values, axis=0) < threshold).astype(int)
        return preds
    except:
        return None

def safe_calibrate_and_predict(train_corpus, test_corpus, user_id, counters):
    train_corpus.tokenise(custom_tokenise)
    test_corpus.tokenise(custom_tokenise)

    if not corpus_has_valid_token_overlap(train_corpus):
        counters['no_overlap'] += 1
        return fallback_predict(train_corpus, test_corpus, user_id)

    try:
        calibrate(train_corpus)
        delta_df = calculate_burrows_delta(train_corpus, test_corpus)
        prob_df = predict_proba(train_corpus, test_corpus)
        probs = prob_df.loc[user_id] if user_id in prob_df.index else prob_df.iloc[0]
        return (probs >= 0.5).astype(int).values

    except ValueError as e:
        if "at least 2 classes" in str(e):
            counters['single_class'] += 1
        return fallback_predict(train_corpus, test_corpus, user_id)

def evaluate_dataset(json_path):
    data = load_json_dataset(json_path)
    global_negative_pool = create_global_negative_pool(data)
    results, counters = [], {'no_overlap': 0, 'single_class': 0}

    all_labels = []
    all_preds = []

    print(f"\nEvaluating {os.path.basename(json_path)}")
    for entry in data:
        user_id = entry["user_id"]
        corpora = build_corpora(entry, global_negative_pool)

        if corpora is None:
            continue

        train_corpus, test_corpus, labels = corpora
        preds = safe_calibrate_and_predict(train_corpus, test_corpus, user_id, counters)

        if preds is None or len(preds) != len(labels):
            continue

        all_labels.extend(labels)
        all_preds.extend(preds)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        results.append({"user_id": user_id, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})

    if results:
        df_results = pd.DataFrame(results)
        avg = df_results.mean(numeric_only=True)
        print("\nAverage Metrics Across Users:")
        print(f"Accuracy={avg['accuracy']:.3f}, Precision={avg['precision']:.3f}, Recall={avg['recall']:.3f}, F1={avg['f1_score']:.3f}")

    print("\n⚠️ Warnings summary:")
    print(f"- No valid token overlap: {counters['no_overlap']} users")
    print(f"- Calibration single-class failures: {counters['single_class']} users")

    if all_labels and all_preds:
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix (Overall):")
        print("True Negative | False Positive")
        print("False Negative | True Positive")
        print(cm)

if __name__ == "__main__":
    random.seed(42)
    dataset_dir = "../../datasets"
    for filename in sorted(os.listdir(dataset_dir)):
        if filename.endswith(".json"):
            evaluate_dataset(os.path.join(dataset_dir, filename))

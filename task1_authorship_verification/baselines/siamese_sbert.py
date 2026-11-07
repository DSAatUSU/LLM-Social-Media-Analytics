"""Siamese network with SBERT (all-MiniLM-L6-v2) sentence embeddings for authorship verification.

Same training/testing pair files as the Random-Forest baseline (random_forest/build_*_pairs.py).
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from collections import Counter
from sentence_transformers import SentenceTransformer
import torch

# Define paths
DATASET_FOLDER = "."
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "random_forest/training_sets")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "random_forest/testing_sets")
RESULTS_CSV = os.path.join(DATASET_FOLDER, "siamese_sbert_evaluation_metrics.csv")

# Load SBERT embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Siamese network with cosine similarity

def create_siamese_model(embedding_dim):
    input_shape = (embedding_dim,)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    cosine_similarity = Lambda(
        lambda tensors: tf.reduce_sum(
            tf.multiply(tf.nn.l2_normalize(tensors[0], axis=1),
                        tf.nn.l2_normalize(tensors[1], axis=1)), axis=1, keepdims=True)
    )([input_a, input_b])

    output = Dense(1, activation='sigmoid')(cosine_similarity)
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

# Load dataset from CSV
def load_csv_dataset(path):
    df = pd.read_csv(path)
    return df['Tweet1'].tolist(), df['Tweet2'].tolist(), df['Label'].tolist()

# Embed tweets using SBERT
def embed_tweets(tweets):
    return sbert_model.encode(tweets, batch_size=64, convert_to_numpy=True)

# Evaluate model and calculate metrics similar to gzip baseline
def evaluate_dataset(train_path, test_path, apply_smote=True):
    print(f"\n🔹 Processing: {os.path.basename(train_path)}")

    # Load data
    train_tweet1, train_tweet2, y_train = load_csv_dataset(train_path)
    test_tweet1, test_tweet2, y_test = load_csv_dataset(test_path)

    # Embeddings
    X1_train = embed_tweets(train_tweet1)
    X2_train = embed_tweets(train_tweet2)
    X1_test = embed_tweets(test_tweet1)
    X2_test = embed_tweets(test_tweet2)

    # Prepare data for SMOTE
    X_train = np.concatenate([X1_train, X2_train], axis=1)

    # Apply SMOTE if necessary
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        split_dim = X1_train.shape[1]
        X1_train, X2_train = X_train[:, :split_dim], X_train[:, split_dim:]

    # Create and train model
    model = create_siamese_model(X1_train.shape[1])
    model.fit([X1_train, X2_train], np.array(y_train), epochs=20,
              batch_size=64, validation_data=([X1_test, X2_test], np.array(y_test)), verbose=2)

    # Prediction
    preds = (model.predict([X1_test, X2_test]) >= 0.5).astype(int).flatten()

    # Metrics
    accuracy = accuracy_score(y_test, preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    precision_0, recall_0, f1_0, _ = precision_recall_fscore_support(y_test, preds, average='binary', pos_label=0, zero_division=0)
    macro_avg_precision, macro_avg_recall, macro_avg_f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)

    result = {
        "model_name": "SIAMESE_SBERT",
        "dataset": os.path.basename(test_path),
        "accuracy": accuracy,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_score_0": f1_0,
        "precision_1": precision,
        "recall_1": recall,
        "f1_score_1": f1,
        "macro_avg_precision": macro_avg_precision,
        "macro_avg_recall": macro_avg_recall,
        "macro_avg_f1": macro_avg_f1,
        "weighted_avg_precision": weighted_avg_precision,
        "weighted_avg_recall": weighted_avg_recall,
        "weighted_avg_f1": weighted_avg_f1
    }

    return result

# Main execution
if __name__ == "__main__":
    dataset_files = sorted([f for f in os.listdir(TRAIN_FOLDER) if f.startswith("training_dataset") and f.endswith(".csv")])

    results = []
    for train_file in dataset_files:
        test_file = train_file.replace("training_dataset", "testing_dataset")
        train_path = os.path.join(TRAIN_FOLDER, train_file)
        test_path = os.path.join(TEST_FOLDER, test_file)

        if not os.path.exists(test_path):
            continue

        apply_smote = any(k in train_file for k in ["followers", "followees", "both"])

        result = evaluate_dataset(train_path, test_path, apply_smote)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)

    print(f"\n✅ All datasets processed successfully! Results saved to {RESULTS_CSV}")

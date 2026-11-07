import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE  # ✅ Added SMOTE
import torch
from collections import Counter

# ✅ Disable parallel tokenization to prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Enable GPU for SBERT encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"Using device: {device}")

# ✅ Define dataset paths
DATASET_FOLDER = "."
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "training_sets")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "testing_sets")
RESULTS_FILE = os.path.join(DATASET_FOLDER, "rf_results.csv")

# ✅ Retrieve all training/testing dataset pairs
dataset_files = [f for f in os.listdir(TRAIN_FOLDER) if f.startswith("training_dataset")]

# ✅ Function to encode sentences in batches
def encode_sentences(sentences, batch_size=32):
    """Encodes a list of sentences efficiently in batches."""
    return sbert_model.encode(sentences, batch_size=batch_size, convert_to_numpy=True).tolist()

# ✅ Iterate over all dataset pairs
for train_file in dataset_files:
    # 🔹 Find the corresponding testing dataset
    test_file = train_file.replace("training_dataset", "testing_dataset")

    # If the testing dataset does not exist, skip
    if not os.path.exists(os.path.join(TEST_FOLDER, test_file)):
        print(f"⚠️ Skipping {train_file} as no matching test file found.")
        continue

    print(f"\n🔄 Processing {train_file} and {test_file}...")

    # ✅ Load datasets
    df_train = pd.read_csv(os.path.join(TRAIN_FOLDER, train_file))
    df_test = pd.read_csv(os.path.join(TEST_FOLDER, test_file))

    # ✅ Extract sampling method from filename
    if "followers" in train_file:
        sampling_method = "followers"
    elif "followees" in train_file:
        sampling_method = "followees"
    elif "both" in train_file:
        sampling_method = "both"
    else:
        sampling_method = "other"

    # ✅ Encode training & testing tweets
    train_sentences = df_train["Tweet1"].tolist() + df_train["Tweet2"].tolist()
    test_sentences = df_test["Tweet1"].tolist() + df_test["Tweet2"].tolist()

    train_embeddings = encode_sentences(train_sentences)
    test_embeddings = encode_sentences(test_sentences)

    # ✅ Assign embeddings correctly
    df_train["Embedding1"] = train_embeddings[:len(df_train)]
    df_train["Embedding2"] = train_embeddings[len(df_train):]

    df_test["Embedding1"] = test_embeddings[:len(df_test)]
    df_test["Embedding2"] = test_embeddings[len(df_test):]

    # ✅ Convert list embeddings to NumPy arrays
    X_train = np.array([np.concatenate((e1, e2)) for e1, e2 in zip(df_train["Embedding1"], df_train["Embedding2"])])
    y_train = df_train["Label"].values

    X_test = np.array([np.concatenate((e1, e2)) for e1, e2 in zip(df_test["Embedding1"], df_test["Embedding2"])])
    y_test = df_test["Label"].values

    # ✅ Apply SMOTE only if dataset is unbalanced and belongs to followers, followees, or both
    if sampling_method in ["followers", "followees", "both"]:
        print(f"⚖️ Applying SMOTE to balance dataset {train_file}...")

        # Check label distribution before SMOTE
        label_counts = Counter(y_train)
        print(f"Before SMOTE: {label_counts}")

        smote = SMOTE(random_state=42, sampling_strategy="auto")  # Balance dataset
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Check label distribution after SMOTE
        label_counts_after = Counter(y_train)
        print(f"After SMOTE: {label_counts_after}")

    # ✅ Fine-Tune Random Forest using Grid Search
    print("🔍 Fine-tuning Random Forest...")
    param_grid = {
        "n_estimators": [100, 200],  # Balanced for speed & accuracy
        "max_depth": [None, 10, 30],  # Controlled tree growth
        "min_samples_split": [2, 5],  # Flexibility vs. overfitting
        "min_samples_leaf": [1, 3]  # Prevent overfitting
    }
    rf_model = RandomForestClassifier(n_jobs=2, random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=3, n_jobs=2, verbose=1)
    grid_search.fit(X_train, y_train)

    # ✅ Get the best model
    best_rf_model = grid_search.best_estimator_
    print(f"✅ Best RF Model: {grid_search.best_params_}")

    # ✅ Predict all test pairs
    df_test["Predicted_Label"] = best_rf_model.predict(X_test)

    # ✅ Perform **Majority Voting** for each **Fixed Tweet**
    majority_vote_results = []
    grouped = df_test.groupby("Tweet2")

    for fixed_tweet, group in grouped:
        predicted_labels = group["Predicted_Label"].tolist()
        true_label = group["Label"].iloc[0]  # All pairs of this tweet have the same ground truth

        # **Majority Vote: Get the most common predicted label**
        majority_label = Counter(predicted_labels).most_common(1)[0][0]

        # Store (Predicted Majority Vote, True Label)
        majority_vote_results.append((majority_label, true_label))

    # ✅ Convert to NumPy array for evaluation
    majority_vote_results = np.array(majority_vote_results)

    # ✅ Compute Final Accuracy
    final_accuracy = accuracy_score(majority_vote_results[:, 1], majority_vote_results[:, 0])
    report = classification_report(majority_vote_results[:, 1], majority_vote_results[:, 0], output_dict=True, zero_division=0)

    print(f"\n🔹 Final Model Accuracy (Majority Voting): {final_accuracy:.4f}")

    # ✅ Format classification report for CSV
    formatted_report = classification_report(majority_vote_results[:, 1], majority_vote_results[:, 0], zero_division=0)

    # ✅ Store results in a DataFrame
    results_df = pd.DataFrame([{
        "model_name": "RandomForest",
        "dataset_file": train_file.replace("training_dataset_", "").replace(".csv", ""),
        "accuracy": final_accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "classification_report": formatted_report.replace("\n", " ")  # Flatten report for CSV
    }])

    # ✅ Append results immediately to the CSV file
    if os.path.exists(RESULTS_FILE):
        results_df.to_csv(RESULTS_FILE, mode="a", index=False, header=False, encoding="utf-8")
    else:
        results_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8")

    print(f"✅ Results for {train_file} appended to {RESULTS_FILE}")

print("\n✅ All datasets processed successfully! Check rf_results.csv for results.")

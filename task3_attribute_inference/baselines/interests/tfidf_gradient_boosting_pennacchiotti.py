"""
pennacchiotti_baseline.py  – Option B version with Progress Tracking
===============================================================
Replicates Pennacchiotti & Popescu (ICWSM 2011) text-only baseline
using an 80/20 train-test split *without* stratification. Class
imbalance is handled via sample-weighting. Reports accuracy,
weighted-F1, and balanced accuracy.

Run: python pennacchiotti_baseline.py
"""

import json
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from tqdm import tqdm

# ---------------------------------------------------------------------
# Paths
JSON_IN  = pathlib.Path("../../selected_users_interests_predicted.json")
JSON_OUT = pathlib.Path("selected_users_interests_with_pennacchiotti.json")

# ---------------------------------------------------------------------
# Load data
print("Loading data...")
data = json.loads(JSON_IN.read_text(encoding="utf-8"))
users = data["results"]

docs   = [" ".join(u["used_tweets"]).replace("\n", " ") for u in users]
labels = np.array([u["ground_truth"] for u in users])
indices = np.arange(len(labels))

# ---------------------------------------------------------------------
# 80/20 split  (no stratify)
print("Splitting dataset...")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    docs, labels, indices,
    test_size=0.20,
    stratify=None,
    random_state=42,
)

# ---------------------------------------------------------------------
# Vectorise tweets with TF-IDF 1–2-grams
print("Vectorizing tweets...")
vectoriser = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20_000,
    min_df=3,
    sublinear_tf=True,
)

Xtr = vectoriser.fit_transform(tqdm(X_train, desc="Vectorizing training data"))
Xte = vectoriser.transform(tqdm(X_test, desc="Vectorizing test data"))

# ---------------------------------------------------------------------
# Balanced sample weights
print("Computing sample weights...")
sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

# Gradient-Boosted Decision Tree (same params as paper)
print("Training Gradient Boosting classifier...")
gbdt = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    random_state=42,
)
gbdt.fit(Xtr.toarray(), y_train, sample_weight=sample_w)

print("Making predictions...")
preds = gbdt.predict(Xte.toarray())

# ---------------------------------------------------------------------
# Metrics
acc  = accuracy_score(y_test, preds)
wf1  = f1_score(y_test, preds, average="weighted")
bacc = balanced_accuracy_score(y_test, preds)

print(f"[Pennacchiotti Option B]  Acc={acc:.4f}  W-F1={wf1:.4f}  BalAcc={bacc:.4f}")

# ---------------------------------------------------------------------
# Update JSON (add predictions for test-set users)
print("Updating JSON...")
for i, p in zip(idx_test, preds):
    users[i].setdefault("predictions", {})["pennacchiotti_baseline"] = str(p)

data.setdefault("metrics", {})["pennacchiotti_baseline"] = {
    "split"              : "80/20 random (no stratify)",
    "accuracy"           : acc,
    "f1_score"           : wf1,
    "balanced_accuracy"  : bacc,
}

JSON_OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print("Done.")

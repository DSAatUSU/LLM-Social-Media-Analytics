import json
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# === Load and parse data ===
with open("../../../selected_users_interests_predicted.json", "r") as f:
    data = json.load(f)

users = data["results"]
texts = [" ".join(user["used_tweets"]) for user in users]
labels = [user["ground_truth"] for user in users]

# === Encode labels ===
label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
labels_idx = np.array([label_to_idx[label] for label in labels])

# === Filter out rare classes (those with < 2 samples) ===
label_counts = Counter(labels_idx)
valid_indices = [i for i, label in enumerate(labels_idx) if label_counts[label] >= 2]

X_texts = [texts[i] for i in valid_indices]
X_labels = [labels_idx[i] for i in valid_indices]
X_users = [users[i] for i in valid_indices]
indices = np.arange(len(X_users))

# === Text preprocessing ===
def preprocess(text):
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z\\s]", "", text.lower())
    return text

texts_clean = [preprocess(text) for text in X_texts]

# === Feature extraction ===
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts_clean)
y = np.array(X_labels)

# === Train-test split ===
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.3, random_state=42, stratify=y
)

# === Train classifier ===
classifier = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# === Predict on test set ===
y_pred = classifier.predict(X_test)

# === Insert predictions into user JSON ===
for i, pred_idx in zip(idx_test, y_pred):
    pred_label = idx_to_label[pred_idx]
    if "predictions" not in X_users[i]:
        X_users[i]["predictions"] = {}
    X_users[i]["predictions"]["text_baseline_lr"] = pred_label

# === Calculate metrics ===
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

# === Prepare updated output JSON ===
output_data = {
    "results": X_users,
    "metrics": {
        "text_baseline_lr": {
            "accuracy": round(accuracy, 4),
            "f1_score": round(weighted_f1, 4)
        }
    }
}

# === Save new JSON ===
output_path = "selected_users_interests_with_text_baseline_lr.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

# === Report ===
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Weighted F1-score: {weighted_f1:.4f}")
print(f"📄 Saved: {output_path}")
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[idx_to_label[i] for i in sorted(set(y))]))

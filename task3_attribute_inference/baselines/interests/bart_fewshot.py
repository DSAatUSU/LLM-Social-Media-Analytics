"""
bart_fewshot_classification.py – Few-Shot Classification Baseline with BART
============================================================================
Fine-tunes BART model on a small portion of the dataset (few-shot), applies oversampling to training set,
and evaluates on unseen test data. Metrics include accuracy and weighted-F1.

Run: python bart_fewshot_classification.py
"""

import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BartForSequenceClassification, BartTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler

# ---------------------------------------------------------------------
# Paths
json_path = "../../selected_users_interests_predicted.json"
output_json_path = "selected_users_interests_with_bart_fewshot.json"

# ---------------------------------------------------------------------
# Load JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

users = data['results']
texts = [" ".join(user["used_tweets"]).replace('\n', ' ').replace('\t', ' ') for user in users]
labels = [user["ground_truth"] for user in users]
unique_labels = list(set(labels))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

# ---------------------------------------------------------------------
# Dataset preparation
train_texts, eval_texts, train_labels, eval_labels, train_indices, eval_indices = train_test_split(
    texts, labels, range(len(labels)),
    test_size=0.2, random_state=42, stratify=None  # No stratification to avoid single-sample class error
)

# ---------------------------------------------------------------------
# Apply oversampling on training data
X_train_np = np.array(train_texts)
y_train_np = np.array(train_labels)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_np.reshape(-1, 1), y_train_np)
train_texts = X_resampled.ravel().tolist()
train_labels = y_resampled.tolist()

# ---------------------------------------------------------------------
# Tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")
model = BartForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli", num_labels=len(unique_labels),
    ignore_mismatched_sizes=True
)

class InterestDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = [label_to_id[label] for label in labels]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = InterestDataset(train_texts, train_labels)
eval_dataset = InterestDataset(eval_texts, eval_labels)

# ---------------------------------------------------------------------
# Training arguments
training_args = TrainingArguments(
    output_dir="./bart_fewshot_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=42
)

# ---------------------------------------------------------------------
# Metrics computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    wf1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "weighted_f1": wf1}

# ---------------------------------------------------------------------
# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# ---------------------------------------------------------------------
# Fine-tune BART
trainer.train()

# ---------------------------------------------------------------------
# Predict on evaluation set
predictions = trainer.predict(eval_dataset)
logits = predictions.predictions
if isinstance(logits, tuple):
    logits = logits[0]
pred_labels = [id_to_label[pred] for pred in np.argmax(logits, axis=-1)]

# ---------------------------------------------------------------------
# Update JSON with predictions for evaluation users only
for idx, pred_label in zip(eval_indices, pred_labels):
    users[idx].setdefault('predictions', {})['bart_fewshot_baseline'] = pred_label

# ---------------------------------------------------------------------
# Compute final metrics
accuracy = accuracy_score(eval_labels, pred_labels)
weighted_f1 = f1_score(eval_labels, pred_labels, average='weighted')

# ---------------------------------------------------------------------
# Save metrics to JSON
data['metrics']['bart_fewshot_baseline'] = {
    'accuracy': accuracy,
    'f1_score': weighted_f1,
    'note': 'Few-shot fine-tuned BART classification with oversampling on training set'
}

# ---------------------------------------------------------------------
# Save updated JSON
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Few-Shot BART Baseline Accuracy: {accuracy:.4f}")
print(f"Few-Shot BART Baseline Weighted F1-score: {weighted_f1:.4f}")

import json
import random
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

# Paths
json_path = "../../../selected_users_occupations_predicted_L2.json"
output_json_path = "selected_users_occupations_L2_with_hf_baseline.json"

# Load the existing JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

users = data['results']

# Define HuggingFace zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = list(set(user["ground_truth"] for user in users))

# Evaluate and predict for all users
true_labels = []
pred_labels = []

for idx, user in enumerate(users, 1):
    print(f"Processing user {idx}/{len(users)}: {user['name']} ({user['user_id']})")
    tweets = " ".join(user["used_tweets"]).replace('\n', ' ').replace('\t', ' ')
    prediction = classifier(tweets, candidate_labels)
    pred_label = prediction['labels'][0]

    # Update JSON with predictions
    user['predictions']['huggingface_baseline'] = pred_label

    pred_labels.append(pred_label)
    true_labels.append(user["ground_truth"])

# Calculate metrics
accuracy = accuracy_score(true_labels, pred_labels)
weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')

# Update JSON metrics
data['metrics']['huggingface_baseline'] = {
    'accuracy': accuracy,
    'f1_score': weighted_f1
}

# Save updated JSON
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"HuggingFace Baseline Accuracy: {accuracy:.4f}")
print(f"HuggingFace Baseline Weighted F1-score: {weighted_f1:.4f}")

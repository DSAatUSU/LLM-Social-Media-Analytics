"""
gp_occupation_L1_classification.py – GP Baseline for Occupation L1 (ACL 2015 methodology + oversampling)
=======================================================================================================
Gaussian Process (ARD Kernel) baseline replicating ACL 2015 paper method for Occupation Level 1 classification,
handling class imbalance via oversampling:
- Word2Vec embeddings (vector size=50)
- Spectral clustering (200 clusters as per ACL 2015)
- Gaussian Process with ARD kernel
- Random oversampling to handle class imbalance
- Evaluation: Accuracy, Weighted-F1, Balanced Accuracy

Run: python gp_occupation_L1_classification.py
"""

import json
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.cluster import SpectralClustering
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# Paths
json_input = "../../selected_users_occupations_predicted_L1.json"
json_output = "selected_users_occupations_L1_with_gp_baseline.json"

# Load existing data
with open(json_input, "r") as file:
    data = json.load(file)

users = data["results"]

# Extract texts and labels
texts = [" ".join(user["used_tweets"]) for user in users]
labels = [user["ground_truth"] for user in users]

# Preprocess texts
def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text

print("Preprocessing texts...")
texts_clean = [preprocess(text) for text in tqdm(texts)]

# Tokenize texts
print("Tokenizing texts...")
tokenized_texts = [word_tokenize(text) for text in tqdm(texts_clean)]

# Word2Vec embeddings
print("Training Word2Vec embeddings...")
embedding_model = Word2Vec(sentences=tokenized_texts, vector_size=50, window=5, min_count=5, epochs=50)

vocab = list(embedding_model.wv.index_to_key)
word_vectors = embedding_model.wv[vocab]

# Spectral clustering
print("Performing spectral clustering...")
num_clusters = 200  # As per ACL 2015 methodology
clustering_model = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
clusters = clustering_model.fit_predict(word_vectors)

word_to_cluster = dict(zip(vocab, clusters))

# Create cluster features
def create_cluster_features(tokenized_text):
    features = np.zeros(num_clusters)
    for word in tokenized_text:
        cluster_id = word_to_cluster.get(word)
        if cluster_id is not None:
            features[cluster_id] += 1
    return features

print("Creating cluster features...")
features = np.array([create_cluster_features(text) for text in tqdm(tokenized_texts)])

# Train-test split
print("Splitting dataset...")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    features, labels, np.arange(len(labels)), test_size=0.2, random_state=42
)

# Oversampling
print("Applying oversampling...")
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Gaussian Process Classifier with ARD kernel
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * \
         RBF(length_scale=np.ones(num_clusters), length_scale_bounds=(1e-1, 1e3)) + \
         WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-3, 1e1))

gp_classifier = GaussianProcessClassifier(kernel=kernel, optimizer='fmin_l_bfgs_b',
                                          n_restarts_optimizer=1, random_state=42)

# Train GP model
print("Training Gaussian Process classifier...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    gp_classifier.fit(X_train_resampled, y_train_resampled)

# Predict on test set
print("Making predictions...")
y_pred = gp_classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"[GP Occupation L1 Baseline] Accuracy={accuracy:.4f}, Weighted-F1={weighted_f1:.4f}, Balanced-Acc={balanced_acc:.4f}")

# Update JSON
print("Updating JSON...")
for i, pred_label in zip(idx_test, y_pred):
    users[i].setdefault("predictions", {})["gp_baseline_occupation_L1"] = str(pred_label)

data.setdefault("metrics", {})["gp_baseline_occupation_L1"] = {
    "split": "80/20 random (with oversampling on train)",
    "accuracy": accuracy,
    "f1_score": weighted_f1,
    "balanced_accuracy": balanced_acc,
    "note": "Gaussian Process with ARD kernel; oversampling applied to handle imbalance (ACL 2015)"
}

with open(json_output, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Done.")

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

# Load existing data
with open("../../selected_users_interests_predicted.json", "r") as file:
    data = json.load(file)

users = data["results"]

# Extract texts and labels
texts = [" ".join(user["used_tweets"]) for user in users]
labels = [user["ground_truth"] for user in users]

# Preprocess function
def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text

print("Preprocessing texts...")
texts_clean = [preprocess(text) for text in tqdm(texts)]

print("Tokenizing texts...")
tokenized_texts = [word_tokenize(text) for text in tqdm(texts_clean)]

print("Training Word2Vec embeddings...")
embedding_model = Word2Vec(sentences=tokenized_texts, vector_size=50, window=5, min_count=5, epochs=50)

# Extract embeddings for vocabulary
vocab = list(embedding_model.wv.index_to_key)
word_vectors = embedding_model.wv[vocab]

print("Performing spectral clustering...")
num_clusters = 200
clustering_model = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
clusters = clustering_model.fit_predict(word_vectors)

# Map each word to a cluster
word_to_cluster = dict(zip(vocab, clusters))

# Create cluster features for each user
def create_cluster_features(tokenized_text):
    features = np.zeros(num_clusters)
    for word in tokenized_text:
        cluster_id = word_to_cluster.get(word)
        if cluster_id is not None:
            features[cluster_id] += 1
    return features

print("Creating cluster features...")
features = np.array([create_cluster_features(text) for text in tqdm(tokenized_texts)])

print("Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    features, labels, np.arange(len(labels)), test_size=0.2, random_state=42
)

print("Applying oversampling...")
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Gaussian Process Classifier
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * \
         RBF(length_scale=np.ones(num_clusters), length_scale_bounds=(1e-1, 1e3)) + \
         WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-3, 1e1))

gp_classifier = GaussianProcessClassifier(kernel=kernel,
                                         optimizer='fmin_l_bfgs_b',
                                         n_restarts_optimizer=1,
                                         random_state=42)

print("Training Gaussian Process classifier...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    gp_classifier.fit(X_train_resampled, y_train_resampled)

print("Making predictions on test set...")
y_pred = gp_classifier.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"[GP Baseline] Accuracy={accuracy:.4f}, Weighted-F1={weighted_f1:.4f}, Balanced-Acc={balanced_acc:.4f}")

print("Updating JSON data...")
for i, pred_label in zip(idx_test, y_pred):
    users[i].setdefault("predictions", {})["gp_baseline"] = str(pred_label)

data.setdefault("metrics", {})["gp_baseline"] = {
    "accuracy": accuracy,
    "f1_score": weighted_f1,
    "balanced_accuracy": balanced_acc,
    "note": "Gaussian Process with ARD kernel and oversampling applied"
}

output_json_path = "selected_users_interests_with_gp_baseline.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Done.")

"""
twopics_baseline_occupation_L2.py – Optimized Michelson & Macskassy (AND 2010) Baseline for Occupation L2
=========================================================================================================
Extracts entities from tweets, maps them to DBpedia categories using caching and parallelization,
and infers Occupation Level 2 based on frequent categories. Evaluates using accuracy, weighted-F1,
and balanced accuracy.

Run: python twopics_baseline_occupation_L2.py
"""

import json
import pathlib
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# ---------------------------------------------------------------------
# Paths
JSON_IN = pathlib.Path("../../selected_users_occupations_predicted_L2.json")
JSON_OUT = pathlib.Path("selected_users_occupations_L2_with_twopics_baseline.json")
CACHE_FILE = pathlib.Path("entity_category_cache_occ_L2.pkl")

# ---------------------------------------------------------------------
# Load data
with JSON_IN.open("r", encoding="utf-8") as file:
    data = json.load(file)

users = data["results"]
docs = [" ".join(user["used_tweets"]) for user in users]
labels = np.array([user["ground_truth"] for user in users])
indices = np.arange(len(labels))

# ---------------------------------------------------------------------
# Load or initialize cache
if CACHE_FILE.exists():
    with CACHE_FILE.open('rb') as f:
        entity_category_cache = pickle.load(f)
else:
    entity_category_cache = {}

# ---------------------------------------------------------------------
# DBpedia Spotlight for entity extraction
def extract_entities(text):
    headers = {'Accept': 'application/json'}
    response = requests.get('https://api.dbpedia-spotlight.org/en/annotate',
                            params={'text': text, 'confidence': 0.5}, headers=headers)
    if response.ok:
        resources = response.json().get('Resources', [])
        return [res['@URI'] for res in resources]
    return []

# DBpedia category fetching (SPARQL) with caching
def get_high_level_categories(uri):
    if uri in entity_category_cache:
        return entity_category_cache[uri]

    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(f"""
    SELECT ?type WHERE {{
        <{uri}> a ?type .
        FILTER (STRSTARTS(STR(?type), \"http://dbpedia.org/ontology/\"))
    }}
    """)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        categories = [result["type"]["value"].split('/')[-1] for result in results["results"]["bindings"]]
    except:
        categories = []

    entity_category_cache[uri] = categories
    return categories

# ---------------------------------------------------------------------
# Build user profiles (categories) with parallelization
def process_user(doc, idx, total):
    entities = extract_entities(doc)
    categories = []
    for uri in entities:
        cats = get_high_level_categories(uri)
        categories.extend(cats)
    print(f"Processed user {idx+1}/{total}")
    return categories

user_profiles = []
total_users = len(docs)
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_user, doc, idx, total_users): idx for idx, doc in enumerate(docs)}
    for future in as_completed(futures):
        user_profiles.append(future.result())

# Create category frequency vectors
all_categories = [cat for profile in user_profiles for cat in profile]
common_categories = [item[0] for item in Counter(all_categories).most_common(50)]

def create_feature_vector(profile):
    vector = np.zeros(len(common_categories))
    profile_counter = Counter(profile)
    for idx, cat in enumerate(common_categories):
        vector[idx] = profile_counter.get(cat, 0)
    return vector

features = np.array([create_feature_vector(profile) for profile in user_profiles])

# ---------------------------------------------------------------------
# Train-test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    features, labels, indices,
    test_size=0.20, random_state=42
)

# Classifier (Multinomial Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# ---------------------------------------------------------------------
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"[Twopics Occupation L2 Baseline] Accuracy={accuracy:.4f}, Weighted-F1={weighted_f1:.4f}, Balanced-Acc={balanced_acc:.4f}")

# ---------------------------------------------------------------------
# Update JSON with predictions for test set
for i, pred_label in zip(idx_test, y_pred):
    users[i].setdefault("predictions", {})["twopics_baseline_occupation_L2"] = str(pred_label)

# Update metrics in JSON
data.setdefault("metrics", {})["twopics_baseline_occupation_L2"] = {
    "split": "80/20 random",
    "accuracy": accuracy,
    "f1_score": weighted_f1,
    "balanced_accuracy": balanced_acc,
    "note": "Michelson & Macskassy (2010) entity-based user profiling for Occupation L2, optimized with caching and parallelization"
}

# Save updated JSON
with JSON_OUT.open('w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Save cache
with CACHE_FILE.open('wb') as f:
    pickle.dump(entity_category_cache, f)

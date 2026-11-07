import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# Paths
DATASET_FOLDER = "."
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "random_forest/training_sets")  # pairs built by random_forest/build_*_pairs.py
TEST_FOLDER = os.path.join(DATASET_FOLDER, "random_forest/testing_sets")
GLOVE_FILE = os.path.join(DATASET_FOLDER, "glove.twitter.27B.200d.txt")
RESULTS_XLSX = os.path.join(DATASET_FOLDER, "siamese_glove_results.xlsx")

# Parameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
EPOCHS = 5
BATCH_SIZE = 64

# Load GloVe embeddings
def load_glove_embeddings(filepath):
    embedding_index = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = vector
    print(f"✅ Loaded {len(embedding_index)} GloVe vectors.")
    return embedding_index

# Prepare embedding matrix
def create_embedding_matrix(word_index, embedding_index):
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Siamese Model Definition using pretrained embeddings
def create_siamese_model(num_words, embedding_matrix):
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(input_dim=num_words,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(input_layer)
    lstm_layer = LSTM(128)(embedding_layer)
    encoder = Model(input_layer, lstm_layer)

    input_a = Input(shape=(MAX_SEQUENCE_LENGTH,))
    input_b = Input(shape=(MAX_SEQUENCE_LENGTH,))

    encoded_a = encoder(input_a)
    encoded_b = encoder(input_b)

    cosine_similarity = Lambda(lambda x: tf.reduce_sum(
        tf.multiply(tf.nn.l2_normalize(x[0], axis=1),
                    tf.nn.l2_normalize(x[1], axis=1)), axis=1, keepdims=True)
    )([encoded_a, encoded_b])

    output = Dense(1, activation='sigmoid')(cosine_similarity)
    model = Model([input_a, input_b], output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

# Load dataset from CSV
def load_csv_dataset(path):
    df = pd.read_csv(path)
    tweets_1 = df['Tweet1'].tolist()
    tweets_2 = df['Tweet2'].tolist()
    labels = df['Label'].tolist()
    return tweets_1, tweets_2, labels

# Tokenize and pad sequences
def tokenize_and_pad(tweets1, tweets2, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tweets1 + tweets2)

    seq1 = pad_sequences(tokenizer.texts_to_sequences(tweets1), maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    seq2 = pad_sequences(tokenizer.texts_to_sequences(tweets2), maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    return seq1, seq2, tokenizer

# Train and evaluate function
def train_and_evaluate(train_path, test_path, embedding_index, apply_smote=True):
    train_tweet1, train_tweet2, y_train = load_csv_dataset(train_path)
    test_tweet1, test_tweet2, y_test = load_csv_dataset(test_path)

    X1_train, X2_train, tokenizer = tokenize_and_pad(train_tweet1, train_tweet2)
    X1_test, X2_test, _ = tokenize_and_pad(test_tweet1, test_tweet2, tokenizer=tokenizer)

    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_index)

    X_train = np.concatenate([X1_train, X2_train], axis=1)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        split_idx = X1_train.shape[1]
        X1_train = X_train_resampled[:, :split_idx]
        X2_train = X_train_resampled[:, split_idx:]
        y_train = y_train_resampled

    model = create_siamese_model(len(tokenizer.word_index) + 1, embedding_matrix)
    model.fit([X1_train, X2_train], np.array(y_train), epochs=EPOCHS,
              batch_size=BATCH_SIZE, validation_data=([X1_test, X2_test], np.array(y_test)))

    preds = (model.predict([X1_test, X2_test]) >= 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    precision_0, recall_0, f1_0, support_0 = precision_recall_fscore_support(y_test, preds, average='binary', pos_label=0, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)

    return {
        "Dataset": os.path.basename(test_path),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Precision_0": precision_0,
        "Recall_0": recall_0,
        "F1_0": f1_0,
        "Macro_avg_precision": macro_precision,
        "Macro_avg_recall": macro_recall,
        "Macro_avg_f1": macro_f1,
        "Weighted_avg_precision": weighted_precision,
        "Weighted_avg_recall": weighted_recall,
        "Weighted_avg_f1": weighted_f1,
        "Confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

if __name__ == "__main__":
    embedding_index = load_glove_embeddings(GLOVE_FILE)
    dataset_files = sorted([f for f in os.listdir(TRAIN_FOLDER) if f.startswith("training_dataset") and f.endswith(".csv")])

    results = []
    for train_file in dataset_files:
        test_file = train_file.replace("training_dataset", "testing_dataset")
        train_path = os.path.join(TRAIN_FOLDER, train_file)
        test_path = os.path.join(TEST_FOLDER, test_file)

        if not os.path.exists(test_path):
            continue

        apply_smote = any(keyword in train_file for keyword in ["followers", "followees", "both"])
        result = train_and_evaluate(train_path, test_path, embedding_index, apply_smote)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_excel(RESULTS_XLSX, index=False)
    print(f"✅ Results saved to {RESULTS_XLSX}")

# Task I – Social Media Authorship Verification

Binary classification: given a user's bio and *k* = 20 older posts (few-shot examples), decide whether a candidate post was written by that user (label 1) or by somebody else (label 0). Every candidate is classified independently with **Prompt 1** (`prompt.py`); the model must answer `0` or `1`.

For each of 50 positive users per setting we take *m₁* = 20 of their newer posts (positives) and *m₂* = 20 posts from other users (negatives), so each setting has ≈ 2,000 candidates. Retweets are excluded; URLs are expanded and emojis normalised (`sampling.preprocess_tweets`).

## Sampling framework (15 settings)

| Positive-user sampling (`user_sampling_method`) | Negative-post sampling (`other_tweets_sampling_method`) |
|---|---|
| `random` – random users | `random` – random posts of other users |
| `recent_activity` – users active in the last 3 weeks of the dataset | `similar_topics` – posts most similar to the user's (all-MiniLM-L6-v2 cosine similarity) |
| `top_active` – users with the most posts | `followees` – posts of accounts the user follows |
| | `followers` – posts of the user's followers |
| | `both` – posts of reciprocal (mutual) connections |

Negative posts are always drawn from the same time window as the user's positive posts. The follower/followee strategies need the weekly follower graphs (`$VAPOR_GRAPHS/gr_week_<round>.pkl`).

## Files

| File | Purpose |
|---|---|
| `data_processing.py` | Loads the user descriptions (`.xlsx`) and the posts (`.csv`), drops retweets, groups posts by user, caches to a pickle. |
| `sampling.py` | `sample_users`, `get_user_tweets` (chronological k / m₁ split), `get_others_tweets` (the five negative strategies), text preprocessing. |
| `generate_datasets.py` | Builds `datasets/dataset_{...}.json` for every combination in `param_grid` (edit the lists to produce all 15 settings) and records the class distribution. |
| `prompt.py` | Prompt 1 of the paper. |
| `model_api.py` | Thin wrappers around OpenAI (GPT-4 / GPT-3.5-Turbo), Gemini 1.5 Pro, DeepSeek-V3, Ollama (Llama 3.2) and `bert-base-uncased`; extracts the first `0`/`1` from the answer. |
| `run_classification.py` | Runs the chosen models over every dataset in `datasets/`, checkpoints progress, writes `results/results_<model>_<dataset>.json` and appends accuracy / precision / recall / F1 to `evaluation_metrics.csv`. |
| `evaluation.py` | Metric computation (invalid answers are counted and skipped). |
| `clean_metrics.py` | Parses the sklearn classification report into per-class and weighted columns → `cleaned_evaluation_metrics.csv` (the format used in `results/task1/`). |
| `plot_results.py` | Class-wise precision/recall bars and the F1 (neg.) vs F1 (pos.) scatter from `results/task1/weighted_metrics_all_models.csv`. |

Run order:

```bash
export VAPOR_DESCRIPTIONS=... VAPOR_TWEETS=... VAPOR_GRAPHS=...   # see ../.env.example
python generate_datasets.py
python run_classification.py --models gpt-4 gpt-3.5-turbo gemini-1.5-pro-002 deepseek llama3.2 bert
python clean_metrics.py
python ../scripts/stats_tests.py
```

Each dataset entry has the form

```json
{"user_id": 123, "description": "...", "previous_tweets": ["..."],
 "future_tweets": [{"tweet": "...", "ground_truth_label": 1}],
 "others_tweets": [{"tweet": "...", "ground_truth_label": 0}]}
```

## Unseen-data evaluation (`unseen_data/`)

To test for "seen data" bias we evaluated the LLMs on posts published **after January 2024**, i.e. after every model's training cutoff, for 50 VAPOR users. Those posts were downloaded with the official X API v2 user-timeline endpoint into one `<user_id>_timeline_data.parquet` file per user (columns include `user_id`, `text`, `tweet_created_at`); the download itself is a plain API call and is not part of this repository.

```bash
cd unseen_data
python combine_parquet_files.py        # parquet_files/*.parquet -> combined_users_original.parquet
python summarize_parquet_files.py      # optional: posts / retweets per user
python prepare_dataset.py              # k = m1 = m2 = 20, retweets removed -> datasets/classification_dataset.json
cd .. && python run_classification.py --datasets unseen_data/datasets --results unseen_data/results --metrics unseen_data/evaluation_metrics.csv --models gpt-4 gemini-1.5-pro-002 deepseek llama3.2
```

## Baselines (`baselines/`)

All baselines read the same `datasets/*.json` files, so they see exactly the same positives/negatives as the LLMs.

| Script | Paper name | Notes |
|---|---|---|
| `random_forest/` | Random Forest | `add_training_negatives.py` adds *training* negatives from the same strategy, `build_training_pairs.py` / `build_testing_pairs.py` turn each dataset into (post, post, same-author?) pairs, `rf_classifier.py` trains an RF on concatenated SBERT embeddings (grid search, SMOTE for the graph-based settings) and decides each candidate by majority vote over its pairs. |
| `tfidf_cosine_baseline.py` | TF-IDF | word 1–3-gram TF-IDF cosine similarity between each candidate and the user's concatenated example posts, thresholded at the per-user mean similarity. |
| `use_cosine_baseline.py` | USE | same idea with Universal Sentence Encoder embeddings (needs `tensorflow_hub`). |
| `compression_ncd_baseline.py` | Compression-NCD | normalised compression distance (gzip) between the candidate and the user's example posts, thresholded at the per-user median. |
| `siamese_glove.py` | SIAMESE + GloVe | Siamese LSTM over GloVe Twitter 200-d embeddings, trained on the RF pair files (SMOTE on the training pairs). |
| `siamese_sbert.py` | SIAMESE + SBERT | Siamese network over frozen all-MiniLM-L6-v2 sentence embeddings, same pair files. |
| `extras/stylometry_baseline.py` | – | Burrows' Delta stylometry baseline (`faststylometry`); explored but not reported in the paper. |

Per-setting results of every model and baseline are in [`../results/task1/weighted_metrics_all_models.csv`](../results/task1/weighted_metrics_all_models.csv).

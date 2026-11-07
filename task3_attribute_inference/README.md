# Task III – User Attribute Inference

Given 50 randomly sampled posts of a VAPOR user, the model must pick **exactly one** category (**Prompt 3**, `create_prompt` in the prediction scripts; the answer is requested as JSON) for

| Attribute | Taxonomy | Categories present among the 383 users |
|---|---|---|
| Occupation L1 | 2018 U.S. Standard Occupational Classification – major groups | 18 of 23 |
| Occupation L2 | 2018 SOC – minor groups | 38 of 98 |
| Interests / hobbies | IAB Tech Lab Content Taxonomy v3.1 – tier-1 categories | 25 of 39 |

The candidate list given to the model is built from the annotation file (`get_unique_occupations` / `get_unique_categories`), i.e. only categories that actually occur among the annotated users are offered. The manual annotations (1,149 labels for 383 users) are released without post text in [`../data/vapor_user_annotations.json`](../data/vapor_user_annotations.json); the full input file (`selected_users2_fully_labeled.json`) additionally holds each user's posts and is therefore not distributed (see `../data/README.md`).

Metrics: accuracy and weighted F1 over the categories in the candidate list; predictions that are not a valid category count as `Uncategorized` (wrong).

## Files

| File | Purpose |
|---|---|
| `model_api_occupation.py`, `model_api_interest.py` | API wrappers (GPT-4o, Gemini 1.5 Pro, DeepSeek-V3, Ollama/Llama 3.2) returning the raw JSON answer. |
| `occupation_prediction_L1.py`, `occupation_prediction_L2.py`, `interest_prediction.py` | Sample 50 posts per user, query all models (with a per-call timeout), parse the JSON, compute accuracy / weighted F1, resume from the output file. Output: `selected_users_{occupations_predicted_L1,occupations_predicted_L2,interests_predicted}.json`. |
| `annotation_summary.py` | Label statistics of the annotation file → `occupations_summary.xlsx`. |
| `annotations_to_csv.py` | Flattens posts + bio + interest label into a CSV (used to fine-tune the few-shot BART baseline). |

```bash
export OPENAI_API_KEY=... GEMINI_API_KEY=... DEEPSEEK_API_KEY=...
python occupation_prediction_L1.py
python occupation_prediction_L2.py
python interest_prediction.py
```

## Baselines (`baselines/<attribute>/`)

The baselines read the LLM output file of the same attribute (they re-use the *same 50 sampled posts per user*, stored as `used_tweets`), evaluate on an 80/20 random split and append their metrics to a copy of the file.

| Script | Paper name | Method |
|---|---|---|
| `gaussian_process_word2vec.py` | Gaussian Process + Word2Vec | Word2Vec (50-d) → spectral clustering of words → cluster features → GP classifier with ARD kernel, random oversampling (Preoţiuc-Pietro et al., ACL 2015 style). |
| `bart_fewshot.py` | BART (few-shot) | `facebook/bart-large-mnli` fine-tuned on the training split (oversampled). |
| `dbpedia_entities_twopics.py` | DBpedia entities | entity mentions → DBpedia categories (SPARQL, cached) → most frequent categories mapped to the label (Michelson & Macskassy, 2010). |
| `tfidf_gradient_boosting_pennacchiotti.py` | TF-IDF + Gradient Boosting | TF-IDF features + gradient boosting with sample weighting (Pennacchiotti & Popescu, ICWSM 2011). |
| `extras/bart_zeroshot.py`, `extras/bow_logistic_regression.py` | – | zero-shot NLI classification and a bag-of-words logistic regression; explored but not reported. |

Summary metrics of all models and baselines: [`../results/task3/summary_metrics.json`](../results/task3/summary_metrics.json); per-user predictions (user ID, gold label, predictions) are next to it.

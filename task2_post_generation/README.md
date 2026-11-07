# Task II – Social Media Post Generation

Each LLM is asked to write *n* = 10 posts (≤ 280 characters) in the style of a user, given *k* = 50 of the user's own posts, the user's bio and follower/followee counts (**Prompt 2**, see `model_api.generate_tweets`). The generated posts are compared with

* the 50 **Prompt** posts the model saw, and
* 50 further **Non-prompt** posts of the same user it did not see,

using semantic similarity between all-MiniLM-L6-v2 embeddings (AOS = average overall similarity, AGR = average of each generated post's best match, ARG = average of each real post's best match), the dispersion ratios GDR / RDR (how many distinct real posts are "covered"), corpus BLEU, ROUGE-1 / ROUGE-L and GPT-2 perplexity. The metric code lives in `generate_and_evaluate.py` (`compute_bleu`, `compute_rouge`, `compute_perplexity` and the similarity block); the same code is copied into every baseline so all models are scored identically. In the result files the metric keys are `avg_overall_similarity` (AOS), `avg_max_row_similarity` (AGR), `avg_max_col_similarity` (ARG), `dispersion_row_ratio` (GDR) and `dispersion_col_ratio` (RDR).

## VAPOR users

`user_selection.py` selects the users used in Tasks II and III: active in 2018–2020, > 200 original posts (retweets excluded), > 500 followers, > 100 followees, a bio longer than 20 characters and a verified badge → **383 users** (*Verified, Active, Profile-rich, ORiginal*). `user_selection_with_tweets.py` attaches each selected user's original posts (used as the input of Task III).

## Files

| File | Purpose |
|---|---|
| `data_processing.py` | Loads profiles + posts from the dataset (`$VAPOR_DESCRIPTIONS`, `$VAPOR_TWEETS`) and caches them. *Re-implemented from the version used on our server; behaviour is identical for the callers.* |
| `user_selection.py` | VAPOR filter → `selected_users2.json`. |
| `model_api.py` | Prompt 2 + wrappers for Gemini 1.5 Pro, DeepSeek-V3 and Ollama/Llama 3.2 (and OpenAI for non-batch use). |
| `model_api_batch.py` | Same prompt submitted to the **OpenAI Batch API** (used for GPT-4o to reduce cost) and the polling/download of the results. |
| `generate_and_evaluate.py` | Samples 50 prompt + 50 non-prompt posts per user, generates with Gemini / Llama / DeepSeek, computes all metrics, resumes from `user_analysis_nonbatch.json`. |
| `generate_and_evaluate_gpt4o_batch.py` | Re-uses exactly the same prompt / non-prompt posts and runs GPT-4o through the batch API → `user_analysis_batch.json`. |
| `baselines/markov_baseline.py` | Markov chain (`markovify`, state size 2) trained on the user's prompt posts. |
| `baselines/bart_baseline.py`, `baselines/t5_baseline.py` | `facebook/bart-large` and `t5-large` conditioned on the prompt posts (sampling decoding). |
| `baselines/extras/` | Retrieval and template baselines that were explored but are not reported in the paper. |

Run order:

```bash
python user_selection.py
python generate_and_evaluate.py
python generate_and_evaluate_gpt4o_batch.py
python baselines/markov_baseline.py && python baselines/bart_baseline.py && python baselines/t5_baseline.py
```

Per-user and overall metrics of every model (with the real posts removed) are in [`../results/task2/`](../results/task2).

## User study (`user_study/`)

Real X users who consented to the study provided their handle; we collected their recent posts, generated 8 posts per model with the same prompt (`generate_for_participants.py`, GPT-4o again via the batch API), exported them to a Qualtrics survey (`export_for_qualtrics.py`) and asked each participant whether each post "could have been written by you" on a five-point scale. `plot_survey_responses.py` draws the response chart from the aggregate counts, which are also stored in [`../results/user_study/survey_response_counts.json`](../results/user_study/survey_response_counts.json). Participants' handles, posts and the generated posts shown to them are not released.

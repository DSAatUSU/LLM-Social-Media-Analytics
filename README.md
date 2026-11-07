# Social Media Analytics with LLMs: A Comprehensive Multi-task Study

Code and aggregate results for the paper

> Ramtin Davoudi, Kartik Thakkar, Nazanin Donyapour, Tyler Derr, Hamid Karimi.
> **Social Media Analytics with LLMs: A Comprehensive Multi-task Study.**
> *2026 IEEE International Conference on Big Data (IEEE BigData 2026)*, 12th Special Session on Intelligent Data Mining, Phoenix, AZ, USA.

We benchmark GPT-4, GPT-4o, GPT-3.5-Turbo, Gemini 1.5 Pro, DeepSeek-V3 and Llama 3.2 on three social media analytics tasks built on a Twitter/X dataset:

| Task | Folder | What it does |
|---|---|---|
| **I – Authorship Verification** | [`task1_authorship_verification/`](task1_authorship_verification) | Can an LLM tell whether a post was written by a given user? 3 positive-user × 5 negative-post sampling strategies (15 settings), an unseen-data (2024+) evaluation, and 6 baselines (RF, TF-IDF, USE, Compression-NCD, Siamese GloVe/SBERT). |
| **II – Post Generation** | [`task2_post_generation/`](task2_post_generation) | Generate posts in a user's own style from k example posts + bio + follower counts; evaluate with semantic similarity (AOS, AGR, ARG), dispersion (GDR, RDR), BLEU, ROUGE and GPT-2 perplexity; Markov-chain, BART-large and T5-large baselines; and the user study. |
| **III – User Attribute Inference** | [`task3_attribute_inference/`](task3_attribute_inference) | Infer occupation (SOC 2018, levels L1/L2) and interests/hobbies (IAB Content Taxonomy v3.1) of the 383 *VAPOR* users from 50 sampled posts; 4 baselines (GP + Word2Vec, few-shot BART, DBpedia entities, TF-IDF + gradient boosting). |

Supporting folders:

* [`results/`](results) – aggregate results behind every table in the paper (no post text, no personal data).
* [`data/`](data) – the occupation / interest annotations of the VAPOR users (user IDs + labels only) and the data policy.
* [`scripts/stats_tests.py`](scripts/stats_tests.py) – reproduces Table II / Table III and the statistical tests (Wilcoxon, Friedman, Kendall's W) from `results/task1/`.

## Setup

```bash
git clone https://github.com/DSAatUSU/LLM-Social-Media-Analytics.git
cd LLM-Social-Media-Analytics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Model access:

* **OpenAI / Gemini / DeepSeek** – API keys are read from the environment (never hard-coded). Copy `.env.example` to `.env`, fill in the keys and `source` it (or `export` them in your shell).
* **Llama 3.2** – runs locally through [Ollama](https://ollama.com) (`ollama pull llama3.2`; the 3B default checkpoint was used).
* **BERT** (Task I) – `bert-base-uncased` from Hugging Face with a *randomly initialised* classification head. It is deliberately **not fine-tuned**: it serves as a reference for what a pre-trained encoder does with the same prompt-free input, which is why it collapses to one class in several settings (Table II).
* **Google USE** (Task I baseline) needs `tensorflow` + `tensorflow_hub`; the Siamese GloVe baseline needs `glove.twitter.27B.200d.txt` from the [GloVe](https://nlp.stanford.edu/projects/glove/) site.

Data: the Twitter dataset itself is **not** part of this repository (see [`data/README.md`](data/README.md)). Point the scripts at your local copy with

```bash
export VAPOR_DESCRIPTIONS=/path/to/user_descriptions_rounds.xlsx   # user profiles (user_id, description, followers_count, following, verified, ...)
export VAPOR_TWEETS=/path/to/tweets_no_duplicates.csv              # posts (user_id, text, tweet_created_at, retweeted_status_text, Round)
export VAPOR_GRAPHS=/path/to/graphs                                 # weekly follower graphs gr_week_<round>.pkl (networkx DiGraph)
```

## Reproducing the paper

Each task folder has its own README with the exact order of the scripts. In short:

```bash
# Task I
cd task1_authorship_verification
python generate_datasets.py                       # 15 sampling settings -> datasets/*.json
python run_classification.py --models gpt-4 gpt-3.5-turbo gemini-1.5-pro-002 deepseek llama3.2 bert
python clean_metrics.py                           # flatten the sklearn reports -> cleaned_evaluation_metrics.csv
python ../scripts/stats_tests.py                  # Table II, Table III, Wilcoxon / Friedman / Kendall's W

# Task II
cd ../task2_post_generation
python user_selection.py                          # -> selected_users2.json (383 VAPOR users)
python generate_and_evaluate.py                   # Gemini, Llama, DeepSeek  -> user_analysis_nonbatch.json
python generate_and_evaluate_gpt4o_batch.py       # GPT-4o via the OpenAI Batch API -> user_analysis_batch.json
python baselines/markov_baseline.py               # + bart_baseline.py, t5_baseline.py

# Task III
cd ../task3_attribute_inference
python occupation_prediction_L1.py                # + occupation_prediction_L2.py, interest_prediction.py
python baselines/occupation_L1/bart_fewshot.py    # + the other baselines
```

The numbers reported in the paper can be regenerated from `results/` without any API access:

```bash
python scripts/stats_tests.py
```

## Data policy (short version)

* Post text is never redistributed (X/Twitter developer policy). The dataset is cited in the paper (Kheiri et al., "An analysis of the dynamics of ties on Twitter", IEEE BigData 2023; Farokhi et al., "EdgeUp", 2025); please contact the authors for access.
* The 2024+ posts used for the unseen-data evaluation were collected with the official X API v2 user-timeline endpoint and are likewise not included.
* `data/vapor_user_annotations.json` releases only user IDs with the annotated occupation / interest labels.
* The user study released nothing but aggregate response counts (`results/user_study/`); participants' handles, posts and generated posts are not included.

## Citation

```bibtex
@inproceedings{davoudi2026social,
  title     = {Social Media Analytics with {LLMs}: A Comprehensive Multi-task Study},
  author    = {Davoudi, Ramtin and Thakkar, Kartik and Donyapour, Nazanin and Derr, Tyler and Karimi, Hamid},
  booktitle = {2026 IEEE International Conference on Big Data (BigData)},
  year      = {2026},
  organization = {IEEE}
}
```

## Contact

Ramtin Davoudi – ramtin.davoudi@usu.edu · [Data Science and Applications Lab](https://github.com/DSAatUSU), Utah State University.

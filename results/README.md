# Results

Aggregate results behind the tables of the paper. None of these files contain post text, names, handles or bios.

## `task1/` – authorship verification

| File | Content |
|---|---|
| `weighted_metrics_all_models.csv` | **Main table.** 12 models × 15 sampling settings (`other_tweets_sampling_method`, `user_sampling_method`): accuracy, binary precision/recall/F1, per-class precision/recall/F1/support, macro and weighted averages. `weighted_avg_f1` × 100 is Table II; `scripts/stats_tests.py` derives Table III and all statistical tests from this file. |
| `raw_evaluation_metrics_llms.csv`, `raw_evaluation_metrics_deepseek.csv` | Raw output of `run_classification.py` (with the sklearn classification report as text) before `clean_metrics.py`. |
| `baseline_*.csv/.xlsx` | Raw outputs of the individual baseline scripts (same numbers as in the main table). |
| `class_distributions.json` | Number of positive/negative candidates per setting. |
| `best_worst_performance.csv` | Best and worst setting per model and metric. |

Model names: `Bert` = BERT, `Llama3.2` = Llama 3.2, `Gemini` = Gemini 1.5 Pro, `DeepSeek` = DeepSeek-V3, `RF` = Random Forest, `Gzip` = Compression-NCD, `USE` = Universal Sentence Encoder, `SIAMESE_SBERT` / `Siamese_GloVe` = Siamese baselines.

## `task2/` – post generation

One JSON per model / baseline with `overall_average_metrics` (per scenario `all` / `prompt` / `nonprompt`) and `user_results` (per user: the metrics and GPT-2 perplexity, and for the LLMs also the generated posts). The real prompt / non-prompt posts were removed (only their counts are kept), and the baselines' generated posts are omitted as well because the Markov chain / BART / T5 outputs reproduce the users' real posts almost verbatim (BLEU up to 0.96).

| File | Models |
|---|---|
| `llm_gemini_llama_deepseek.json` | Gemini 1.5 Pro, Llama 3.2, DeepSeek-V3 |
| `llm_gpt4o_batch.json` | GPT-4o (OpenAI batch API) |
| `baseline_markov.json`, `baseline_bart.json`, `baseline_t5.json` | Markov chain, BART-large, T5-large |

Metric keys: `avg_overall_similarity` = AOS, `avg_max_row_similarity` = AGR, `avg_max_col_similarity` = ARG, `dispersion_row_ratio` = GDR, `dispersion_col_ratio` = RDR, plus `bleu`, `rouge.rouge1`, `rouge.rougeL` and `avg_perplexity`.

## `task3/` – user attribute inference

* `summary_metrics.json` – accuracy / weighted F1 of the four LLMs and every baseline for interests, occupation L1 and occupation L2 (Table IX). Baselines are evaluated on their 80/20 test split; `balanced_accuracy` is reported in addition where the script computed it.
* `<attribute>_llm_predictions.json` – per user: `user_id`, `ground_truth`, and the prediction of each LLM (`openai` = GPT-4o, `gemini`, `deepseek`, `ollama` = Llama 3.2).
* `<attribute>_baseline_<name>.json` – the same with the baseline's prediction added (`gp_baseline` = Gaussian Process + Word2Vec, `bart_fewshot`, `twopics_baseline` = DBpedia entities, `pennacchiotti` = TF-IDF + gradient boosting; `hf_baseline` = zero-shot BART and `gp_ard_baseline` are extra runs not reported in the paper).

## `user_study/`

`survey_response_counts.json` – number of responses per model and answer option ("Could this post have been written by you?").

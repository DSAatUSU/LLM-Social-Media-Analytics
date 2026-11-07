import json
import markovify
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk

nltk.download('punkt', quiet=True)

# ✅ Paths
DATA_PATH = "user_analysis_nonbatch.json"
OUTPUT_JSON = "user_analysis_markov.json"
NUM_GENERATED_TWEETS = 10

# ✅ Load existing JSON data (prompt and non-prompt tweets)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    previous_results = json.load(f)

# ✅ Load evaluation models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
perplexity_model.eval()

# ✅ Metric functions
def compute_perplexity(texts):
    perplexities = []
    for text in texts:
        inputs = perplexity_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = perplexity_model(**inputs, labels=inputs["input_ids"]).loss
        perplexities.append(torch.exp(loss).item())
    return sum(perplexities) / len(perplexities)

def compute_bleu(references, candidates):
    refs = [[word_tokenize(ref.lower()) for ref in references]] * len(candidates)
    cands = [word_tokenize(cand.lower()) for cand in candidates]
    smoothing = SmoothingFunction().method1
    return corpus_bleu(refs, cands, smoothing_function=smoothing)

def compute_rouge(references, candidates):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1, rougeL = [], []
    for cand in candidates:
        scores = [scorer.score(ref, cand) for ref in references]
        rouge1.append(max(score['rouge1'].fmeasure for score in scores))
        rougeL.append(max(score['rougeL'].fmeasure for score in scores))
    return {
        "rouge1": sum(rouge1) / len(rouge1),
        "rougeL": sum(rougeL) / len(rougeL)
    }

# ✅ Generate tweets using Markov chain
def generate_markov_tweets(tweets, num_tweets=NUM_GENERATED_TWEETS):
    text_corpus = "\n".join(tweets)
    model = markovify.NewlineText(text_corpus, state_size=2)
    generated_tweets = []
    attempts = 0
    max_attempts = num_tweets * 20

    while len(generated_tweets) < num_tweets and attempts < max_attempts:
        tweet = model.make_short_sentence(280, tries=100)
        if tweet:
            generated_tweets.append(tweet)
        attempts += 1

    while len(generated_tweets) < num_tweets:
        generated_tweets.append("Unable to generate tweet.")

    return generated_tweets

# ✅ Main processing loop using existing prompt and non-prompt tweets
user_results = []
total_users = len(previous_results["user_results"])

for idx, user in enumerate(previous_results["user_results"], 1):
    user_id = user["user_id"]
    prompt_tweets = user["prompt_tweets"]
    non_prompt_tweets = user["non_prompt_tweets"]

    print(f"\n⏳ Processing user {user_id} ({idx}/{total_users})")

    generated_tweets = generate_markov_tweets(prompt_tweets)

    scenarios = {
        "all": prompt_tweets + non_prompt_tweets,
        "prompt": prompt_tweets,
        "nonprompt": non_prompt_tweets
    }

    metrics = {}
    for scenario, real_tweets in scenarios.items():
        real_embeds = similarity_model.encode(real_tweets, convert_to_tensor=True)
        gen_embeds = similarity_model.encode(generated_tweets, convert_to_tensor=True)
        similarities = util.cos_sim(gen_embeds, real_embeds)

        metrics[scenario] = {
            "avg_overall_similarity": similarities.mean().item(),
            "avg_max_row_similarity": similarities.max(dim=1).values.mean().item(),
            "avg_max_col_similarity": similarities.max(dim=0).values.mean().item(),
            "dispersion_row_ratio": len(similarities.max(dim=1).indices.unique()) / similarities.shape[1],
            "dispersion_col_ratio": len(similarities.max(dim=0).indices.unique()) / similarities.shape[0],
            "bleu": compute_bleu(real_tweets, generated_tweets),
            "rouge": compute_rouge(real_tweets, generated_tweets)
        }

    avg_perplexity = compute_perplexity(generated_tweets)

    user_results.append({
        "user_id": user_id,
        "prompt_tweets": prompt_tweets,
        "non_prompt_tweets": non_prompt_tweets,
        "generated_tweets": generated_tweets,
        "metrics": metrics,
        "avg_perplexity": avg_perplexity
    })

    print(f"✅ Finished user {user_id} ({idx}/{total_users})")

# ✅ Calculate overall metrics
overall_metrics = {
    scenario: {
        metric: sum(u["metrics"][scenario][metric] for u in user_results) / total_users
        if metric != "rouge" else {
            rouge_metric: sum(u["metrics"][scenario]["rouge"][rouge_metric] for u in user_results) / total_users
            for rouge_metric in ["rouge1", "rougeL"]
        }
        for metric in user_results[0]["metrics"][scenario]
    } for scenario in ["all", "prompt", "nonprompt"]
}

overall_avg_perplexity = sum(u["avg_perplexity"] for u in user_results) / total_users

# ✅ Save final JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "user_results": user_results,
        "overall_metrics": overall_metrics,
        "overall_avg_perplexity": overall_avg_perplexity
    }, f, indent=4, ensure_ascii=False)

print(f"\n🎉 Processing completed successfully. Results saved to '{OUTPUT_JSON}'")

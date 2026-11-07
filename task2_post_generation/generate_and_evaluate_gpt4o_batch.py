import os
import json
import torch
from model_api_batch import generate_tweets_batch, retrieve_batch_results
from data_processing import load_data
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk

nltk.download('punkt', force=True)

# Parameters
num_generated_tweets = 10
n_users = 414

# Paths
DESCRIPTION_FILE = os.environ.get("VAPOR_DESCRIPTIONS", "data/raw/user_descriptions_rounds.xlsx")
TWEETS_FILE = os.environ.get("VAPOR_TWEETS", "data/raw/tweets_no_duplicates.csv")
CACHE_PATH = "preprocessed_users.pkl"
SELECTED_USERS_JSON = "selected_users2.json"
PREVIOUS_RESULTS_JSON = "user_analysis_nonbatch.json"
OUTPUT_JSON = "user_analysis_batch.json"

# Model
model_name = "gpt-4o"

# API Key
api_key = os.environ.get("OPENAI_API_KEY")

# Load data
users_data, tweets_data = load_data(DESCRIPTION_FILE, TWEETS_FILE, cache_path=CACHE_PATH)
with open(SELECTED_USERS_JSON, "r") as f:
    selected_users = json.load(f)

selected_user_ids = [user["user_id"] for user in selected_users[:n_users]]

# Load previously selected tweets from the nonbatch results
with open(PREVIOUS_RESULTS_JSON, "r") as f:
    previous_results = json.load(f)

previously_selected_tweets = {
    user["user_id"]: {
        "prompt_tweets": user["prompt_tweets"],
        "non_prompt_tweets": user["non_prompt_tweets"]
    }
    for user in previous_results["user_results"]
}

# Checkpoints
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        saved_data = json.load(f)
        output_data = saved_data.get("user_results", [])
    processed_user_ids = {entry["user_id"] for entry in output_data}
    print(f"✅ Loaded {len(processed_user_ids)} previously processed users. Resuming...")
else:
    output_data = []
    processed_user_ids = set()

# Embedding model
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perplexity model (GPT-2)
perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
perplexity_model.eval()

# Metric functions
def compute_perplexity(texts, model, tokenizer, device):
    perplexities = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
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
        "rouge1": sum(rouge1)/len(rouge1),
        "rougeL": sum(rougeL)/len(rougeL)
    }

# Prepare batch requests from previously selected tweets
batch_requests = []
for user_id in selected_user_ids:
    if user_id in processed_user_ids:
        print(f"⏩ User {user_id} already processed. Skipping...")
        continue

    user_selected_tweets = previously_selected_tweets.get(user_id)
    if not user_selected_tweets:
        print(f"⚠️ User {user_id} not found in previous results, skipping...")
        continue

    prompt_tweets = user_selected_tweets["prompt_tweets"]

    user_info = users_data.get(user_id, {})
    user_context = {
        "description": user_info.get("description", ""),
        "followers_count": user_info.get("followers_count", 0),
        "following_count": user_info.get("following", 0)
    }

    batch_requests.append({
        "custom_id": user_id,
        "input_data": {
            "sampled_tweets": prompt_tweets,
            "user_context": user_context
        }
    })

if not batch_requests:
    print("✅ No new users to process or matching users not found!")
    exit()

# Batch generation
batch_id = generate_tweets_batch(batch_requests, model_name, api_key)
print(f"✅ Batch submitted! Batch ID: {batch_id}")

generated_results = retrieve_batch_results(batch_id, api_key)

# Evaluation
for result in generated_results:
    user_id = result["custom_id"]
    if result["response"]["status_code"] == 200:
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        generated_tweets = [tweet.strip() for tweet in content.split("\n") if tweet.strip()]
    else:
        print(f"⚠️ Skipping user {user_id} due to API error: {result['error']}")
        continue  # Gracefully handle API errors per user

    prompt_tweets = previously_selected_tweets[user_id]["prompt_tweets"]
    non_prompt_tweets = previously_selected_tweets[user_id]["non_prompt_tweets"]

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

    avg_perplexity = compute_perplexity(generated_tweets, perplexity_model, perplexity_tokenizer, device)

    output_data.append({
        "user_id": user_id,
        "prompt_tweets": prompt_tweets,
        "non_prompt_tweets": non_prompt_tweets,
        "generated_tweets": generated_tweets,
        "metrics": metrics,
        "avg_perplexity": avg_perplexity
    })

# Calculate overall metrics
total_users = len(output_data)
overall_metrics = {
    scenario: {
        metric: sum(u["metrics"][scenario][metric] for u in output_data) / total_users
        if metric != "rouge" else {
            rouge_metric: sum(u["metrics"][scenario]["rouge"][rouge_metric] for u in output_data) / total_users
            for rouge_metric in ["rouge1", "rougeL"]
        }
        for metric in output_data[0]["metrics"][scenario]
    } for scenario in ["all", "prompt", "nonprompt"]
}

overall_avg_perplexity = sum(u["avg_perplexity"] for u in output_data) / total_users

# Save final JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump({
        "user_results": output_data,
        "overall_average_metrics": overall_metrics,
        "overall_avg_perplexity": overall_avg_perplexity
    }, f, indent=4)

print(f"📊 Evaluation complete! Results saved to {OUTPUT_JSON}")

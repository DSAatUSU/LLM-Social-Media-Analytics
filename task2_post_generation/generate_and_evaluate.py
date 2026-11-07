import os
import json
import random
import torch
from model_api import initialize_model, generate_tweets
from data_processing import load_data
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk

nltk.download('punkt', force=True)

# Parameters
sampling_number = 50
non_prompt_tweets_number = 50
num_generated_tweets = 10
n_users = 414

# Paths
DESCRIPTION_FILE = os.environ.get("VAPOR_DESCRIPTIONS", "data/raw/user_descriptions_rounds.xlsx")
TWEETS_FILE = os.environ.get("VAPOR_TWEETS", "data/raw/tweets_no_duplicates.csv")
CACHE_PATH = "preprocessed_users.pkl"
SELECTED_USERS_JSON = "selected_users2.json"
OUTPUT_JSON = "user_analysis_nonbatch.json"

# Models (excluding OpenAI)
models = {
    "gemini": ["gemini-1.5-pro-002"],
    "ollama": ["llama3.2"],
    "deepseek": ["deepseek"]
}

# API Keys
gemini_api_key = os.environ.get("GEMINI_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

# Load data
users_data, tweets_data = load_data(DESCRIPTION_FILE, TWEETS_FILE, cache_path=CACHE_PATH)
with open(SELECTED_USERS_JSON, "r") as f:
    selected_users = json.load(f)

random.shuffle(selected_users)
selected_users = selected_users[:n_users]
selected_user_ids = [user["user_id"] for user in selected_users]

# Checkpoint handling
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        saved_data = json.load(f)
        output_data = saved_data.get("user_results", [])
    processed_user_ids = {entry["user_id"] for entry in output_data}
    print(f"✅ Loaded {len(processed_user_ids)} previously processed users. Resuming...")
else:
    output_data = []
    processed_user_ids = set()

# Evaluation Models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perplexity Model (GPT-2)
perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
perplexity_model.eval()

# Metrics functions (unchanged)
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
        "rouge1": sum(rouge1) / len(rouge1),
        "rougeL": sum(rougeL) / len(rougeL)
    }

# Main processing loop with retweet removal
for user_id in selected_user_ids:
    if user_id in processed_user_ids:
        print(f"⏩ User {user_id} already processed. Skipping...")
        continue

    print(f"🔸 Processing user: {user_id}")

    all_tweets = [t["text"] for t in tweets_data.get(user_id, []) if not t["text"].startswith("RT @")]
    random.shuffle(all_tweets)

    if len(all_tweets) < sampling_number + non_prompt_tweets_number:
        print(f"⚠️ Skipping user {user_id} due to insufficient non-retweet tweets.")
        continue

    prompt_tweets = all_tweets[:sampling_number]
    non_prompt_tweets = all_tweets[sampling_number:sampling_number + non_prompt_tweets_number]

    user_info = users_data.get(user_id, {})
    user_context = {
        "description": user_info.get("description", ""),
        "followers_count": user_info.get("followers_count", 0),
        "following_count": user_info.get("following", 0)
    }

    input_data = {
        "sampled_tweets": prompt_tweets,
        "user_context": user_context
    }

    user_output = {
        "user_id": user_id,
        "prompt_tweets": prompt_tweets,
        "non_prompt_tweets": non_prompt_tweets,
        "generated_tweets": {},
        "metrics": {},
        "avg_perplexity": {}
    }

    for model_type, model_names in models.items():
        for model_name in model_names:
            print(f"  🔹 Generating tweets: {model_name}")

            model_resources = initialize_model(
                model_type=model_type,
                model_name=model_name,
                gemini_api_key=gemini_api_key,
                deepseek_api_key=deepseek_api_key
            )

            try:
                generated_tweets = generate_tweets(
                    input_data=input_data,
                    model_name=model_name,
                    model_type=model_type,
                    num_generated_tweets=num_generated_tweets,
                    model_resources=model_resources,
                    sampling_number=sampling_number
                )
            except Exception as e:
                print(f"⚠️ Error with {model_name} for user {user_id}: {e}. Skipping this user-model pair.")
                continue

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

            user_output["generated_tweets"][model_name] = generated_tweets
            user_output["metrics"][model_name] = metrics
            user_output["avg_perplexity"][model_name] = avg_perplexity

    output_data.append(user_output)

# Compute overall metrics
overall_metrics = {}
for model_type, model_names in models.items():
    for model_name in model_names:
        model_users = [u for u in output_data if model_name in u["metrics"]]
        if not model_users:
            continue

        overall_metrics[model_name] = {}

        for scenario in ["all", "prompt", "nonprompt"]:
            scenario_metrics = {
                key: sum(u["metrics"][model_name][scenario][key] for u in model_users) / len(model_users)
                for key in model_users[0]["metrics"][model_name][scenario] if key != "rouge"
            }
            scenario_metrics["rouge"] = {
                rouge_key: sum(u["metrics"][model_name][scenario]["rouge"][rouge_key] for u in model_users) / len(model_users)
                for rouge_key in ["rouge1", "rougeL"]
            }
            overall_metrics[model_name][scenario] = scenario_metrics

        overall_metrics[model_name]["avg_perplexity"] = sum(u["avg_perplexity"][model_name] for u in model_users) / len(model_users)

# Save results
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "user_results": output_data,
        "overall_average_metrics": overall_metrics
    }, f, indent=4)

print(f"📊 Evaluation completed! Results saved to {OUTPUT_JSON}")

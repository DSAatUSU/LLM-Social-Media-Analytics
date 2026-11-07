import json
import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk

nltk.download('punkt', quiet=True)

# Paths
DATA_PATH = "user_analysis_nonbatch.json"
OUTPUT_JSON = "user_analysis_bart.json"
NUM_GENERATED_TWEETS = 10
TEMPERATURE = 0.7

# Load existing JSON data (prompt and non-prompt tweets)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    previous_results = json.load(f)

# Load evaluation models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

perplexity_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
perplexity_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Metric functions
def compute_perplexity(texts):
    perplexities = []
    for text in texts:
        inputs = perplexity_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
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
    return {"rouge1": sum(rouge1) / len(rouge1), "rougeL": sum(rougeL) / len(rougeL)}

# Generate tweets using BART
def generate_bart_tweets(tweets, user_context, num_tweets=NUM_GENERATED_TWEETS, temperature=TEMPERATURE):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    base_prompt = (
        f"Below are tweets from a Twitter user:\n\n"
        + "\n".join([f"{i+1}. {tweet}" for i, tweet in enumerate(tweets)]) +
        f"\n\nUser Bio: {user_context.get('description', '')}\n"
        f"Followings: {user_context.get('following_count', 0)}\n"
        f"Followers: {user_context.get('followers_count', 0)}\n\n"
        "Generate exactly ONE new tweet this user could post (max 280 chars). "
        "Do NOT repeat tweets above.\nTweet:"
    )

    inputs = tokenizer(
        base_prompt,
        return_tensors='pt',
        max_length=1024,
        truncation=True,
        padding=True
    ).to(device)

    generated_tweets = []
    unwanted_prefix = "Below are tweets from a Twitter user:"
    for _ in range(num_tweets):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=60,  # shorter since it's just one tweet
                num_beams=1,  # reduce beams for clarity
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                early_stopping=True
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        # Remove the unwanted prefix if it appears
        if decoded.startswith(unwanted_prefix):
            decoded = decoded[len(unwanted_prefix):].strip()
        generated_tweets.append(decoded)

    return generated_tweets

# Main processing loop (only two users)
user_results = []
total_users = len(previous_results["user_results"])

for idx, user in enumerate(previous_results["user_results"], 1):
    user_id = user["user_id"]
    prompt_tweets = user["prompt_tweets"]
    non_prompt_tweets = user["non_prompt_tweets"]
    user_context = {
        "description": user.get("user_description", ""),
        "following_count": user.get("following_count", 0),
        "followers_count": user.get("followers_count", 0)
    }

    print(f"\n⏳ Processing user {user_id} ({idx}/{total_users})")

    generated_tweets = generate_bart_tweets(prompt_tweets, user_context)

    scenarios = {"all": prompt_tweets + non_prompt_tweets, "prompt": prompt_tweets, "nonprompt": non_prompt_tweets}

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

# Save final JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"user_results": user_results}, f, indent=4, ensure_ascii=False)

print(f"\n✅ Test run with 2 users complete. Results saved to '{OUTPUT_JSON}'")

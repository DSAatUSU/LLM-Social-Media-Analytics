"""Statistical analysis for Task I (authorship verification) reported in the paper.

Reads results/task1/weighted_metrics_all_models.csv (12 models x 15 sampling
settings) and reproduces:

  * Table II  - weighted F1 per (model, setting), per-model average and average rank
  * Wilcoxon signed-rank tests across the 15 settings (GPT-4 vs Gemini, GPT-4 vs
    DeepSeek, Gemini vs DeepSeek)
  * Table III - positive-user effect and negative-post effect (pp) per LLM
  * Friedman tests + Kendall's W (tie-corrected) for the two sampling dimensions
    over the five LLMs
  * average weighted F1 per negative-post sampling strategy
  * number of model-setting cells where a model collapsed to a single class

Usage:  python scripts/stats_tests.py [--csv results/task1/weighted_metrics_all_models.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

LLMS = ["GPT-4", "Gemini", "DeepSeek", "GPT-3.5-Turbo", "Llama3.2"]
PRETTY = {"Llama3.2": "Llama", "Bert": "BERT", "Gzip": "Compression-NCD", "SIAMESE_SBERT": "Siamese+SBERT",
          "Siamese_GloVe": "Siamese+GloVe"}


def kendall_w(block):
    """Kendall's coefficient of concordance W with tie correction.
    `block` is a DataFrame: rows = blocks (judges), columns = conditions."""
    r = block.rank(axis=1)
    n, k = r.shape
    R = r.sum(axis=0)
    S = ((R - n * (k + 1) / 2) ** 2).sum()
    T = sum(sum(t ** 3 - t for t in row.value_counts().values) for _, row in r.iterrows())
    return 12 * S / (n ** 2 * (k ** 3 - k) - n * T)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "..", "results", "task1", "weighted_metrics_all_models.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["wf1"] = df["weighted_avg_f1"] * 100
    df["model"] = df["model_name"].replace(PRETTY)

    # ---- Table II -------------------------------------------------------------
    table = df.pivot_table(index="model", columns=["other_tweets_sampling_method", "user_sampling_method"], values="wf1")
    avg = table.mean(axis=1)
    rank = table.rank(ascending=False, axis=0).mean(axis=1)
    summary = pd.DataFrame({"avg_weighted_F1": avg.round(1), "avg_rank": rank.round(2)}).sort_values("avg_rank")
    print("=== Table II: average weighted F1 (x100) and average rank over the 15 settings ===")
    print(summary.to_string())
    print("\nPer-setting weighted F1 (x100):")
    print(table.round(0).astype(int).to_string())

    # ---- Wilcoxon signed-rank tests ---------------------------------------------
    print("\n=== Wilcoxon signed-rank tests over the 15 settings ===")
    for a, b in [("GPT-4", "Gemini"), ("GPT-4", "DeepSeek"), ("Gemini", "DeepSeek")]:
        res = stats.wilcoxon(table.loc[a], table.loc[b])
        wins = int((table.loc[a] > table.loc[b]).sum())
        print(f"{a} vs {b}: W={res.statistic:.1f}, p={res.pvalue:.4f}  ({a} better in {wins}/15 settings)")

    # ---- Table III: sampling effects -------------------------------------------
    print("\n=== Table III: user effect / tweet effect (pp) ===")
    rows = []
    for m in [PRETTY.get(x, x) for x in LLMS]:
        sub = df[df.model == m].pivot(index="other_tweets_sampling_method", columns="user_sampling_method", values="wf1")
        user_eff = (sub.max(axis=1) - sub.min(axis=1)).mean()   # range over user strategies, averaged over tweet strategies
        tweet_eff = (sub.max(axis=0) - sub.min(axis=0)).mean()  # range over tweet strategies, averaged over user strategies
        rows.append({"model": m, "user_effect_pp": round(user_eff, 2), "tweet_effect_pp": round(tweet_eff, 2)})
    eff = pd.DataFrame(rows).set_index("model")
    eff.loc["Average"] = eff.mean().round(2)
    print(eff.to_string())

    # ---- Friedman tests + Kendall's W over the five LLMs ------------------------
    print("\n=== Friedman tests / Kendall's W (five LLMs) ===")
    d = df[df.model.isin([PRETTY.get(x, x) for x in LLMS])]
    for name, idx, col in [("positive user sampling", ["model", "other_tweets_sampling_method"], "user_sampling_method"),
                           ("negative post sampling", ["model", "user_sampling_method"], "other_tweets_sampling_method")]:
        block = d.pivot_table(index=idx, columns=col, values="wf1")
        fr = stats.friedmanchisquare(*[block[c] for c in block.columns])
        print(f"{name}: Friedman chi2={fr.statistic:.2f}, p={fr.pvalue:.2e}, Kendall's W={kendall_w(block):.2f} "
              f"(blocks = {len(block)} model x other-dimension combinations)")

    # ---- Difficulty of negative sampling strategies -----------------------------
    print("\n=== Average weighted F1 per negative post sampling strategy (all 12 models) ===")
    print((df.groupby("other_tweets_sampling_method").wf1.mean() / 100).round(2).to_string())

    # ---- Single-class collapse ---------------------------------------------------
    collapsed = df[(df["recall_0"] == 0) | (df["recall_1"] == 0)]
    near = df[(df["recall_0"] <= 0.01) | (df["recall_1"] <= 0.01)]
    print(f"\n=== Single-class collapse (recall of one class = 0): {len(collapsed)} of {len(df)} model-setting cells; "
          f"near-collapse (recall <= 1%): {len(near)} ===")
    print(near.groupby("model").size().to_string())


if __name__ == "__main__":
    main()

"""
Scoring Sensitivity Analysis for KFinEval Rebuttal
- How stable are model rankings when using different scoring dimensions?
- Does overall_quality depend on a single dimension or multiple?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import glob

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results/2_fin_reasoning")
OUTPUT_DIR = Path(__file__).parent

DIMS = ["coherence", "consistency", "accuracy", "completeness", "reasoning", "overall_quality"]


def load_model_dim_means():
    """Load per-model mean scores for each dimension."""
    rows = []
    for f in sorted(RESULTS_DIR.glob("*_eval.csv")):
        model = f.stem.replace("2_fin_reasoning_", "").replace("_eval", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        row = {"model": model}
        for dim in DIMS:
            row[dim] = pd.to_numeric(df[dim], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def analysis_1_rank_correlation(df):
    """Spearman rank correlation between overall_quality ranking and each dimension ranking."""
    print("=== Analysis 1: Rank Correlation (each dim vs overall_quality) ===\n")

    overall_rank = df["overall_quality"].rank(ascending=False)
    results = []

    for dim in DIMS:
        if dim == "overall_quality":
            continue
        dim_rank = df[dim].rank(ascending=False)
        rho, p = stats.spearmanr(overall_rank, dim_rank)
        results.append({
            "dimension": dim,
            "spearman_rho": round(rho, 3),
            "p_value": p,
        })

    result_df = pd.DataFrame(results).sort_values("spearman_rho", ascending=False)
    print("| Dimension | Spearman ρ (vs overall_quality) | p-value |")
    print("|---|---|---|")
    for _, r in result_df.iterrows():
        p_str = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        print(f"| {r['dimension']} | {r['spearman_rho']} | {p_str} |")

    result_df.to_csv(OUTPUT_DIR / "rank_correlation_vs_overall.csv", index=False)
    return result_df


def analysis_2_pairwise_dim_correlation(df):
    """Pairwise Spearman rank correlation between all dimensions."""
    print("\n=== Analysis 2: Pairwise Dimension Rank Correlation ===\n")

    matrix = {}
    for d1 in DIMS:
        matrix[d1] = {}
        for d2 in DIMS:
            rho, _ = stats.spearmanr(df[d1].rank(), df[d2].rank())
            matrix[d1][d2] = round(rho, 3)

    matrix_df = pd.DataFrame(matrix)
    print(matrix_df.to_string())
    matrix_df.to_csv(OUTPUT_DIR / "pairwise_dim_correlation.csv")
    return matrix_df


def analysis_3_alternative_rankings(df):
    """How many rank swaps occur when using each dimension instead of overall_quality?"""
    print("\n=== Analysis 3: Ranking Stability per Dimension ===\n")

    overall_rank = df["overall_quality"].rank(ascending=False)
    results = []

    for dim in DIMS:
        if dim == "overall_quality":
            continue
        dim_rank = df[dim].rank(ascending=False)

        # Count rank changes
        rank_diff = (overall_rank - dim_rank).abs()
        mean_shift = rank_diff.mean()
        max_shift = rank_diff.max()

        # Top-5 stability: how many of overall top-5 remain in dim top-5?
        overall_top5 = set(df.nlargest(5, "overall_quality")["model"])
        dim_top5 = set(df.nlargest(5, dim)["model"])
        top5_overlap = len(overall_top5 & dim_top5)

        # Top-10 stability
        overall_top10 = set(df.nlargest(10, "overall_quality")["model"])
        dim_top10 = set(df.nlargest(10, dim)["model"])
        top10_overlap = len(overall_top10 & dim_top10)

        results.append({
            "dimension": dim,
            "mean_rank_shift": round(mean_shift, 1),
            "max_rank_shift": int(max_shift),
            "top5_overlap": f"{top5_overlap}/5",
            "top10_overlap": f"{top10_overlap}/10",
        })

    result_df = pd.DataFrame(results)
    print("| Dimension | Mean Rank Shift | Max Rank Shift | Top-5 Overlap | Top-10 Overlap |")
    print("|---|---|---|---|---|")
    for _, r in result_df.iterrows():
        print(f"| {r['dimension']} | {r['mean_rank_shift']} | {r['max_rank_shift']} | {r['top5_overlap']} | {r['top10_overlap']} |")

    result_df.to_csv(OUTPUT_DIR / "alternative_rankings.csv", index=False)
    return result_df


def analysis_4_weighted_sensitivity(df):
    """How do different weighting schemes affect model rankings?"""
    print("\n=== Analysis 4: Weighted Score Sensitivity ===\n")

    sub_dims = ["coherence", "consistency", "accuracy", "completeness", "reasoning"]
    overall_rank = df["overall_quality"].rank(ascending=False)

    schemes = {
        "equal_weight": {d: 1.0 for d in sub_dims},
        "accuracy_heavy": {"coherence": 0.5, "consistency": 0.5, "accuracy": 3.0, "completeness": 1.0, "reasoning": 1.0},
        "reasoning_heavy": {"coherence": 0.5, "consistency": 0.5, "accuracy": 1.0, "completeness": 1.0, "reasoning": 3.0},
        "completeness_heavy": {"coherence": 0.5, "consistency": 0.5, "accuracy": 1.0, "completeness": 3.0, "reasoning": 1.0},
    }

    results = []
    for name, weights in schemes.items():
        total_w = sum(weights.values())
        weighted_score = sum(df[d] * (w / total_w) for d, w in weights.items())
        weighted_rank = weighted_score.rank(ascending=False)
        rho, p = stats.spearmanr(overall_rank, weighted_rank)
        tau, _ = stats.kendalltau(overall_rank, weighted_rank)
        results.append({
            "scheme": name,
            "spearman_rho": round(rho, 3),
            "kendall_tau": round(tau, 3),
            "p_value": p,
        })

    result_df = pd.DataFrame(results)
    print("| Weighting Scheme | Spearman ρ | Kendall τ | p-value |")
    print("|---|---|---|---|")
    for _, r in result_df.iterrows():
        p_str = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        print(f"| {r['scheme']} | {r['spearman_rho']} | {r['kendall_tau']} | {p_str} |")

    result_df.to_csv(OUTPUT_DIR / "weighted_sensitivity.csv", index=False)
    return result_df


if __name__ == "__main__":
    print("Loading reasoning evaluation data...\n")
    df = load_model_dim_means()
    print(f"Models: {len(df)}\n")

    r1 = analysis_1_rank_correlation(df)
    r2 = analysis_2_pairwise_dim_correlation(df)
    r3 = analysis_3_alternative_rankings(df)
    r4 = analysis_4_weighted_sensitivity(df)

    print(f"\nResults saved to {OUTPUT_DIR}")

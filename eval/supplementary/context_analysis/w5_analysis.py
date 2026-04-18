"""
W5 Rebuttal Analysis: Explanatory analysis beyond descriptive results
1. Context configuration degradation by model strength groups
2. Error dimension breakdown (which sub-dimension degrades first)
3. Model family weak-spot patterns
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results")
OUTPUT_DIR = Path("/home/work/kftc_model/KFinEval/_manuscript/rebuttal/w5_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REASONING_DIMS = ["coherence", "consistency", "accuracy", "completeness", "reasoning", "overall_quality"]

# Context config display order (from least to most challenging)
CONTEXT_ORDER = [
    "context_relevant_only",
    "context_relevant_only_shuffled",
    "context_relevant_front",
    "context_relevant_middle",
    "context_relevant_end",
    "context_relevant_scattered",
    "context_relevant_dispersed",
    "context_relevant_middle_with_en_noise",
]


def load_all_reasoning():
    pattern = str(RESULTS_DIR / "2_fin_reasoning" / "*_eval.csv")
    files = sorted(glob.glob(pattern))
    all_dfs = []
    for f in files:
        model = Path(f).stem.replace("2_fin_reasoning_", "").replace("_eval", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        for dim in REASONING_DIMS:
            df[dim] = pd.to_numeric(df[dim], errors="coerce")
        df["model"] = model
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def analysis_1_context_degradation(df):
    """Group models by strength, show per-context mean scores and degradation."""
    model_overall = df.groupby("model")["overall_quality"].mean().reset_index()
    model_overall.columns = ["model", "overall_mean"]

    # Categorize models into strong/medium/weak
    q33 = model_overall["overall_mean"].quantile(0.33)
    q66 = model_overall["overall_mean"].quantile(0.66)
    model_overall["group"] = pd.cut(
        model_overall["overall_mean"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Weak", "Medium", "Strong"]
    )

    df = df.merge(model_overall[["model", "group"]], on="model")

    # Per group, per context mean
    pivot = df.groupby(["group", "category"])["overall_quality"].mean().unstack(level=0)
    pivot = pivot.reindex(CONTEXT_ORDER)

    # Compute degradation from baseline (context_relevant_only)
    baseline = pivot.loc["context_relevant_only"]
    degradation = pivot.apply(lambda row: ((row - baseline) / baseline * 100), axis=1)

    pivot.to_csv(OUTPUT_DIR / "context_degradation_scores.csv", float_format="%.2f")
    degradation.to_csv(OUTPUT_DIR / "context_degradation_pct.csv", float_format="%.2f")

    print("\n=== Analysis 1: Context Degradation by Model Group ===")
    print("\nMean Scores:")
    print(pivot.round(2).to_string())
    print("\nDegradation (% from baseline):")
    print(degradation.round(2).to_string())

    return pivot, degradation


def analysis_2_dimension_breakdown(df):
    """Which sub-dimension degrades first under noise/dispersion?"""
    # Compare baseline vs most challenging contexts
    baseline = df[df["category"] == "context_relevant_only"]
    noisy = df[df["category"] == "context_relevant_middle_with_en_noise"]
    dispersed = df[df["category"] == "context_relevant_dispersed"]

    results = []
    for dim in REASONING_DIMS:
        base_mean = baseline[dim].mean()
        noise_mean = noisy[dim].mean()
        disp_mean = dispersed[dim].mean()
        results.append({
            "dimension": dim,
            "baseline": round(base_mean, 2),
            "with_noise": round(noise_mean, 2),
            "noise_drop": round(noise_mean - base_mean, 2),
            "noise_drop_pct": round((noise_mean - base_mean) / base_mean * 100, 2),
            "dispersed": round(disp_mean, 2),
            "disp_drop": round(disp_mean - base_mean, 2),
            "disp_drop_pct": round((disp_mean - base_mean) / base_mean * 100, 2),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "dimension_degradation.csv", index=False)

    print("\n=== Analysis 2: Dimension Degradation ===")
    print(result_df.to_string(index=False))

    return result_df


def analysis_3_model_family_patterns(df):
    """Do different model families have different weak contexts?"""
    # Define model families
    family_map = {}
    for model in df["model"].unique():
        ml = model.lower()
        if "gemma" in ml:
            family_map[model] = "Gemma"
        elif "qwen" in ml:
            family_map[model] = "Qwen"
        elif "gpt" in ml or "o1" in ml or "o3" in ml or "o4" in ml:
            family_map[model] = "GPT"
        elif "claude" in ml:
            family_map[model] = "Claude"
        elif "mistral" in ml or "ministral" in ml:
            family_map[model] = "Mistral"
        elif "llama" in ml:
            family_map[model] = "Llama"
        elif "exaone" in ml:
            family_map[model] = "EXAONE"
        elif "deepseek" in ml:
            family_map[model] = "DeepSeek"
        else:
            family_map[model] = "Other"

    df["family"] = df["model"].map(family_map)

    # Per family, find worst and best context
    family_ctx = df.groupby(["family", "category"])["overall_quality"].mean().reset_index()
    results = []
    for fam in sorted(df["family"].unique()):
        fam_data = family_ctx[family_ctx["family"] == fam]
        if len(fam_data) < 3:
            continue
        best = fam_data.loc[fam_data["overall_quality"].idxmax()]
        worst = fam_data.loc[fam_data["overall_quality"].idxmin()]
        mean_score = fam_data["overall_quality"].mean()
        results.append({
            "family": fam,
            "n_models": df[df["family"] == fam]["model"].nunique(),
            "mean_score": round(mean_score, 2),
            "best_context": best["category"],
            "best_score": round(best["overall_quality"], 2),
            "worst_context": worst["category"],
            "worst_score": round(worst["overall_quality"], 2),
            "spread": round(best["overall_quality"] - worst["overall_quality"], 2),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "family_weakspot_patterns.csv", index=False)

    print("\n=== Analysis 3: Model Family Weak-Spot Patterns ===")
    print(result_df.to_string(index=False))

    return result_df


def analysis_4_scale_vs_sensitivity(df):
    """Correlation between model scale (params) and context sensitivity."""
    import re

    # Extract parameter size from model name
    def extract_params(name):
        # Match patterns like "70B", "3-27b", "4B", "120b", "1.2B", "236B"
        matches = re.findall(r'(\d+\.?\d*)[bB]', name)
        if matches:
            return max(float(m) for m in matches)  # take largest match
        return None

    model_stats = df.groupby("model").agg(
        overall_mean=("overall_quality", "mean"),
    ).reset_index()

    # Per-model: baseline vs middle degradation
    baseline_scores = df[df["category"] == "context_relevant_only"].groupby("model")["overall_quality"].mean()
    middle_scores = df[df["category"] == "context_relevant_middle"].groupby("model")["overall_quality"].mean()

    model_stats["baseline"] = model_stats["model"].map(baseline_scores)
    model_stats["middle"] = model_stats["model"].map(middle_scores)
    model_stats["degradation_abs"] = model_stats["middle"] - model_stats["baseline"]
    model_stats["degradation_pct"] = (model_stats["degradation_abs"] / model_stats["baseline"]) * 100
    model_stats["params_b"] = model_stats["model"].apply(extract_params)

    # Filter models with known params
    known = model_stats.dropna(subset=["params_b", "baseline", "middle"]).copy()

    if len(known) > 5:
        from scipy import stats
        # Correlation: params vs degradation %
        r_deg, p_deg = stats.pearsonr(np.log10(known["params_b"]), known["degradation_pct"])
        # Correlation: params vs overall mean
        r_ovr, p_ovr = stats.pearsonr(np.log10(known["params_b"]), known["overall_mean"])

        print("\n=== Analysis 4: Model Scale vs Context Sensitivity ===")
        print(f"Models with known params: {len(known)}")
        print(f"log10(params) vs degradation%: Pearson r={r_deg:.3f}, p={p_deg:.4f}")
        print(f"log10(params) vs overall_mean: Pearson r={r_ovr:.3f}, p={p_ovr:.4f}")

        # Show representative models sorted by params
        known_sorted = known.sort_values("params_b")
        display_cols = ["model", "params_b", "overall_mean", "baseline", "middle", "degradation_pct"]
        print("\nRepresentative models (sorted by scale):")
        print(known_sorted[display_cols].round(2).to_string(index=False))

        known_sorted[display_cols].to_csv(OUTPUT_DIR / "scale_vs_sensitivity.csv", index=False)

        return r_deg, p_deg, r_ovr, p_ovr, known_sorted
    else:
        print("Not enough models with known params for correlation analysis")
        return None


if __name__ == "__main__":
    print("Loading reasoning data...")
    df = load_all_reasoning()
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models")

    pivot, degradation = analysis_1_context_degradation(df)
    dim_df = analysis_2_dimension_breakdown(df)
    family_df = analysis_3_model_family_patterns(df)
    scale_result = analysis_4_scale_vs_sensitivity(df)

    print(f"\nResults saved to {OUTPUT_DIR}")

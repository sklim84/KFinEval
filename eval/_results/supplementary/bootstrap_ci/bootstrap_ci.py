"""
Bootstrap Confidence Interval Analysis for KFinEval Rebuttal
- Knowledge: category-level accuracy 95% CI
- Reasoning: category-level overall_quality 95% CI
- Toxicity: category-level score 95% CI
"""

import os
import glob
import pandas as pd
import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results")
OUTPUT_DIR = Path("/home/work/kftc_model/KFinEval/_manuscript/rebuttal/bootstrap_results")
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
SEED = 42


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL, statistic=np.mean):
    rng = np.random.RandomState(SEED)
    n = len(values)
    boot_stats = np.array([
        statistic(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci_level) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    return float(np.mean(values)), float(lower), float(upper)


def process_knowledge():
    pattern = str(RESULTS_DIR / "1_fin_knowledge" / "*_response.csv")
    files = sorted(glob.glob(pattern))
    all_results = []

    for f in files:
        model_name = Path(f).stem.replace("1_fin_knowledge_", "").replace("_response", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        df["is_correct_bool"] = df["is_correct"].astype(str).str.strip().str.lower() == "true"

        # Overall
        vals = df["is_correct_bool"].astype(float).values
        mean, lower, upper = bootstrap_ci(vals)
        all_results.append({
            "model": model_name, "category": "OVERALL",
            "n": len(vals), "mean": round(mean, 4),
            "ci_lower": round(lower, 4), "ci_upper": round(upper, 4)
        })

        # Per category
        for cat, grp in df.groupby("category"):
            vals = grp["is_correct_bool"].astype(float).values
            mean, lower, upper = bootstrap_ci(vals)
            all_results.append({
                "model": model_name, "category": cat,
                "n": len(vals), "mean": round(mean, 4),
                "ci_lower": round(lower, 4), "ci_upper": round(upper, 4)
            })

    return pd.DataFrame(all_results)


def process_reasoning():
    pattern = str(RESULTS_DIR / "2_fin_reasoning" / "*_eval.csv")
    files = sorted(glob.glob(pattern))
    all_results = []

    for f in files:
        model_name = Path(f).stem.replace("2_fin_reasoning_", "").replace("_eval", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        df["overall_quality"] = pd.to_numeric(df["overall_quality"], errors="coerce")
        df = df.dropna(subset=["overall_quality"])

        # Overall
        vals = df["overall_quality"].values
        if len(vals) == 0:
            continue
        mean, lower, upper = bootstrap_ci(vals)
        all_results.append({
            "model": model_name, "category": "OVERALL",
            "n": len(vals), "mean": round(mean, 4),
            "ci_lower": round(lower, 4), "ci_upper": round(upper, 4)
        })

        # Per category
        for cat, grp in df.groupby("category"):
            vals = grp["overall_quality"].values
            if len(vals) == 0:
                continue
            mean, lower, upper = bootstrap_ci(vals)
            all_results.append({
                "model": model_name, "category": cat,
                "n": len(vals), "mean": round(mean, 4),
                "ci_lower": round(lower, 4), "ci_upper": round(upper, 4)
            })

    return pd.DataFrame(all_results)


def process_toxicity():
    pattern = str(RESULTS_DIR / "3_fin_toxicity" / "*_eval.csv")
    files = sorted(glob.glob(pattern))
    all_results = []

    for f in files:
        model_name = Path(f).stem.replace("3_fin_toxicity_", "").replace("_eval", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["score"])

        # Overall
        vals = df["score"].values
        if len(vals) == 0:
            continue
        mean, lower, upper = bootstrap_ci(vals)
        all_results.append({
            "model": model_name, "category": "OVERALL",
            "n": len(vals), "mean": round(mean, 4),
            "ci_lower": round(lower, 4), "ci_upper": round(upper, 4)
        })

        # Per category
        for cat, grp in df.groupby("category"):
            vals = grp["score"].values
            if len(vals) == 0:
                continue
            mean, lower, upper = bootstrap_ci(vals)
            all_results.append({
                "model": model_name, "category": cat,
                "n": len(vals), "mean": round(mean, 4),
                "ci_lower": round(lower, 4), "ci_upper": round(upper, 4)
            })

    return pd.DataFrame(all_results)


def generate_summary(knowledge_df, reasoning_df, toxicity_df):
    lines = ["# Bootstrap 95% CI Summary\n"]
    lines.append(f"- Bootstrap iterations: {N_BOOTSTRAP}")
    lines.append(f"- Confidence level: {CI_LEVEL * 100}%")
    lines.append(f"- Seed: {SEED}\n")

    for domain, df in [("Knowledge (Accuracy)", knowledge_df),
                       ("Reasoning (Overall Quality)", reasoning_df),
                       ("Toxicity (Score)", toxicity_df)]:
        lines.append(f"## {domain}\n")
        overall = df[df["category"] == "OVERALL"]
        n_models = overall["model"].nunique()
        cats = df[df["category"] != "OVERALL"]["category"].unique()
        lines.append(f"- Models: {n_models}")
        lines.append(f"- Categories: {len(cats)}\n")

        # Category-level sample size summary
        cat_df = df[df["category"] != "OVERALL"]
        if len(cat_df) > 0:
            cat_sizes = cat_df.groupby("category")["n"].first()
            lines.append("| Category | N | Mean CI Width (avg across models) |")
            lines.append("|---|---|---|")
            for cat in sorted(cats):
                cat_rows = cat_df[cat_df["category"] == cat]
                n_samples = cat_rows["n"].iloc[0] if len(cat_rows) > 0 else 0
                avg_width = (cat_rows["ci_upper"] - cat_rows["ci_lower"]).mean()
                lines.append(f"| {cat} | {n_samples} | {avg_width:.4f} |")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Processing knowledge...")
    knowledge_df = process_knowledge()
    knowledge_df.to_csv(OUTPUT_DIR / "bootstrap_ci_knowledge.csv", index=False)
    print(f"  → {len(knowledge_df)} rows")

    print("Processing reasoning...")
    reasoning_df = process_reasoning()
    reasoning_df.to_csv(OUTPUT_DIR / "bootstrap_ci_reasoning.csv", index=False)
    print(f"  → {len(reasoning_df)} rows")

    print("Processing toxicity...")
    toxicity_df = process_toxicity()
    toxicity_df.to_csv(OUTPUT_DIR / "bootstrap_ci_toxicity.csv", index=False)
    print(f"  → {len(toxicity_df)} rows")

    summary = generate_summary(knowledge_df, reasoning_df, toxicity_df)
    with open(OUTPUT_DIR / "bootstrap_ci_summary.md", "w") as f:
        f.write(summary)

    print(f"\nDone! Results saved to {OUTPUT_DIR}")

"""
Wilson Score 95% Confidence Interval Analysis for KFinEval.

Binomial accuracy (Financial Knowledge) is the natural target for Wilson Score
intervals. For each (model, category) pair we compute the Wilson 95% CI for
the accuracy proportion and report:
  - per-cell CI bounds
  - per-category mean CI width averaged across models (comparable to
    bootstrap_ci_summary.md format)
"""

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results")
OUTPUT_DIR = Path("/home/work/kftc_model/KFinEval/eval/supplementary/wilson_ci")

Z_95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Wilson Score 95% CI for a binomial proportion.

    Returns (lower, upper, width).
    For n == 0, returns (0.0, 0.0, 0.0).
    For k == 0 or k == n, Wilson still yields a valid non-degenerate interval
    (unlike the Wald interval), which is a key reason for preferring it here.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + (z * z) / (4.0 * n * n))
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return lower, upper, upper - lower


def process_knowledge() -> pd.DataFrame:
    pattern = str(RESULTS_DIR / "1_fin_knowledge" / "*_response.csv")
    files = sorted(glob.glob(pattern))
    rows = []

    for f in files:
        model_name = (
            Path(f).stem.replace("1_fin_knowledge_", "").replace("_response", "")
        )
        df = pd.read_csv(f, encoding="utf-8-sig")
        df["is_correct_bool"] = (
            df["is_correct"].astype(str).str.strip().str.lower() == "true"
        )

        # Overall
        n = int(len(df))
        k = int(df["is_correct_bool"].sum())
        lo, hi, w = wilson_ci(k, n)
        rows.append(
            {
                "model": model_name,
                "category": "OVERALL",
                "n": n,
                "k": k,
                "p_hat": round(k / n, 4) if n > 0 else 0.0,
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "ci_width": round(w, 4),
            }
        )

        # Per category
        for cat, grp in df.groupby("category"):
            n = int(len(grp))
            k = int(grp["is_correct_bool"].sum())
            lo, hi, w = wilson_ci(k, n)
            rows.append(
                {
                    "model": model_name,
                    "category": cat,
                    "n": n,
                    "k": k,
                    "p_hat": round(k / n, 4) if n > 0 else 0.0,
                    "ci_lower": round(lo, 4),
                    "ci_upper": round(hi, 4),
                    "ci_width": round(w, 4),
                }
            )

    return pd.DataFrame(rows)


def generate_summary(knowledge_df: pd.DataFrame) -> str:
    lines = ["# Wilson Score 95% CI Summary (Financial Knowledge)\n"]
    lines.append("- Method: Wilson Score interval (Wilson 1927)")
    lines.append(f"- Confidence level: 95%")
    lines.append(f"- z (two-sided, alpha=0.05): {Z_95:.6f}\n")

    overall = knowledge_df[knowledge_df["category"] == "OVERALL"]
    n_models = overall["model"].nunique()
    cat_df = knowledge_df[knowledge_df["category"] != "OVERALL"]
    cats = cat_df["category"].unique()

    lines.append(f"- Models: {n_models}")
    lines.append(f"- Categories: {len(cats)}\n")

    lines.append("| Category | N | Mean CI Width (avg across models) |")
    lines.append("|---|---|---|")
    for cat in sorted(cats):
        cat_rows = cat_df[cat_df["category"] == cat]
        n_samples = int(cat_rows["n"].iloc[0]) if len(cat_rows) > 0 else 0
        avg_width = float(cat_rows["ci_width"].mean())
        lines.append(f"| {cat} | {n_samples} | {avg_width:.4f} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Processing knowledge (Wilson Score CI)...")
    knowledge_df = process_knowledge()
    knowledge_df.to_csv(OUTPUT_DIR / "wilson_ci_knowledge.csv", index=False)
    print(f"  -> {len(knowledge_df)} rows")

    summary = generate_summary(knowledge_df)
    with open(OUTPUT_DIR / "wilson_ci_summary.md", "w") as f:
        f.write(summary)

    print(f"\nDone. Results saved to {OUTPUT_DIR}")

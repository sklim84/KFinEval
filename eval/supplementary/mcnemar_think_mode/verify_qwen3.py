"""
Verify Qwen3 Standard(Instruct) vs Think(Thinking) accuracy on Knowledge MCQ.
Also run McNemar's test for comparison with GPT models.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
from pathlib import Path

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results/1_fin_knowledge")

MODELS = [
    ("Qwen3-30B", "1_fin_knowledge_Qwen3-30B-A3B-Instruct-2507_response.csv",
     "1_fin_knowledge_Qwen3-30B-A3B-Thinking-2507_response.csv"),
    ("Qwen3-4B", "1_fin_knowledge_Qwen3-4B-Instruct-2507_response.csv",
     "1_fin_knowledge_Qwen3-4B-Thinking-2507_response.csv"),
]


def mcnemar_test(std_correct, think_correct):
    a = np.sum(std_correct & think_correct)
    b = np.sum(std_correct & ~think_correct)
    c = np.sum(~std_correct & think_correct)
    d = np.sum(~std_correct & ~think_correct)
    if (b + c) == 0:
        chi2_stat, p_value = 0.0, 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return a, b, c, d, chi2_stat, p_value


for model_name, std_file, think_file in MODELS:
    std_df = pd.read_csv(RESULTS_DIR / std_file, encoding="utf-8-sig")
    think_df = pd.read_csv(RESULTS_DIR / think_file, encoding="utf-8-sig")

    std_correct = std_df["is_correct"].astype(str).str.strip().str.lower() == "true"
    think_correct = think_df["is_correct"].astype(str).str.strip().str.lower() == "true"

    std_acc = std_correct.mean() * 100
    think_acc = think_correct.mean() * 100

    # Align by id if possible
    if "id" in std_df.columns and "id" in think_df.columns:
        merged = std_df[["id", "is_correct"]].merge(
            think_df[["id", "is_correct"]], on="id", suffixes=("_std", "_think"))
        s = merged["is_correct_std"].astype(str).str.strip().str.lower() == "true"
        t = merged["is_correct_think"].astype(str).str.strip().str.lower() == "true"
        a, b, c, d, chi2_stat, p_val = mcnemar_test(s.values, t.values)
    else:
        a, b, c, d, chi2_stat, p_val = mcnemar_test(std_correct.values, think_correct.values)

    print(f"\n--- {model_name} ---")
    print(f"  Instruct (Standard): {std_acc:.1f}% ({int(std_correct.sum())}/{len(std_correct)})")
    print(f"  Thinking (Think):    {think_acc:.1f}% ({int(think_correct.sum())}/{len(think_correct)})")
    print(f"  Δ: {think_acc - std_acc:+.1f}pp")
    print(f"  Contingency: a={a}, b={b}, c={c}, d={d}")
    print(f"  McNemar χ²={chi2_stat:.2f}, p={p_val:.6f}")
    print(f"  Think solved {c} that Standard failed; reverse: {b}")

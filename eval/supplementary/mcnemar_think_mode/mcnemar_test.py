"""
McNemar's Test: Standard vs Think mode on Financial Knowledge MCQ
Tests whether Think mode improvement is statistically significant.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
from pathlib import Path

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results/1_fin_knowledge")
OUTPUT_DIR = Path(__file__).parent

MODELS = [
    ("gpt-5.2", "1_fin_knowledge_gpt-5.2_response.csv", "1_fin_knowledge_gpt-5.2_reasoning_response.csv"),
    ("gpt-5-mini", "1_fin_knowledge_gpt-5-mini_response.csv", "1_fin_knowledge_gpt-5-mini_reasoning_response.csv"),
    ("gpt-5-nano", "1_fin_knowledge_gpt-5-nano_response.csv", "1_fin_knowledge_gpt-5-nano_reasoning_response.csv"),
]


def mcnemar_test(standard_correct, think_correct):
    """
    McNemar's test for paired binary data.

    Contingency table:
                    Think Correct  Think Wrong
    Standard Correct      a            b
    Standard Wrong        c            d

    H0: b == c (no difference between modes)
    χ² = (|b - c| - 1)² / (b + c)  (with continuity correction)
    """
    a = np.sum(standard_correct & think_correct)      # both correct
    b = np.sum(standard_correct & ~think_correct)     # standard correct, think wrong
    c = np.sum(~standard_correct & think_correct)     # standard wrong, think correct
    d = np.sum(~standard_correct & ~think_correct)    # both wrong

    n = len(standard_correct)

    # McNemar's chi-squared with continuity correction
    if (b + c) == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        "both_correct (a)": int(a),
        "standard_only (b)": int(b),
        "think_only (c)": int(c),
        "both_wrong (d)": int(d),
        "total": int(n),
        "chi2": round(chi2_stat, 2),
        "p_value": p_value,
    }


def main():
    results = []

    for model_name, std_file, think_file in MODELS:
        std_path = RESULTS_DIR / std_file
        think_path = RESULTS_DIR / think_file

        if not std_path.exists() or not think_path.exists():
            print(f"  Skipping {model_name}: file not found")
            continue

        std_df = pd.read_csv(std_path, encoding="utf-8-sig")
        think_df = pd.read_csv(think_path, encoding="utf-8-sig")

        # Align by id
        merged = std_df[["id", "is_correct"]].merge(
            think_df[["id", "is_correct"]],
            on="id",
            suffixes=("_std", "_think")
        )

        std_correct = merged["is_correct_std"].astype(str).str.strip().str.lower() == "true"
        think_correct = merged["is_correct_think"].astype(str).str.strip().str.lower() == "true"

        test_result = mcnemar_test(std_correct.values, think_correct.values)

        std_acc = std_correct.mean() * 100
        think_acc = think_correct.mean() * 100

        row = {
            "model": model_name,
            "standard_acc": round(std_acc, 1),
            "think_acc": round(think_acc, 1),
            "delta": round(think_acc - std_acc, 1),
            **test_result,
        }
        results.append(row)

        print(f"\n--- {model_name} ---")
        print(f"  Standard: {std_acc:.1f}%  Think: {think_acc:.1f}%  Δ: +{think_acc - std_acc:.1f}pp")
        print(f"  Contingency: a={test_result['both_correct (a)']}, b={test_result['standard_only (b)']}, c={test_result['think_only (c)']}, d={test_result['both_wrong (d)']}")
        print(f"  McNemar χ²={test_result['chi2']}, p={test_result['p_value']:.6f}")
        print(f"  Think solved {test_result['think_only (c)']} that Standard failed; reverse: {test_result['standard_only (b)']}")

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "mcnemar_results.csv", index=False)

    # Print markdown table
    print("\n\n## McNemar's Test Results\n")
    print("| Model | Standard% | Think% | Δ | Think→Correct (c) | Standard→Correct (b) | McNemar χ² | p-value |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in result_df.iterrows():
        p_str = f"<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        print(f"| {r['model']} | {r['standard_acc']} | {r['think_acc']} | +{r['delta']} | {r['think_only (c)']} | {r['standard_only (b)']} | {r['chi2']} | {p_str} |")

    print(f"\nResults saved to {OUTPUT_DIR / 'mcnemar_results.csv'}")


if __name__ == "__main__":
    main()

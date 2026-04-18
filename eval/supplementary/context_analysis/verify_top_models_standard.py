"""
Verify top model (Standard mode) context degradation for PmKU-W2 rebuttal.
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results/2_fin_reasoning")

MODELS = {
    "gpt-5.2": "2_fin_reasoning_gpt-5_2_eval.csv",
    "claude-opus-4.5": "2_fin_reasoning_claude-opus-4-5_eval.csv",
    "claude-sonnet-4.5": "2_fin_reasoning_claude-sonnet-4-5_eval.csv",
}

CONTEXTS = [
    "context_relevant_only",
    "context_relevant_middle",
    "context_relevant_dispersed",
    "context_relevant_end",
]

results = []
for name, fname in MODELS.items():
    df = pd.read_csv(RESULTS_DIR / fname, encoding="utf-8-sig")
    df["overall_quality"] = pd.to_numeric(df["overall_quality"], errors="coerce")
    baseline = df[df["category"] == "context_relevant_only"]["overall_quality"].mean()

    row = {"model": name, "overall_avg": round(df["overall_quality"].mean(), 2)}
    for ctx in CONTEXTS:
        sub = df[df["category"] == ctx]
        m = sub["overall_quality"].mean()
        drop = (m - baseline) / baseline * 100
        row[ctx] = round(m, 2)
        row[f"{ctx}_drop"] = round(drop, 1)
    results.append(row)

result_df = pd.DataFrame(results)
result_df.to_csv(Path(__file__).parent / "top_models_standard_context.csv", index=False)

print("| Model | relevant_only | middle | dispersed | end | Max Drop |")
print("|---|---|---|---|---|---|")
for _, r in result_df.iterrows():
    drops = [r.get(f"{c}_drop", 0) for c in CONTEXTS[1:]]
    max_drop = min(drops)
    print(f"| {r['model']} | {r['context_relevant_only']} | {r['context_relevant_middle']} | {r['context_relevant_dispersed']} | {r['context_relevant_end']} | {max_drop}% |")

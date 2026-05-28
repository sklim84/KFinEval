"""
Find cases where Standard mode answered wrong but Think mode answered correctly.
Identify the category with the highest frequency of such reversals.
Save a summary and pick representative case examples.
"""
import pandas as pd
import os
import json
from collections import Counter

BASE = "/home/work/kftc_model/KFinEval/eval/_results/1_fin_knowledge"
OUT = "/home/work/kftc_model/KFinEval/_manuscript/rebuttal/exp/mcnemar_think_mode"

# Use gpt-5.2 as representative (consistent with earlier rebuttal tables)
STD = os.path.join(BASE, "1_fin_knowledge_gpt-5.2_response.csv")
THK = os.path.join(BASE, "1_fin_knowledge_gpt-5.2_reasoning_response.csv")

std = pd.read_csv(STD)
thk = pd.read_csv(THK)

# Align by id
std = std.set_index("id")
thk = thk.set_index("id")

common = std.index.intersection(thk.index)
std = std.loc[common]
thk = thk.loc[common]

# Think→Correct: Standard wrong, Think correct
mask_think_wins = (std["is_correct"] == 0) & (thk["is_correct"] == 1)
mask_reverse   = (std["is_correct"] == 1) & (thk["is_correct"] == 0)

print(f"Total items compared: {len(common)}")
print(f"Think→Correct (Std wrong, Think right): {mask_think_wins.sum()}")
print(f"Reverse (Std right, Think wrong):       {mask_reverse.sum()}")

# Group reversals by category
cat_counts = std.loc[mask_think_wins, "category"].value_counts()
print("\n=== Think→Correct frequency by category ===")
print(cat_counts.to_string())

sub_counts = std.loc[mask_think_wins].groupby(["category", "sub_category"]).size().sort_values(ascending=False)
print("\n=== Think→Correct frequency by category × sub_category (top 15) ===")
print(sub_counts.head(15).to_string())

# Pick TOP category, then within it, pick 3 representative items
top_cat = cat_counts.index[0]
print(f"\n=== TOP category: {top_cat} ({cat_counts.iloc[0]} items) ===")

picks = std.loc[mask_think_wins & (std["category"] == top_cat)].head(5)

# Save detailed case examples
cases = []
for idx in picks.index:
    s = std.loc[idx]
    t = thk.loc[idx]
    cases.append({
        "id": int(idx),
        "category": s["category"],
        "sub_category": s["sub_category"],
        "level": s["level"],
        "question": s["question"],
        "choices": {k: s[k] for k in ["A", "B", "C", "D", "E"] if pd.notna(s.get(k))},
        "gold": s["gold"],
        "standard_answer": s["answer"],
        "standard_is_correct": int(s["is_correct"]),
        "standard_raw": str(s["raw_response"])[:1500],
        "think_answer": t["answer"],
        "think_is_correct": int(t["is_correct"]),
        "think_raw": str(t["raw_response"])[:3000],
    })

with open(os.path.join(OUT, "case_study_top_category.json"), "w", encoding="utf-8") as f:
    json.dump({"top_category": top_cat, "count": int(cat_counts.iloc[0]), "cases": cases},
              f, ensure_ascii=False, indent=2)

# Also save the aggregate stats
summary = {
    "total": int(len(common)),
    "think_wins": int(mask_think_wins.sum()),
    "reverse": int(mask_reverse.sum()),
    "category_counts": cat_counts.to_dict(),
    "sub_category_top15": {f"{a}|{b}": int(v) for (a,b), v in sub_counts.head(15).items()},
}
with open(os.path.join(OUT, "case_study_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\nSaved: case_study_top_category.json, case_study_summary.json")

"""
Verify top model context degradation for W2 rebuttal.
Check per-context-config mean scores for gpt-5.2, claude-opus-4.5, claude-sonnet-4.5.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results/2_fin_reasoning")

MODELS = {
    "gpt-5.2": "2_fin_reasoning_gpt-5.2_reasoning_eval.csv",
    "claude-opus-4.5": "2_fin_reasoning_claude-opus-4-5_eval.csv",
    "claude-sonnet-4.5": "2_fin_reasoning_claude-sonnet-4-5_eval.csv",
}

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

# Try alternative filenames if not found
ALT_NAMES = {
    "gpt-5.2": ["2_fin_reasoning_gpt-5.2_reasoning_eval.csv", "2_fin_reasoning_gpt-5.2_eval.csv"],
    "claude-opus-4.5": ["2_fin_reasoning_claude-opus-4-5_eval.csv", "2_fin_reasoning_claude-opus-4.5_eval.csv"],
    "claude-sonnet-4.5": ["2_fin_reasoning_claude-sonnet-4-5_eval.csv", "2_fin_reasoning_claude-sonnet-4.5_eval.csv"],
}

for model_name, alts in ALT_NAMES.items():
    found = False
    for fname in alts:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, encoding="utf-8-sig")
            df["overall_quality"] = pd.to_numeric(df["overall_quality"], errors="coerce")

            print(f"\n--- {model_name} ({fname}) ---")
            print(f"Total rows: {len(df)}")

            baseline = df[df["category"] == "context_relevant_only"]["overall_quality"].mean()

            for ctx in CONTEXT_ORDER:
                subset = df[df["category"] == ctx]
                if len(subset) == 0:
                    continue
                mean = subset["overall_quality"].mean()
                drop_pct = (mean - baseline) / baseline * 100 if baseline > 0 else 0
                print(f"  {ctx:45s}  N={len(subset):3d}  mean={mean:.2f}  drop={drop_pct:+.1f}%")

            found = True
            break

    if not found:
        print(f"\n--- {model_name}: FILE NOT FOUND ---")
        # List similar files
        import glob
        similar = glob.glob(str(RESULTS_DIR / f"*{model_name.split('-')[0].lower()}*eval*"))
        if similar:
            print(f"  Similar files: {[Path(s).name for s in similar[:5]]}")

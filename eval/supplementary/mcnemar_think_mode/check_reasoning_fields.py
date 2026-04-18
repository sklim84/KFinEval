"""
Inspect whether Think-mode results contain reasoning traces.
Check multiple models, not just gpt-5.2.
"""
import pandas as pd
import os
import json

BASE = "/home/work/kftc_model/KFinEval/eval/_results/1_fin_knowledge"

# Pick a variety of Think/reasoning files
models_to_check = [
    "1_fin_knowledge_gpt-5.2_reasoning_response.csv",
    "1_fin_knowledge_gpt-5-mini_reasoning_response.csv",
    "1_fin_knowledge_gpt-5-nano_reasoning_response.csv",
]

for fn in models_to_check:
    fp = os.path.join(BASE, fn)
    if not os.path.exists(fp):
        print(f"[MISSING] {fn}")
        continue
    df = pd.read_csv(fp)
    print(f"\n=== {fn} ===")
    print(f"rows: {len(df)}")
    # Check first row raw_response
    raw = str(df.iloc[0]["raw_response"])
    has_reasoning_field = '"reasoning"' in raw
    has_encrypted = 'reasoning.encrypted' in raw
    has_reasoning_text = False
    # Try to actually find text content
    try:
        # raw_response is a JSON string (sometimes a list)
        if raw.startswith("["):
            parsed = json.loads(raw)
            for item in parsed:
                choices = item.get("choices", [])
                for ch in choices:
                    msg = ch.get("message", {})
                    r = msg.get("reasoning")
                    if isinstance(r, str) and len(r) > 20:
                        has_reasoning_text = True
                        break
        else:
            # OpenAI-style response
            parsed = json.loads(raw)
            for out in parsed.get("output", []):
                if out.get("type") == "reasoning":
                    for c in out.get("content", []):
                        if c.get("text"):
                            has_reasoning_text = True
    except Exception as e:
        pass
    print(f"  has '\"reasoning\"' literal in raw: {has_reasoning_field}")
    print(f"  has encrypted reasoning: {has_encrypted}")
    print(f"  has accessible reasoning TEXT: {has_reasoning_text}")
    print(f"  raw[:300]: {raw[:300]}")

# Also check an open-source model (Qwen3) for comparison
qw = "1_fin_knowledge_Qwen3-4B_reasoning_response.csv"
fp = os.path.join(BASE, qw)
if os.path.exists(fp):
    df = pd.read_csv(fp)
    print(f"\n=== {qw} ===")
    print(f"rows: {len(df)}")
    raw = str(df.iloc[0]["raw_response"])
    print(f"  raw[:300]: {raw[:300]}")
else:
    # Try variants
    import glob
    matches = glob.glob(os.path.join(BASE, "*Qwen3*"))
    print("\nQwen3 files found:")
    for m in matches:
        print(f"  {os.path.basename(m)}")

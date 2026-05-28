"""
Verify the claim: 'The pattern (Standard answers without reasoning, Think applies
formulas step-by-step) is observed across all 125 Think->Correct cases.'

We examine:
1. Does Standard consistently output answer without reasoning?
2. Does Think consistently show multi-step reasoning?
3. Are there exceptions that would weaken the claim?
"""
import pandas as pd
import os
import json

BASE = "/home/work/kftc_model/KFinEval/eval/_results/1_fin_knowledge"
STD = os.path.join(BASE, "1_fin_knowledge_gpt-5.2_response.csv")
THK = os.path.join(BASE, "1_fin_knowledge_gpt-5.2_reasoning_response.csv")

std = pd.read_csv(STD).set_index("id")
thk = pd.read_csv(THK).set_index("id")
common = std.index.intersection(thk.index)
std = std.loc[common]
thk = thk.loc[common]

# 125 Think-wins items
mask = (std["is_correct"] == 0) & (thk["is_correct"] == 1)
win_ids = std.index[mask].tolist()
print(f"Think->Correct total: {len(win_ids)}")

# Helper: extract reasoning text from raw_response
def extract_reasoning(raw):
    s = str(raw)
    # OpenAI response-API style: output contains reasoning sections
    try:
        if s.startswith("{"):
            parsed = json.loads(s)
            texts = []
            for out in parsed.get("output", []):
                if out.get("type") == "reasoning":
                    for c in out.get("content", []):
                        t = c.get("text") or ""
                        if t:
                            texts.append(t)
            return " || ".join(texts)
        elif s.startswith("["):
            parsed = json.loads(s)
            texts = []
            for item in parsed:
                for ch in item.get("choices", []):
                    msg = ch.get("message", {})
                    r = msg.get("reasoning")
                    if isinstance(r, str) and r.strip():
                        texts.append(r)
            return " || ".join(texts)
    except Exception:
        pass
    return ""

# Helper: extract the actual answer/content text from Standard
def extract_content(raw):
    s = str(raw)
    try:
        if s.startswith("{"):
            parsed = json.loads(s)
            texts = []
            for out in parsed.get("output", []):
                if out.get("type") == "message":
                    for c in out.get("content", []):
                        t = c.get("text") or ""
                        if t:
                            texts.append(t)
            return " ".join(texts)
        elif s.startswith("["):
            parsed = json.loads(s)
            for item in parsed:
                for ch in item.get("choices", []):
                    return ch.get("message", {}).get("content", "")
    except Exception:
        pass
    return s[:200]

# Analyze all 125 items
std_has_reasoning = 0
std_answer_only = 0
std_long_content = 0
thk_has_reasoning = 0
thk_empty_reasoning = 0
thk_encrypted_only = 0

std_content_lengths = []
thk_reasoning_lengths = []

category_breakdown = {}

for idx in win_ids:
    s_raw = std.loc[idx, "raw_response"]
    t_raw = thk.loc[idx, "raw_response"]
    cat = std.loc[idx, "category"]

    # Standard: extract content
    s_content = extract_content(s_raw)
    s_reasoning = extract_reasoning(s_raw)
    # Standard model has no reasoning field typically; content should be JSON like {"answer":"X"}
    s_len = len(s_content.strip())
    std_content_lengths.append(s_len)
    if len(s_reasoning) > 20:
        std_has_reasoning += 1
    if s_len < 50:
        std_answer_only += 1
    if s_len > 200:
        std_long_content += 1

    # Think: extract reasoning
    t_reasoning = extract_reasoning(t_raw)
    t_len = len(t_reasoning.strip())
    thk_reasoning_lengths.append(t_len)
    if t_len > 50:
        thk_has_reasoning += 1
    elif t_len == 0:
        # Check if encrypted
        if "reasoning.encrypted" in str(t_raw):
            thk_encrypted_only += 1
        else:
            thk_empty_reasoning += 1

    category_breakdown.setdefault(cat, {"total": 0, "std_short": 0, "thk_reasoning": 0})
    category_breakdown[cat]["total"] += 1
    if s_len < 50:
        category_breakdown[cat]["std_short"] += 1
    if t_len > 50:
        category_breakdown[cat]["thk_reasoning"] += 1

print("\n=== STANDARD mode analysis (on 125 Think-wins) ===")
print(f"Items where Standard output is <50 chars (answer only): {std_answer_only}/125")
print(f"Items where Standard output is >200 chars (long content): {std_long_content}/125")
print(f"Items where Standard has reasoning field: {std_has_reasoning}/125")
print(f"Standard content length — min/median/max: {min(std_content_lengths)}/{sorted(std_content_lengths)[len(std_content_lengths)//2]}/{max(std_content_lengths)}")

print("\n=== THINK mode analysis (on 125 Think-wins) ===")
print(f"Items where Think has accessible reasoning (>50 chars): {thk_has_reasoning}/125")
print(f"Items where Think reasoning is ONLY encrypted: {thk_encrypted_only}/125")
print(f"Items where Think has empty reasoning: {thk_empty_reasoning}/125")
print(f"Think reasoning length — min/median/max: {min(thk_reasoning_lengths)}/{sorted(thk_reasoning_lengths)[len(thk_reasoning_lengths)//2]}/{max(thk_reasoning_lengths)}")

print("\n=== Category breakdown ===")
for cat, d in sorted(category_breakdown.items(), key=lambda x: -x[1]["total"]):
    print(f"  {cat}: total={d['total']}, std_short={d['std_short']}, thk_reasoning={d['thk_reasoning']}")

# Sample 3 items from non-재무관리 categories to see if pattern holds
print("\n=== Sample items from OTHER categories ===")
other_ids = [i for i in win_ids if std.loc[i, "category"] != "재무관리"]
import random
random.seed(42)
sample = random.sample(other_ids, min(5, len(other_ids)))
for idx in sample:
    s_content = extract_content(std.loc[idx, "raw_response"])
    t_reasoning = extract_reasoning(thk.loc[idx, "raw_response"])
    print(f"\n  id={idx}, cat={std.loc[idx,'category']}/{std.loc[idx,'sub_category']}")
    print(f"    Standard content: {s_content[:120]}")
    print(f"    Think reasoning (first 200 chars): {t_reasoning[:200]}")

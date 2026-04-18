"""
From the top category's cases, pick the single cleanest example:
- Has a clear reasoning trace
- Reasonable question length
- Clear contrast between Standard's wrong answer and Think's reasoning path
"""
import json

with open("/home/work/kftc_model/KFinEval/_manuscript/rebuttal/exp/mcnemar_think_mode/case_study_top_category.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["cases"]
print(f"Top category: {data['top_category']} ({data['count']} items)")
print(f"Number of example cases loaded: {len(cases)}\n")

for i, c in enumerate(cases):
    print(f"\n{'='*80}")
    print(f"CASE {i+1} — id={c['id']}, sub={c['sub_category']}, level={c['level']}")
    print(f"{'='*80}")
    print(f"Q: {c['question'][:400]}")
    print(f"Choices: {list(c['choices'].keys())}")
    print(f"Gold: {c['gold']}")
    print(f"Standard answer: {c['standard_answer']} (correct={c['standard_is_correct']})")
    print(f"Think answer: {c['think_answer']} (correct={c['think_is_correct']})")
    print(f"\n[Standard raw (first 400 chars)]")
    print(c['standard_raw'][:400])
    print(f"\n[Think raw (first 800 chars)]")
    print(c['think_raw'][:800])

import json

with open("/home/work/kftc_model/KFinEval/_manuscript/rebuttal/exp/mcnemar_think_mode/case_study_top_category.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Cases we want: id=69 (OCF) and id=72 (risk-free portfolio)
for c in data["cases"]:
    if c["id"] in (69, 72):
        print(f"\n{'='*80}")
        print(f"id={c['id']}, category={c['category']}, sub={c['sub_category']}, level={c['level']}")
        print(f"gold={c['gold']}, std={c['standard_answer']}, think={c['think_answer']}")
        print(f"{'='*80}")
        print("[QUESTION]")
        print(c["question"])
        print("\n[CHOICES]")
        for k, v in c["choices"].items():
            print(f"  {k}: {v}")

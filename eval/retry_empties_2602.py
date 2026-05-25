#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reasoning 2602 빈답변(8건) 재생성: reasoning OFF + per-request timeout.
content가 비면 reasoning 텍스트로 폴백. 8건 처리 후 answer CSV 전체 재기록."""
import csv, sys, json, time
import gen_gemini_2602_openrouter as G
from openai import OpenAI

csv.field_size_limit(sys.maxsize)

# 타임아웃 있는 클라이언트 (이전 hang 방지)
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=G.OPENROUTER_API_KEY, timeout=90.0, max_retries=2)

rows = list(csv.DictReader(open(G.REAS_OUT, encoding="utf-8-sig")))
empties = [r for r in rows if not (r.get("answer") or "").strip()]
print(f"빈답변 {len(empties)}건: {[r['id'] for r in empties]}", flush=True)

filled = 0
for r in empties:
    prompt = G.create_reasoning_prompt(r["context"], r["question"])
    got = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=G.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=G.MAX_TOKENS,
            )  # reasoning effort 미지정 => content 강제
            msg = resp.choices[0].message
            got = (msg.content or "").strip()
            if not got:
                got = (getattr(msg, "reasoning", None) or "").strip()
            if got:
                r["answer"] = got
                r["raw_response"] = json.dumps(resp.model_dump(), ensure_ascii=False)
                filled += 1
                print(f"  id={r['id']} -> len={len(got)}", flush=True)
                break
        except Exception as e:
            print(f"  id={r['id']} attempt{attempt} err: {e}", flush=True)
            time.sleep(2 ** attempt)
    if not got:
        print(f"  id={r['id']} 여전히 빈답변 (유지)", flush=True)

import pandas as pd
pd.DataFrame(rows, columns=G.REAS_COLS).to_csv(G.REAS_OUT, index=False, encoding="utf-8-sig")
left = sum(1 for r in rows if not (r.get("answer") or "").strip())
print(f"DONE filled={filled} rows={len(rows)} remaining_empty={left}", flush=True)

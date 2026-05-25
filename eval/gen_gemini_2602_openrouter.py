#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A안 재실험: gemini-3.1-pro-preview (2602) 지식/추론 답변 생성 (OpenRouter).

- 비파괴: 출력은 *_gemini-3.1-pro-preview_2602_* 로 별도 저장 (2511 파일 안 건드림)
- 문항 동일성: 입력으로 기존 2511 결과 파일의 문항(id/question/options/context/gold)을 그대로 사용
  -> 2511 vs 2602 apples-to-apples 비교 보장
- 프롬프트 동일: 1_1_eval_knowledge_openai.create_prompt / 2_1_gen_reasoning_openai.create_reasoning_prompt 와 동일
- 행 단위 idempotent resume (append-only), per-row raw_response 보존, 멀티스레드 + append lock

사용:
  python gen_gemini_2602_openrouter.py --domain knowledge --workers 4
  python gen_gemini_2602_openrouter.py --domain reasoning --workers 4
"""
import os, re, csv, json, time, argparse, threading, random, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

csv.field_size_limit(sys.maxsize)

# ---- .env (repo root) ----
ROOT = Path(__file__).resolve().parent.parent
for envp in (ROOT / ".env", Path(__file__).resolve().parent / ".env"):
    if envp.exists():
        from dotenv import load_dotenv
        load_dotenv(envp)
        break

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "OPENROUTER_API_KEY not found in .env"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

MODEL = "google/gemini-3.1-pro-preview"      # 2602 (살아있음 확인됨)
REASONING_EFFORT = "minimal"                  # toxicity 2602 run과 동일
MAX_TOKENS = 8192

RESULTS = Path(__file__).resolve().parent / "_results"
KNOW_SRC = RESULTS / "1_fin_knowledge" / "1_fin_knowledge_gemini-3-pro-preview_response.csv"
KNOW_OUT = RESULTS / "1_fin_knowledge" / "1_fin_knowledge_gemini-3.1-pro-preview_2602_response.csv"
REAS_SRC = RESULTS / "2_fin_reasoning" / "2_fin_reasoning_gemini-3-pro-preview_answer.csv"
REAS_OUT = RESULTS / "2_fin_reasoning" / "2_fin_reasoning_gemini-3.1-pro-preview_2602_answer.csv"

KNOW_COLS = ["id","category","sub_category","level","has_table","has_fomula","question",
             "A","B","C","D","E","gold","answer","answer_structured","raw_response","is_correct"]
REAS_COLS = ["id","source","category","question","context","gold","answer","raw_response"]

ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mcq_answer", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["A","B","C","D","E"]}},
            "required": ["answer"], "additionalProperties": False,
        },
    },
}

def create_prompt(q,a,b,c,d,e):
    return f"""다음 객관식 질문에 답하세요.

질문:
{q}

선택지:
A. {a}
B. {b}
C. {c}
D. {d}
E. {e}""".strip()

def create_reasoning_prompt(context, question):
    return f"""주어진 문맥을 기반으로 질문에 대한 답을 작성해주세요. 답은 반드시 결론을 포함한 추론과정을 단계별로 작성해야합니다.

문맥: {context}
질문: {question}

- 문체는 반드시 문어체로 작성해주세요.
- 추론과정에 거짓된 정보는 없어야 합니다.
- 반드시 한글로 작성되어야 합니다.

출력 형식은 JSON 으로 작성하며, 반드시 아래와 같은 예시를 따르세요:
{{
"step 1": "추론 내용 1",
"step 2": "추론 내용 2",
"step 3": "추론 내용 3",
...
}}
"""

def read_src(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_done(out_path):
    if not out_path.exists():
        return set()
    try:
        with open(out_path, encoding="utf-8-sig") as f:
            return {str(r["id"]) for r in csv.DictReader(f) if (r.get("id") or "").strip()}
    except Exception:
        return set()

_write_lock = threading.Lock()
def append_row(out_path, cols, row):
    with _write_lock:
        header = not out_path.exists()
        pd.DataFrame([row], columns=cols).to_csv(
            out_path, mode="a", header=header, index=False, encoding="utf-8-sig")

def call_model(prompt, response_format=None, max_retries=5):
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "extra_body": {"reasoning": {"effort": REASONING_EFFORT}},
    }
    if response_format:
        kwargs["response_format"] = response_format
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "")
            return content, json.dumps(resp.model_dump(), ensure_ascii=False)
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + random.uniform(0, 0.5))
    raise last

def parse_mcq(content):
    try:
        return (json.loads(content) or {}).get("answer")
    except Exception:
        m = re.search(r"\b([A-E])\b", content)
        return m.group(1) if m else None

def run_knowledge(workers):
    src = read_src(KNOW_SRC)
    done = load_done(KNOW_OUT)
    todo = [r for r in src if str(r.get("id") or r.get("﻿id")) not in done]
    print(f"[knowledge] total={len(src)} done={len(done)} todo={len(todo)} -> {KNOW_OUT.name}")
    def work(r):
        rid = str(r.get("id") or r.get("﻿id"))
        prompt = create_prompt(r["question"], r["A"], r["B"], r["C"], r["D"], r["E"])
        content, raw = call_model(prompt, response_format=ANSWER_SCHEMA)
        ans = parse_mcq(content)
        gold = str(r["gold"]).strip()
        row = {c: r.get(c, r.get("﻿"+c, "")) for c in KNOW_COLS if c in r or ("﻿"+c) in r}
        row.update({
            "id": rid, "answer": ans,
            "answer_structured": json.dumps({"answer": ans}, ensure_ascii=False) if ans else None,
            "raw_response": raw,
            "is_correct": (ans is not None and str(ans).strip() == gold),
        })
        append_row(KNOW_OUT, KNOW_COLS, row)
        return rid, ans
    _run(todo, work, workers)

def run_reasoning(workers):
    src = read_src(REAS_SRC)
    done = load_done(REAS_OUT)
    todo = [r for r in src if str(r.get("id") or r.get("﻿id")) not in done]
    print(f"[reasoning] total={len(src)} done={len(done)} todo={len(todo)} -> {REAS_OUT.name}")
    def work(r):
        rid = str(r.get("id") or r.get("﻿id"))
        prompt = create_reasoning_prompt(r["context"], r["question"])
        content, raw = call_model(prompt)
        row = {
            "id": rid, "source": r.get("source",""), "category": r.get("category",""),
            "question": r["question"], "context": r["context"], "gold": r["gold"],
            "answer": content.strip(), "raw_response": raw,
        }
        append_row(REAS_OUT, REAS_COLS, row)
        return rid, len(content)
    _run(todo, work, workers)

def _run(todo, work, workers):
    ok = empties = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rid, val = fut.result()
                ok += 1
                if not val:
                    empties += 1
            except Exception as e:
                fail += 1
                print(f"  FAIL: {e}")
            if i % 20 == 0:
                print(f"  ...{i}/{len(todo)} ok={ok} empty={empties} fail={fail} ({time.time()-t0:.0f}s)")
    print(f"DONE ok={ok} empty={empties} fail={fail} elapsed={time.time()-t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["knowledge", "reasoning"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    print(f"model={MODEL} effort={REASONING_EFFORT} max_tokens={MAX_TOKENS} domain={a.domain} workers={a.workers}")
    (run_knowledge if a.domain == "knowledge" else run_reasoning)(a.workers)

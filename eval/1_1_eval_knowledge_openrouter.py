#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
지식(객관식) 응답 생성 스크립트 (OpenRouter)

- 데이터셋: `_datasets/0_integration/1_fin_knowledge.csv` (296문항)
- 두 가지 모드 (지식 도메인에만 존재하는 structured ↔ free 분기):
    [default] Structured Outputs (`response_format` json_schema strict) — A~E 강제,
              비추론 모델용 (구 1_1_eval_knowledge_{openai,claude}.py 의 동작)
    [--think] 자유 생성 (structured 미사용) — 추론 모델용. 모델 출력에서
              `</think>` 뒤 + 첫 A~E 글자 regex 로 답 추출.
              raw_response 그대로 저장 → 정밀 판정은
              `1_2_stats_eval_knowledge.py --llm-judge` 로 LLM judge 사용.
              (구 1_1_eval_plus_run_plain_eval.py + 1_2_eval_plus_judge_knowledge_response.py 의 동작)
- 행 단위 idempotent resume (append-only), per-row raw_response 보존
- 멀티스레드 + append lock
- 출력 컬럼: dataset 컬럼 + `answer`, `answer_structured`, `raw_response`
  (`is_correct` / `_response_stats.json` 은 `1_2_stats_eval_knowledge.py` 에서 추가)

사용 예:
    # 비추론 모델 (기본 structured)
    python eval/1_1_eval_knowledge_openrouter.py --model openai/gpt-5.2
    # 추론 모델 (think 모드)
    python eval/1_1_eval_knowledge_openrouter.py --model deepseek/deepseek-r1 --think
    # 출력 파일명 override
    python eval/1_1_eval_knowledge_openrouter.py --model anthropic/claude-opus-4.5:claude-opus-4-5
    # 여러 모델 (반복 지정 가능)
    python eval/1_1_eval_knowledge_openrouter.py \
        --model openai/gpt-5.2 \
        --model google/gemini-3.1-pro-preview --workers 8

환경 변수:
    OPENROUTER_API_KEY  (필수, .env 권장)

주의:
- `--max-tokens` 기본 8192. reasoning 모델은 reasoning 토큰이 포함되므로 작게
  잡으면 답이 잘립니다. `--think` 모드는 8192 이상 권장.
- `--reasoning-effort` 기본 "minimal". `""`/`none` 전달 시 reasoning 파라미터 생략.
- 생성 직후 `python eval/1_2_stats_eval_knowledge.py [--llm-judge]` 실행.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import APIError, OpenAI
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# =================================
# 경로 / 환경
# =================================
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_DATASET = REPO_ROOT / "_datasets" / "0_integration" / "1_fin_knowledge.csv"
RESULTS_DIR = EVAL_DIR / "_results" / "1_fin_knowledge"

load_dotenv(ENV_PATH)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(f"OPENROUTER_API_KEY not set. Add it to {ENV_PATH} or export it.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/sklim84/KFinEval",
        "X-Title": "KFinEval Knowledge Generation",
    },
)

# =================================
# Structured Output JSON Schema (A~E 한 글자 강제)
# =================================
ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mcq_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


# =================================
# 프롬프트 (knowledge 객관식 공통 문구)
# =================================
def create_prompt(question: str, a: str, b: str, c: str, d: str, e: str) -> str:
    return f"""다음 객관식 질문에 답하세요.

질문:
{question}

선택지:
A. {a}
B. {b}
C. {c}
D. {d}
E. {e}""".strip()


# =================================
# OpenRouter 호출
# =================================
def build_request_kwargs(
    backend_id: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str,
    think: bool = False,
) -> dict:
    kwargs: dict = {
        "model": backend_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    # 비추론 모델: structured 강제로 A~E 강제 / 추론 모델: 자유 생성
    if not think:
        kwargs["response_format"] = ANSWER_SCHEMA
    if reasoning_effort and reasoning_effort.lower() != "none":
        kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    return kwargs


def call_with_retry(kwargs: dict, max_retries: int = 5):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except APIError as e:
            last_exc = e
            wait = min(2 ** attempt, 30) + random.uniform(0, 0.5)
            tqdm.write(f"  APIError ({e.__class__.__name__}): {e}. retry in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            last_exc = e
            wait = min(2 ** attempt, 30) + random.uniform(0, 0.5)
            tqdm.write(f"  Exception ({e.__class__.__name__}): {e}. retry in {wait:.1f}s")
            time.sleep(wait)
    raise last_exc if last_exc else RuntimeError("call_with_retry: exhausted retries")


def parse_mcq_answer(content: str):
    """structured output JSON 파싱; 실패 시 regex 백업."""
    if not content:
        return None
    try:
        return (json.loads(content) or {}).get("answer")
    except Exception:
        m = re.search(r"\b([A-E])\b", content)
        return m.group(1) if m else None


def parse_mcq_answer_freeform(content: str):
    """추론 모델 자유 출력에서 A~E 추출.
    - `</think>` 가 있으면 그 뒤 텍스트만 사용 (구 plus_judge.extract_answer 동일)
    - 그 다음 텍스트의 첫 A~E 글자 (구 plus_run_plain_eval.parse_knowledge_answer 동일)
    - 단순 regex 가 모호한 경우(장문 reasoning) → stats 의 --llm-judge 사용 권장.
    """
    if not content:
        return None
    if "</think>" in content:
        content = content.split("</think>")[-1]
    s = content.strip().upper()
    if not s:
        return None
    if s[0] in "ABCDE":
        return s[0]
    for ch in "ABCDE":
        if ch in s:
            return ch
    return None


# =================================
# Resume / Append
# =================================
def load_done_ids(output_csv: Path, id_col: str) -> set:
    if not output_csv.exists():
        return set()
    try:
        with output_csv.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return {str(r.get(id_col, "")).strip() for r in reader if (r.get(id_col) or "").strip()}
    except Exception as e:
        print(f"  [warn] existing output unreadable; treating as empty: {e}")
        return set()


_write_lock = threading.Lock()


def append_row(output_csv: Path, columns: list, row: dict) -> None:
    with _write_lock:
        write_header = not output_csv.exists()
        pd.DataFrame([row], columns=columns).to_csv(
            output_csv,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )


# =================================
# 모델 1건 처리
# =================================
def process_model(
    backend_id: str,
    out_name: str,
    dataset_path: Path,
    max_tokens: int,
    reasoning_effort: str,
    workers: int,
    limit: int,
    think: bool = False,
) -> None:
    data = pd.read_csv(dataset_path, encoding="utf-8-sig")
    # 데이터셋 첫 컬럼이 'id' (BOM 제거된 표준 이름) 라고 가정
    id_col = data.columns[0]
    if id_col != "id":
        # BOM 인한 ﻿id 같은 케이스 — 안전하게 rename
        data = data.rename(columns={id_col: "id"})
        id_col = "id"

    output_columns = list(data.columns) + ["answer", "answer_structured", "raw_response"]
    output_csv = RESULTS_DIR / f"1_fin_knowledge_{out_name}_response.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {backend_id}  →  {output_csv.name} ===")
    print(f"  backend_id       : {backend_id}")
    print(f"  mode             : {'think (free, regex)' if think else 'structured (A~E forced)'}")
    print(f"  reasoning_effort : {reasoning_effort or '(none)'}")
    print(f"  max_tokens       : {max_tokens}")
    print(f"  workers          : {workers}")
    print(f"  dataset          : {dataset_path}  ({len(data)} rows)")

    done = load_done_ids(output_csv, id_col)
    pending = data[~data[id_col].astype(str).isin(done)]
    if limit:
        pending = pending.head(limit)
    print(f"  resume           : done={len(done)}  todo={len(pending)}")

    if pending.empty:
        print("  nothing to do.")
        return

    def work(row_dict: dict) -> tuple:
        rid = str(row_dict[id_col])
        prompt = create_prompt(
            row_dict["question"],
            row_dict["A"], row_dict["B"], row_dict["C"], row_dict["D"], row_dict["E"],
        )
        kwargs = build_request_kwargs(backend_id, prompt, max_tokens, reasoning_effort, think=think)
        resp = call_with_retry(kwargs)
        content = (resp.choices[0].message.content or "")
        if think:
            answer = parse_mcq_answer_freeform(content)
            answer_structured = None  # think 모드는 structured 미사용
        else:
            answer = parse_mcq_answer(content)
            answer_structured = (
                json.dumps({"answer": answer}, ensure_ascii=False) if answer else None
            )
        raw = json.dumps(resp.model_dump(), ensure_ascii=False)
        row_out = dict(row_dict)
        row_out["answer"] = answer
        row_out["answer_structured"] = answer_structured
        row_out["raw_response"] = raw
        append_row(output_csv, output_columns, row_out)
        return rid, answer

    n_ok = n_empty = n_fail = 0
    t0 = time.time()
    todo_records = pending.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, r): r for r in todo_records}
        pbar = tqdm(as_completed(futs), total=len(futs), desc=out_name)
        for fut in pbar:
            try:
                _, ans = fut.result()
                n_ok += 1
                if not ans:
                    n_empty += 1
            except Exception as e:
                n_fail += 1
                tqdm.write(f"  FAIL: {e}")
    print(
        f"  DONE  ok={n_ok}  empty={n_empty}  fail={n_fail}  "
        f"elapsed={time.time()-t0:.0f}s"
    )
    print(f"  output: {output_csv}")


# =================================
# 메인
# =================================
def parse_model_spec(spec: str) -> tuple:
    """'openai/gpt-5.2'  → ('openai/gpt-5.2', 'gpt-5.2')
       'anthropic/claude-opus-4.5:claude-opus-4-5' → ('anthropic/claude-opus-4.5', 'claude-opus-4-5')
    """
    if ":" in spec:
        bid, name = spec.split(":", 1)
        return bid.strip(), name.strip()
    bid = spec.strip()
    return bid, bid.split("/")[-1]


def main():
    p = argparse.ArgumentParser(
        description="Knowledge MCQ answer generation via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", action="append", required=True,
        help='OpenRouter backend id (예: "openai/gpt-5.2"). '
             '"BACKEND_ID:OUTPUT_NAME" 형식으로 출력 파일명 override 가능. '
             '여러 번 지정하면 순차 실행.',
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help=f"knowledge CSV 경로 (기본: {DEFAULT_DATASET})")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="max_tokens (reasoning 토큰 포함; 기본 8192)")
    p.add_argument("--reasoning-effort", default="minimal",
                   help='reasoning effort: "minimal"|"low"|"medium"|"high" 또는 ""/"none" (기본 minimal)')
    p.add_argument("--think", action="store_true",
                   help="추론 모델 모드: structured output 미사용, 자유 생성 후 "
                        "</think>+regex 로 A~E 추출. 정밀 판정은 stats 의 --llm-judge 사용.")
    p.add_argument("--limit", type=int, default=None,
                   help="앞 N개만 처리 (디버그)")
    args = p.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")

    print("=" * 60)
    print("OpenRouter Knowledge Generation Mode")
    print("=" * 60)
    print(f"dataset          : {args.dataset}")
    print(f"mode             : {'think (free, regex)' if args.think else 'structured (A~E forced)'}")
    print(f"reasoning_effort : {args.reasoning_effort or '(none)'}")
    print(f"max_tokens       : {args.max_tokens}")
    print(f"workers          : {args.workers}")

    specs = [parse_model_spec(s) for s in args.model]
    for backend_id, out_name in specs:
        process_model(
            backend_id=backend_id,
            out_name=out_name,
            dataset_path=args.dataset,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            workers=args.workers,
            limit=args.limit,
            think=args.think,
        )

    print("\n모든 모델 생성 완료. 다음 단계:")
    if args.think:
        print("  python eval/1_2_stats_eval_knowledge.py --llm-judge  # think 출력은 LLM judge 권장")
    else:
        print("  python eval/1_2_stats_eval_knowledge.py              # is_correct + _response_stats.json")


if __name__ == "__main__":
    main()

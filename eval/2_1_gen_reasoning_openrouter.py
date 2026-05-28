#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
추론(reasoning) 응답 생성 스크립트 (OpenRouter 사용)

- 데이터셋: `_datasets/0_integration/2_fin_reasoning.csv` (575문항)
- 자유 텍스트 생성 (structured output 없음; 모델이 단계별 추론 JSON 텍스트 생성)
- 행 단위 idempotent resume (append-only), per-row raw_response 보존
- 멀티스레드 + append lock
- 출력 컬럼: dataset 컬럼 + `answer`, `raw_response`
  (judge: `2_2_eval_reasoning_openrouter.py`, stats: `2_3_stats_eval_reasoning.py`)

사용 예:
    # 단일 모델
    python eval/2_1_gen_reasoning_openrouter.py --model openai/gpt-5.2
    # 출력 파일명 override
    python eval/2_1_gen_reasoning_openrouter.py --model anthropic/claude-opus-4.5:claude-opus-4-5
    # 여러 모델 + 병렬
    python eval/2_1_gen_reasoning_openrouter.py \
        --model openai/gpt-5.2 \
        --model google/gemini-3.1-pro-preview \
        --workers 8

환경 변수:
    OPENROUTER_API_KEY  (필수, .env 권장)

주의:
- `--max-tokens` 기본 8192. reasoning 토큰이 포함되므로 작게 잡으면 추론이 잘립니다.
- `--reasoning-effort` 기본 "minimal". `""`/`none` 전달 시 reasoning 파라미터 미전송.
- 생성 후 다음 단계:
    python eval/2_2_eval_reasoning_openrouter.py --target-model <model>
    python eval/2_3_stats_eval_reasoning.py
"""

import argparse
import csv
import json
import os
import random
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
DEFAULT_DATASET = REPO_ROOT / "_datasets" / "0_integration" / "2_fin_reasoning.csv"
RESULTS_DIR = EVAL_DIR / "_results" / "2_fin_reasoning"

load_dotenv(ENV_PATH)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(f"OPENROUTER_API_KEY not set. Add it to {ENV_PATH} or export it.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/sklim84/KFinEval",
        "X-Title": "KFinEval Reasoning Generation",
    },
)


# =================================
# 프롬프트 (reasoning 공통 문구)
# =================================
def create_reasoning_prompt(context: str, question: str) -> str:
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


# =================================
# OpenRouter 호출
# =================================
def build_request_kwargs(
    backend_id: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str,
) -> dict:
    kwargs: dict = {
        "model": backend_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
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
) -> None:
    data = pd.read_csv(dataset_path, encoding="utf-8-sig")
    id_col = data.columns[0]
    if id_col != "id":
        data = data.rename(columns={id_col: "id"})
        id_col = "id"

    output_columns = list(data.columns) + ["answer", "raw_response"]
    output_csv = RESULTS_DIR / f"2_fin_reasoning_{out_name}_answer.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {backend_id}  →  {output_csv.name} ===")
    print(f"  backend_id       : {backend_id}")
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
        prompt = create_reasoning_prompt(row_dict["context"], row_dict["question"])
        kwargs = build_request_kwargs(backend_id, prompt, max_tokens, reasoning_effort)
        resp = call_with_retry(kwargs)
        content = (resp.choices[0].message.content or "").strip()
        raw = json.dumps(resp.model_dump(), ensure_ascii=False)
        row_out = dict(row_dict)
        row_out["answer"] = content
        row_out["raw_response"] = raw
        append_row(output_csv, output_columns, row_out)
        return rid, len(content)

    n_ok = n_empty = n_fail = 0
    t0 = time.time()
    todo_records = pending.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, r): r for r in todo_records}
        pbar = tqdm(as_completed(futs), total=len(futs), desc=out_name)
        for fut in pbar:
            try:
                _, n_chars = fut.result()
                n_ok += 1
                if n_chars == 0:
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
        description="Reasoning answer generation via OpenRouter",
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
                   help=f"reasoning CSV 경로 (기본: {DEFAULT_DATASET})")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="max_tokens (reasoning 토큰 포함; 기본 8192)")
    p.add_argument("--reasoning-effort", default="minimal",
                   help='reasoning effort: "minimal"|"low"|"medium"|"high" 또는 ""/"none" (기본 minimal)')
    p.add_argument("--limit", type=int, default=None,
                   help="앞 N개만 처리 (디버그)")
    args = p.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")

    print("=" * 60)
    print("OpenRouter Reasoning Generation Mode")
    print("=" * 60)
    print(f"dataset          : {args.dataset}")
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
        )

    print("\n모든 모델 생성 완료. 다음 단계:")
    print("  python eval/2_2_eval_reasoning_openrouter.py --target-model <model>")
    print("  python eval/2_3_stats_eval_reasoning.py")


if __name__ == "__main__":
    main()

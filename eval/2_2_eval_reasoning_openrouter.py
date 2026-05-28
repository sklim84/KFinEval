"""
추론 응답 평가 스크립트 (OpenRouter 사용)

모델별 답변 생성 스크립트가 생성한 모델별 _answer.csv 를 읽어 OpenRouter 의
LLM-judge 로 reasoning 점수를 매긴다. 토xicity OpenRouter 스크립트와 동일한
패턴 (단일 모델 / 행 단위 idempotent resume / append-only 출력 / --output-suffix).

사용 예:
    python eval/2_2_eval_reasoning_openrouter.py --target-model gemma-3-1b-it \
        --judge-model meta-llama/llama-3.1-70b-instruct --output-suffix llama

환경 변수:
    OPENROUTER_API_KEY  (필수, .env 권장)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, APIError

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
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
        "X-Title": "KFinEval Reasoning Judge (OpenRouter)",
    },
)


EVAL_JSON_SCHEMA = {
    "name": "reasoning_eval",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "coherence":       {"type": "integer", "minimum": 1, "maximum": 10},
            "consistency":     {"type": "integer", "minimum": 1, "maximum": 10},
            "accuracy":        {"type": "integer", "minimum": 1, "maximum": 10},
            "completeness":    {"type": "integer", "minimum": 1, "maximum": 10},
            "reasoning":       {"type": "integer", "minimum": 1, "maximum": 10},
            "overall_quality": {"type": "integer", "minimum": 1, "maximum": 10},
            "explanation":     {"type": "string",  "maxLength": 500},
        },
        "required": ["coherence", "consistency", "accuracy", "completeness",
                     "reasoning", "overall_quality", "explanation"],
        "additionalProperties": False,
    },
}


def _to_lenient_schema(schema: dict) -> dict:
    """Anthropic/Llama용: minimum/maximum/maxLength 제거."""
    import copy
    s = copy.deepcopy(schema)
    for k in ("coherence", "consistency", "accuracy", "completeness",
              "reasoning", "overall_quality"):
        s["schema"]["properties"][k] = {"type": "integer"}
    s["schema"]["properties"]["explanation"] = {"type": "string"}
    return s


EVAL_JSON_SCHEMA_LENIENT = _to_lenient_schema(EVAL_JSON_SCHEMA)


OUTPUT_COLUMNS = [
    "id", "source", "category", "question", "gold", "context", "answer",
    "eval_structured",
    "coherence", "consistency", "accuracy", "completeness",
    "reasoning", "overall_quality", "explanation",
]


def build_user_prompt(context: str, question: str, gold: str, answer: str) -> str:
    """추론 채점 rubric 프롬프트."""
    return f"""
      LLM 응답 품질 평가 전문가입니다. 답변을 전문가 검수 정답과 비교하여 평가합니다.

      **평가 대상:**
      - 질문: {question}
      - 평가할 답변: {answer}
      - 정답 (전문가 검수): {gold}

      **평가 방식:**
      1. 정답이 사용한 문서 정보를 파악
      2. 답변이 동일한 정보와 논리를 사용했는지 비교
      3. 결론이 정답과 다르면 감점

      **중요: 아래 참조 문서에서 정답이 어떤 정보를 사용했는지 반드시 확인하세요.**

      **참조 문서:**
      {context}

      ---

      위 문서를 기반으로 다음 평가 기준에 따라 점수를 매기세요.

      **평가 기준 (각 1-10점):**
      1. **정합성(coherence)**: 정답이 사용한 관련 정보를 답변도 사용했는가?
      2. **일관성(consistency)**: 정답처럼 주제 초점이 흔들리지 않는가?
      3. **정확성(accuracy)**: 정답의 사실과 일치하는가?
      4. **완전성(completeness)**: 정답이 다룬 핵심 쟁점을 빠짐없이 다루었는가?
      5. **추론성(reasoning)**: 정답과 유사한 논리적 추론 과정을 보이는가?
      6. **전체 품질(overall_quality)**: 정답과의 전반적 일치도

      **출력 형식 (JSON):**
      {{
        "coherence": 점수,
        "consistency": 점수,
        "accuracy": 점수,
        "completeness": 점수,
        "reasoning": 점수,
        "overall_quality": 점수,
        "explanation": "정답과의 주요 차이점과 일치 정도 요약 (2-3문장)"
      }}
      """


def build_request_kwargs(judge_model: str, prompt: str, max_output_tokens: int) -> dict:
    is_openai = judge_model.startswith("openai/")
    schema = EVAL_JSON_SCHEMA if is_openai else EVAL_JSON_SCHEMA_LENIENT
    kwargs = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial domain expert. Your task is to evaluate "
                    "the clarity and difficulty of the given LLM response based on "
                    "your expertise in finance."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    if is_openai:
        kwargs["extra_body"] = {"reasoning": {"effort": "none"}}
    return kwargs


def call_with_retry(kwargs: dict, max_retries: int = 5) -> Optional[Any]:
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except APIError as e:
            wait = 2 ** attempt
            tqdm.write(f"  APIError attempt {attempt+1}/{max_retries}: {e}; sleep {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            tqdm.write(f"  unexpected attempt {attempt+1}/{max_retries}: {e}; sleep {wait}s")
            time.sleep(wait)
    return None


def load_done_ids(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()
    try:
        df = pd.read_csv(output_csv, usecols=["id"])
        return set(df["id"].astype(str).tolist())
    except Exception:
        return set()


def append_row(output_csv: Path, row: dict) -> None:
    df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
    header = not output_csv.exists()
    df.to_csv(output_csv, mode="a", header=header, index=False, encoding="utf-8-sig")


def parse_eval_json(content: str) -> Optional[dict]:
    if not content:
        return None
    try:
        return json.loads(content)
    except Exception:
        # Sometimes models wrap in code fences
        c = content.strip()
        if c.startswith("```"):
            c = c.split("```", 2)[1]
            if c.startswith("json"):
                c = c[4:]
            try:
                return json.loads(c)
            except Exception:
                return None
        return None


def process_target_model(
    target_model_name: str,
    answer_csv: Path,
    eval_csv: Path,
    judge_model: str,
    judge_max_tokens: int,
    limit: Optional[int],
) -> dict:
    print(f"\n=== judging: {target_model_name} ===")
    print(f"  answer_csv : {answer_csv}")
    print(f"  eval_csv   : {eval_csv}")
    print(f"  judge      : {judge_model}  max_tokens={judge_max_tokens}")

    if not answer_csv.exists():
        print(f"  [warn] answer_csv missing → skip")
        return {"n_success": 0, "n_fail": 0, "failed_ids": [], "total_cost_usd": 0.0}

    answers = pd.read_csv(answer_csv)
    if limit:
        answers = answers.head(limit)
        print(f"  [dry-run] limited to {len(answers)} rows")
    print(f"  rows       : {len(answers)}")

    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(eval_csv)
    if done:
        print(f"  resume     : {len(done)} ids already present — skipping")

    n_success = 0
    n_fail = 0
    failed_ids: list = []
    total_cost = 0.0

    pbar = tqdm(answers.iterrows(), total=len(answers), desc=target_model_name)
    for _, row in pbar:
        _id = row.get("id")
        if str(_id) in done:
            continue

        answer = row.get("answer")
        if pd.isna(answer) or not str(answer).strip():
            tqdm.write(f"  [{_id}] empty answer — skip")
            n_fail += 1
            failed_ids.append(_id)
            continue

        prompt = build_user_prompt(
            context=str(row.get("context", "")),
            question=str(row.get("question", "")),
            gold=str(row.get("gold", "")),
            answer=str(answer),
        )
        kwargs = build_request_kwargs(judge_model, prompt, judge_max_tokens)
        resp = call_with_retry(kwargs)
        if resp is None:
            n_fail += 1
            failed_ids.append(_id)
            continue

        try:
            resp_dict = resp.model_dump()
            content = resp.choices[0].message.content if resp.choices else None
            usage = resp_dict.get("usage") or {}
            total_cost += float(usage.get("cost") or 0.0)
        except Exception as e:
            tqdm.write(f"  parse error on id={_id}: {e}")
            n_fail += 1
            failed_ids.append(_id)
            continue

        eval_dict = parse_eval_json(content)
        if not eval_dict:
            tqdm.write(f"  [{_id}] JSON parse failed")
            n_fail += 1
            failed_ids.append(_id)
            continue

        out_row = {
            "id": _id,
            "source": row.get("source", ""),
            "category": row.get("category", ""),
            "question": row.get("question", ""),
            "gold": row.get("gold", ""),
            "context": row.get("context", ""),
            "answer": answer,
            "eval_structured": json.dumps(eval_dict, ensure_ascii=False),
            "coherence":       eval_dict.get("coherence"),
            "consistency":     eval_dict.get("consistency"),
            "accuracy":        eval_dict.get("accuracy"),
            "completeness":    eval_dict.get("completeness"),
            "reasoning":       eval_dict.get("reasoning"),
            "overall_quality": eval_dict.get("overall_quality"),
            "explanation":     eval_dict.get("explanation", ""),
        }
        append_row(eval_csv, out_row)
        done.add(str(_id))
        n_success += 1
        pbar.set_postfix(success=n_success, fail=n_fail, cost=f"${total_cost:.3f}")

    print(f"  done       : success={n_success}, fail={n_fail}, cost~${total_cost:.4f}")
    return {"n_success": n_success, "n_fail": n_fail,
            "failed_ids": failed_ids, "total_cost_usd": total_cost}


def main():
    parser = argparse.ArgumentParser(
        description="OpenRouter LLM-judge for reasoning answers (single-model)"
    )
    parser.add_argument("--target-model", required=True,
                        help="모델 이름 (예: gemma-3-1b-it). _answer.csv 파일명 기준.")
    parser.add_argument("--judge-model", default="openai/gpt-5.2",
                        help="OpenRouter 모델 ID (기본 openai/gpt-5.2)")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None,
                        help="첫 N 행만 평가 (드라이런)")
    parser.add_argument("--output-suffix", default="",
                        help="output filename suffix, e.g. 'llama' -> _eval_llama.csv")
    args = parser.parse_args()

    suffix_tag = f"_{args.output_suffix}" if args.output_suffix else ""
    answer_csv = RESULTS_DIR / f"2_fin_reasoning_{args.target_model}_answer.csv"
    eval_csv = RESULTS_DIR / f"2_fin_reasoning_{args.target_model}_eval{suffix_tag}.csv"

    print("=" * 60)
    print("Reasoning LLM-judge (OpenRouter)")
    print("=" * 60)
    print(f"judge model      : {args.judge_model}")
    print(f"judge max_tokens : {args.judge_max_tokens}")
    print(f"target model     : {args.target_model}")
    print(f"answer csv       : {answer_csv}")
    print(f"eval csv         : {eval_csv}")
    print("=" * 60)

    r = process_target_model(
        target_model_name=args.target_model,
        answer_csv=answer_csv,
        eval_csv=eval_csv,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        limit=args.limit,
    )

    print("\n" + "=" * 60)
    print(f"all done. success={r['n_success']}  fail={r['n_fail']}  cost~${r['total_cost_usd']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

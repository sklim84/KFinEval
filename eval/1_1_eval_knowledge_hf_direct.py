#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
지식(객관식) 응답 생성 — HF transformers direct inference

`1_1_eval_knowledge_vllm.py` 가 처리하지 못하는 모델용
(예: vLLM 0.20.1/0.21.0 이 architecture 를 등록하지 않는 `SolarOpenForCausalLM`,
`VaetkiForCausalLM` 등).

- 데이터셋: `_datasets/0_integration/1_fin_knowledge.csv` (296문항)
- HF transformers 로 직접 generate. structured output 미지원이므로
  **항상 free generation + `</think>` + regex 로 A~E 추출** (--think 모드와 동일).
- 행 단위 idempotent resume (append-only), per-row raw 보존
- 출력 컬럼: dataset 컬럼 + `answer`, `answer_structured`, `raw_response`
  (`is_correct` / `_response_stats.json` 은 `1_2_stats_eval_knowledge.py [--llm-judge]` 가 추가)

사용 예 (보통 `run_hf_direct_env.sh` 로 wrap):
    ./run_hf_direct_env.sh eval/1_1_eval_knowledge_hf_direct.py \
        --hf-model upstage/Solar-Open-100B --name Solar_Open_100B --max-new-tokens 8192
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "_datasets" / "0_integration" / "1_fin_knowledge.csv"
RESULTS_DIR = EVAL_DIR / "_results" / "1_fin_knowledge"


def _ensure_hf_token():
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("✓ HuggingFace 토큰 확인됨")
        return
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                key, val = line.split("=", 1)
                os.environ[key] = val.strip().strip('"').strip("'")
                print(f"✓ HuggingFace 토큰 ({key}) 로드됨 from .env")
                return
    print("⚠ HF_TOKEN not set — gated repos may fail")


# =================================
# 프롬프트 — 1_1_eval_knowledge_{vllm,openrouter}.py 와 동일 문구
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


def parse_mcq_answer_freeform(content: str):
    """`</think>` 뒤 텍스트의 첫 A~E 글자."""
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
# HF 모델 로딩 (toxicity hf_direct 와 동일 패턴)
# =================================
def load_model_hf(hf_model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n모델 로딩 중: {hf_model}")
    print(f"가용 GPU 수: {torch.cuda.device_count()}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if hasattr(model, "hf_device_map"):
        gpus = sorted({str(v) for v in model.hf_device_map.values() if str(v).startswith("cuda")})
        print(f"  device_map shards across: {gpus}")
    print(f"✓ 모델 로딩 완료: {hf_model}")
    return model, tokenizer


# =================================
# 생성
# =================================
def generate_answer_hf(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    think_mode: bool = False,
) -> tuple[str, dict]:
    try:
        apply_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if think_mode:
            apply_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **apply_kwargs,
        )
        chat_template_applied = True
    except Exception:
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            chat_template_applied = True
        except Exception:
            templated = prompt
            chat_template_applied = False

    inputs = tokenizer(templated, return_tensors="pt").to(model.device)
    n_prompt = int(inputs.input_ids.shape[-1])
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][n_prompt:]
    n_gen = int(gen_ids.shape[-1])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    finish_reason = "length" if n_gen >= max_new_tokens else "stop"
    raw = {
        "backend": "hf_transformers",
        "chat_template_applied": chat_template_applied,
        "think_mode": think_mode,
        "max_new_tokens": max_new_tokens,
        "finish_reason": finish_reason,
        "n_generated_tokens": n_gen,
        "n_prompt_tokens": n_prompt,
        "text": text,
    }
    return text, raw


# =================================
# Resume / Append
# =================================
def _load_done_ids(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()
    try:
        df = pd.read_csv(output_csv, usecols=["id"])
        return set(df["id"].astype(str).tolist())
    except Exception as e:
        print(f"  [warn] existing output unreadable; treating as empty: {e}")
        return set()


def _append_row(output_csv: Path, columns: list, row: dict) -> None:
    write_header = not output_csv.exists()
    pd.DataFrame([row], columns=columns).to_csv(
        output_csv, mode="a", index=False, header=write_header, encoding="utf-8-sig"
    )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =================================
# 메인 처리
# =================================
def process_dataset(
    model, tokenizer, hf_model: str, out_name: str,
    dataset: pd.DataFrame, output_csv: Path,
    max_new_tokens: int, think_mode: bool,
):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    id_col = dataset.columns[0]
    if id_col != "id":
        dataset = dataset.rename(columns={id_col: "id"})
        id_col = "id"
    output_columns = list(dataset.columns) + ["answer", "answer_structured", "raw_response"]
    done = _load_done_ids(output_csv)
    pending = dataset[~dataset[id_col].astype(str).isin(done)]
    if pending.empty:
        print(f"  nothing to do (done={len(done)}).")
        return
    print(f"  HF generate: {len(pending)} prompts "
          f"(max_new_tokens={max_new_tokens}, think_mode={think_mode})")
    t0 = _iso_now()
    n_ok = n_empty = 0
    for _, row in tqdm(pending.iterrows(), total=len(pending), desc=out_name):
        prompt = create_prompt(
            row["question"], row["A"], row["B"], row["C"], row["D"], row["E"],
        )
        text, raw = generate_answer_hf(model, tokenizer, prompt, max_new_tokens, think_mode)
        answer = parse_mcq_answer_freeform(text)
        if answer is None:
            n_empty += 1
        else:
            n_ok += 1
        row_out = row.to_dict()
        row_out["answer"] = answer
        row_out["answer_structured"] = None  # free generation 이므로 structured 없음
        row_out["raw_response"] = json.dumps(raw, ensure_ascii=False)
        _append_row(output_csv, output_columns, row_out)
    print(f"  DONE  ok={n_ok}  empty={n_empty}  started={t0}  finished={_iso_now()}")
    print(f"  output: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge MCQ via HF transformers direct (for vLLM-unsupported archs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hf-model", required=True,
                        help="HuggingFace model id (예: upstage/Solar-Open-100B)")
    parser.add_argument("--name", default=None,
                        help="출력 파일명용 short name (기본: hf_model 의 마지막 segment)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"knowledge CSV 경로 (기본: {DEFAULT_DATASET})")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="최대 생성 토큰 (기본 8192; 추론 모델은 더 크게)")
    parser.add_argument("--think", action="store_true",
                        help="chat template enable_thinking=True 사용 (Qwen3-Thinking 등)")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞 N개만 (디버그)")
    args = parser.parse_args()

    _ensure_hf_token()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")

    out_name = args.name or args.hf_model.split("/")[-1]
    output_csv = RESULTS_DIR / f"1_fin_knowledge_{out_name}_response.csv"

    print("=" * 60)
    print("HF transformers direct Knowledge Generation")
    print("=" * 60)
    print(f"hf_model         : {args.hf_model}")
    print(f"output           : {output_csv}")
    print(f"max_new_tokens   : {args.max_new_tokens}")
    print(f"think_mode       : {args.think}")
    print(f"dataset          : {args.dataset}")

    dataset = pd.read_csv(args.dataset, encoding="utf-8-sig")
    if args.limit:
        dataset = dataset.head(args.limit)

    model, tokenizer = load_model_hf(args.hf_model)
    process_dataset(
        model, tokenizer, args.hf_model, out_name,
        dataset, output_csv,
        max_new_tokens=args.max_new_tokens, think_mode=args.think,
    )

    print("\n다음 단계:")
    print("  python eval/1_2_stats_eval_knowledge.py --llm-judge")


if __name__ == "__main__":
    main()

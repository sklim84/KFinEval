#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
지식(객관식) 응답 생성 — VAETKI vLLM plugin

VAETKI 공식 vLLM plugin (vllm==0.11.2 pinned) 환경에서 실행.
반드시 `run_vaetki_env.sh` 로 wrap 해야 함 (PYTHONPATH unset + 전용 venv activate).

- 데이터셋: `_datasets/0_integration/1_fin_knowledge.csv` (296문항)
- vLLM batched generate (single batch). structured output 미사용 → free generation
  + `</think>` + regex 로 A~E 추출.
- 행 단위 idempotent resume, raw 보존
- 출력 컬럼: dataset 컬럼 + `answer`, `answer_structured`, `raw_response`
  (`is_correct` / stats 는 `1_2_stats_eval_knowledge.py [--llm-judge]` 가 추가)

사용 예:
    ./run_vaetki_env.sh eval/1_1_eval_knowledge_vaetki.py \
        --hf-model NC-AI-consortium-VAETKI/VAETKI --name VAETKI --max-tokens 8192
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "_datasets" / "0_integration" / "1_fin_knowledge.csv"
RESULTS_DIR = EVAL_DIR / "_results" / "1_fin_knowledge"


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


def process_dataset(llm, tokenizer, hf_model: str, out_name: str,
                    dataset: pd.DataFrame, output_csv: Path, max_tokens: int):
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, top_k=20,
        max_tokens=max_tokens, skip_special_tokens=False,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    id_col = dataset.columns[0]
    if id_col != "id":
        dataset = dataset.rename(columns={id_col: "id"})
        id_col = "id"
    output_columns = list(dataset.columns) + ["answer", "answer_structured", "raw_response"]

    done = _load_done_ids(output_csv)
    if done:
        print(f"  resume: {len(done)} ids already present — skipping")

    pending_meta = []
    pending_prompts = []
    for _, row in dataset.iterrows():
        if str(row[id_col]) in done:
            continue
        prompt = create_prompt(row["question"], row["A"], row["B"], row["C"], row["D"], row["E"])
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        pending_meta.append(row.to_dict())
        pending_prompts.append(templated)

    n = len(pending_prompts)
    print(f"  vLLM batched generate: {n} prompts (max_tokens={max_tokens}, plugin=vaetki)")
    if n == 0:
        return
    outputs = llm.generate(pending_prompts, sampling_params)

    n_ok = n_empty = 0
    for meta, req_out in tqdm(zip(pending_meta, outputs), total=n, desc=out_name):
        out = req_out.outputs[0]
        text = (out.text or "").strip()
        answer = parse_mcq_answer_freeform(text)
        if answer is None:
            n_empty += 1
        else:
            n_ok += 1
        raw = {
            "backend": "vllm_vaetki_plugin",
            "hf_model": hf_model,
            "max_tokens": max_tokens,
            "chat_template_applied": True,
            "think_mode": True,
            "finish_reason": getattr(out, "finish_reason", None),
            "stop_reason": getattr(out, "stop_reason", None),
            "n_generated_tokens": len(getattr(out, "token_ids", []) or []),
            "n_prompt_tokens": len(req_out.prompt_token_ids or []),
            "text": text,
        }
        row_out = dict(meta)
        row_out["answer"] = answer
        row_out["answer_structured"] = None
        row_out["raw_response"] = json.dumps(raw, ensure_ascii=False)
        _append_row(output_csv, output_columns, row_out)
    print(f"  DONE  ok={n_ok}  empty={n_empty}")
    print(f"  output: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge MCQ via VAETKI vLLM plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hf-model", required=True,
                        help="HuggingFace id (예: NC-AI-consortium-VAETKI/VAETKI)")
    parser.add_argument("--name", default=None,
                        help="출력 파일명 short name (기본: hf_model 의 마지막 segment)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")
    out_name = args.name or args.hf_model.split("/")[-1]
    output_csv = RESULTS_DIR / f"1_fin_knowledge_{out_name}_response.csv"

    print("=" * 60)
    print("VAETKI vLLM plugin Knowledge Generation")
    print("=" * 60)
    print(f"hf_model         : {args.hf_model}")
    print(f"output           : {output_csv}")
    print(f"max_tokens       : {args.max_tokens}")
    print(f"tp               : {args.tensor_parallel_size}")

    dataset = pd.read_csv(args.dataset, encoding="utf-8-sig")
    if args.limit:
        dataset = dataset.head(args.limit)

    print("\n[1/3] vLLM 모델 로딩...")
    llm = LLM(
        model=args.hf_model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    print(f"✓ loaded: {args.hf_model}")

    print("\n[2/3] 응답 생성...")
    try:
        process_dataset(llm, tokenizer, args.hf_model, out_name,
                        dataset, output_csv, args.max_tokens)
    finally:
        del llm
        torch.cuda.empty_cache()

    print("\n[3/3] 다음 단계:")
    print("  python eval/1_2_stats_eval_knowledge.py --llm-judge")


if __name__ == "__main__":
    main()

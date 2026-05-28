#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
추론(reasoning) 응답 생성 — VAETKI vLLM plugin

VAETKI 공식 vLLM plugin (vllm==0.11.2 pinned) 환경에서 실행.
반드시 `run_vaetki_env.sh` 로 wrap 해야 함 (PYTHONPATH unset + 전용 venv activate).

- 데이터셋: `_datasets/0_integration/2_fin_reasoning.csv` (575문항)
- vLLM batched generate (single batch). 자유 텍스트 생성, 그대로 저장
- 행 단위 idempotent resume, raw 보존
- 출력 컬럼: dataset 컬럼 + `answer`, `raw_response`
  (judge: `2_2_eval_reasoning_openrouter.py`, stats: `2_3_stats_eval_reasoning.py`)

사용 예:
    ./run_vaetki_env.sh eval/2_1_gen_reasoning_vaetki.py \
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
DEFAULT_DATASET = REPO_ROOT / "_datasets" / "0_integration" / "2_fin_reasoning.csv"
RESULTS_DIR = EVAL_DIR / "_results" / "2_fin_reasoning"


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
    output_columns = list(dataset.columns) + ["answer", "raw_response"]

    done = _load_done_ids(output_csv)
    if done:
        print(f"  resume: {len(done)} ids already present — skipping")

    pending_meta = []
    pending_prompts = []
    for _, row in dataset.iterrows():
        if str(row[id_col]) in done:
            continue
        prompt = create_reasoning_prompt(row["context"], row["question"])
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
        if not text:
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
        row_out["answer"] = text
        row_out["raw_response"] = json.dumps(raw, ensure_ascii=False)
        _append_row(output_csv, output_columns, row_out)
    print(f"  DONE  ok={n_ok}  empty={n_empty}")
    print(f"  output: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning gen via VAETKI vLLM plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")
    out_name = args.name or args.hf_model.split("/")[-1]
    output_csv = RESULTS_DIR / f"2_fin_reasoning_{out_name}_answer.csv"

    print("=" * 60)
    print("VAETKI vLLM plugin Reasoning Generation")
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
    print(f"  python eval/2_2_eval_reasoning_openrouter.py --target-model {out_name}")
    print("  python eval/2_3_stats_eval_reasoning.py")


if __name__ == "__main__":
    main()

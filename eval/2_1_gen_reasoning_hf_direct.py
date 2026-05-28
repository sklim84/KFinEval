#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
추론(reasoning) 응답 생성 — HF transformers direct inference

`2_1_gen_reasoning_vllm.py` 가 처리하지 못하는 모델용 (예: Solar-Open-100B).

- 데이터셋: `_datasets/0_integration/2_fin_reasoning.csv` (575문항)
- HF transformers 자유 생성. 출력은 원문 그대로 저장
- 행 단위 idempotent resume, raw 보존
- 출력 컬럼: dataset 컬럼 + `answer`, `raw_response`
  (judge: `2_2_eval_reasoning_openrouter.py`, stats: `2_3_stats_eval_reasoning.py`)

사용 예:
    ./run_hf_direct_env.sh eval/2_1_gen_reasoning_hf_direct.py \
        --hf-model upstage/Solar-Open-100B --name Solar_Open_100B --max-new-tokens 8192
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

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "_datasets" / "0_integration" / "2_fin_reasoning.csv"
RESULTS_DIR = EVAL_DIR / "_results" / "2_fin_reasoning"


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


def generate_answer_hf(model, tokenizer, prompt: str, max_new_tokens: int,
                       think_mode: bool = False) -> tuple[str, dict]:
    try:
        apply_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if think_mode:
            apply_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], **apply_kwargs,
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


def process_dataset(model, tokenizer, hf_model: str, out_name: str,
                    dataset: pd.DataFrame, output_csv: Path,
                    max_new_tokens: int, think_mode: bool):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    id_col = dataset.columns[0]
    if id_col != "id":
        dataset = dataset.rename(columns={id_col: "id"})
        id_col = "id"
    output_columns = list(dataset.columns) + ["answer", "raw_response"]
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
        prompt = create_reasoning_prompt(row["context"], row["question"])
        text, raw = generate_answer_hf(model, tokenizer, prompt, max_new_tokens, think_mode)
        if not text:
            n_empty += 1
        else:
            n_ok += 1
        row_out = row.to_dict()
        row_out["answer"] = text
        row_out["raw_response"] = json.dumps(raw, ensure_ascii=False)
        _append_row(output_csv, output_columns, row_out)
    print(f"  DONE  ok={n_ok}  empty={n_empty}  started={t0}  finished={_iso_now()}")
    print(f"  output: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning gen via HF transformers direct (for vLLM-unsupported archs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    _ensure_hf_token()

    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")
    out_name = args.name or args.hf_model.split("/")[-1]
    output_csv = RESULTS_DIR / f"2_fin_reasoning_{out_name}_answer.csv"

    print("=" * 60)
    print("HF transformers direct Reasoning Generation")
    print("=" * 60)
    print(f"hf_model         : {args.hf_model}")
    print(f"output           : {output_csv}")
    print(f"max_new_tokens   : {args.max_new_tokens}")
    print(f"think_mode       : {args.think}")

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
    print(f"  python eval/2_2_eval_reasoning_openrouter.py --target-model {out_name}")
    print("  python eval/2_3_stats_eval_reasoning.py")


if __name__ == "__main__":
    main()

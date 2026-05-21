#!/usr/bin/env bash
# HF transformers direct inference for toxicity benchmark (Solar-Open, VAETKI).
# Mirrors run_vllm_env.sh PYTHONPATH layout so we get transformers 5.5.1 from
# _eval_pylib (needed for SolarOpenForCausalLM).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHIM_DIR="$REPO_ROOT/eval/_vllm_shim"
EVAL_PYLIB="$REPO_ROOT/eval/_eval_pylib"

export PYTHONPATH="$SHIM_DIR:$EVAL_PYLIB:/home/work/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/home/work/kftc_model/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

exec python "$REPO_ROOT/eval/3_1_gen_toxicity_hf_direct.py" "$@"

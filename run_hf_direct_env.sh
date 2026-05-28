#!/usr/bin/env bash
# HF transformers direct inference for benchmark gen scripts (e.g. Solar-Open).
# Mirrors run_vllm_env.sh PYTHONPATH layout so we get transformers 5.5.1 from
# _eval_pylib (needed for SolarOpenForCausalLM).
#
# Usage (benchmark-agnostic):
#   ./run_hf_direct_env.sh eval/1_1_eval_knowledge_hf_direct.py --hf-model <id>
#   ./run_hf_direct_env.sh eval/2_1_gen_reasoning_hf_direct.py  --hf-model <id>
#   ./run_hf_direct_env.sh eval/3_1_gen_toxicity_hf_direct.py   --model <name>
#
# Backwards compat: if the first arg is not a `.py` path, falls back to
#   eval/3_1_gen_toxicity_hf_direct.py (the original target).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SHIM_DIR="$REPO_ROOT/eval/_vllm_shim"
EVAL_PYLIB="$REPO_ROOT/eval/_eval_pylib"

export PYTHONPATH="$SHIM_DIR:$EVAL_PYLIB:/home/work/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/home/work/kftc_model/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

if [[ "${1:-}" == *.py ]]; then
  SCRIPT_REL="$1"; shift
else
  SCRIPT_REL="eval/3_1_gen_toxicity_hf_direct.py"
fi

exec python "$REPO_ROOT/$SCRIPT_REL" "$@"

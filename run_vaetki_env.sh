#!/usr/bin/env bash
# VAETKI vLLM plugin via separate venv.
#
# Why this wrapper:
#  - VAETKI's vLLM plugin (NC-AI-consortium-VAETKI) is pinned to vllm==0.11.2,
#    torch==2.9.0, transformers~=4.57.3, etc. Incompatible with the
#    user-local vllm 0.20.1 / torch 2.11.0 used by other eval scripts.
#  - A dedicated venv at /home/work/kftc_model/vaetki_venv holds the pinned
#    stack. PYTHONPATH must be unset, otherwise the user-local site-packages
#    leaks system vllm 0.20.1 (no VaetkiForCausalLM, would silently fall
#    back and crash).
#
# Usage (benchmark-agnostic):
#   ./run_vaetki_env.sh eval/1_1_eval_knowledge_vaetki.py --hf-model <id>
#   ./run_vaetki_env.sh eval/2_1_gen_reasoning_vaetki.py  --hf-model <id>
#   ./run_vaetki_env.sh eval/3_1_gen_toxicity_vaetki.py   --model <name>
#
# Backwards compat: if the first arg is not a `.py` path, falls back to
#   eval/3_1_gen_toxicity_vaetki.py (the original target).
set -euo pipefail

# Critical: kill PYTHONPATH so venv site-packages take precedence.
unset PYTHONPATH

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VAETKI_VENV=/home/work/kftc_model/vaetki_venv

source "$VAETKI_VENV/bin/activate"

# Shared HF cache (so we don't redownload the 209GB weights).
export HF_HOME="${HF_HOME:-/home/work/kftc_model/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# HF_TOKEN from .env (gated repos)
if [ -f "$REPO_ROOT/.env" ]; then
  HF_TOKEN_LINE=$(grep -E "^HF_TOKEN=|^HUGGING_FACE_HUB_TOKEN=" "$REPO_ROOT/.env" | head -1 || true)
  if [ -n "$HF_TOKEN_LINE" ]; then
    export $HF_TOKEN_LINE
  fi
fi

if [[ "${1:-}" == *.py ]]; then
  SCRIPT_REL="$1"; shift
else
  SCRIPT_REL="eval/3_1_gen_toxicity_vaetki.py"
fi

exec python "$REPO_ROOT/$SCRIPT_REL" "$@"

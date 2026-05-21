#!/usr/bin/env bash
# Run vLLM-based toxicity generation in an environment compatible with the
# concurrently-running training job's user-local Python install.
#
# Why this wrapper exists:
#   - System Python has flash_attn built against a different libtorch ABI than
#     our torch 2.11.0+cu130 install, so vLLM's flash_attn import raises an
#     ImportError that crashes the EngineCore.  The sitecustomize shim raises
#     ModuleNotFoundError instead, letting vLLM fall back to TRITON_ATTN.
#   - We reuse user-local site-packages (where the training job's torch + vllm
#     live, both v2.11.0+cu130 / 0.20.1) and the shared HF / vLLM cache under
#     /home/work/kftc_model/.cache so models are not re-downloaded.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHIM_DIR="$REPO_ROOT/eval/_vllm_shim"
EVAL_PYLIB="$REPO_ROOT/eval/_eval_pylib"

# EVAL_PYLIB carries newer transformers (5.5.1) + huggingface_hub (1.15.0) so
# eval can load architectures (e.g. ExaoneMoE) that system transformers 4.57.1
# does not recognize. Order: shim → eval-only deps → user-local torch/vllm.
export PYTHONPATH="$SHIM_DIR:$EVAL_PYLIB:/home/work/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# Force vLLM to a backend that does not require flash_attn.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"

# Shared caches with the training job.
export HF_HOME="${HF_HOME:-/home/work/kftc_model/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/kftc_model/.cache/huggingface/datasets}"
export HF_XET_CACHE_DIR="${HF_XET_CACHE_DIR:-/home/work/kftc_model/.cache/huggingface/xet}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/work/kftc_model/.cache/huggingface/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/home/work/kftc_model/.cache/vllm}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

exec python "$REPO_ROOT/eval/3_1_gen_toxicity_openlm.py" "$@"

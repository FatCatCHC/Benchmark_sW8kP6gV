#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
UNIVERSALRAG_EMBEDDING_ROOT="${UNIVERSALRAG_EMBEDDING_ROOT:?set UNIVERSALRAG_EMBEDDING_ROOT to the per-cluster embedding folder root}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"

python ./main.py \
    --dataset_parent_dir "${DATASET_PARENT_DIR}" \
    --save_root_path "${SAVE_ROOT_PATH}" \
    --eval_strategy time_sequential_RAG \
    --eval_format multiple_choice \
    --eval_rounds "${EVAL_ROUNDS:-1}" \
    --eval_agent vrag \
    --model "${MODEL}" \
    --vrag_embedding_root "${UNIVERSALRAG_EMBEDDING_ROOT}" \
    --vrag_top_k "${VRAG_TOP_K:-3}" \
    --vrag_max_steps "${VRAG_MAX_STEPS:-3}" \
    --vrag_gpu "${VRAG_GPU:-0}"

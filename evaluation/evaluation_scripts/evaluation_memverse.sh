#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"

EXTRA_ARGS=()
if [[ "${MEMVERSE_USE_PM:-0}" =~ ^(1|true|yes)$ ]]; then
    EXTRA_ARGS+=(--memverse_use_pm)
fi

python ./main.py \
    --dataset_parent_dir "${DATASET_PARENT_DIR}" \
    --save_root_path "${SAVE_ROOT_PATH}" \
    --eval_strategy "${EVAL_STRATEGY:-time_sequential_memory}" \
    --eval_format multiple_choice \
    --eval_rounds "${EVAL_ROUNDS:-1}" \
    --eval_agent memverse \
    --model "${MODEL}" \
    --memverse_api_url "${MEMVERSE_API_URL:-http://127.0.0.1:8000}" \
    --memverse_insert_timeout_s "${MEMVERSE_INSERT_TIMEOUT_S:-1800}" \
    --memverse_query_timeout_s "${MEMVERSE_QUERY_TIMEOUT_S:-180}" \
    --memverse_chunk_turns "${MEMVERSE_CHUNK_TURNS:-6}" \
    --memverse_chunk_max_chars "${MEMVERSE_CHUNK_MAX_CHARS:-12000}" \
    --memverse_query_mode "${MEMVERSE_QUERY_MODE:-hybrid}" \
    "${EXTRA_ARGS[@]}"

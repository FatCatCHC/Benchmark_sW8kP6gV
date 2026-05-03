#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"

EXTRA_ARGS=()
if [[ "${MIRIX_NO_EMBEDDINGS:-0}" =~ ^(1|true|yes)$ ]]; then
    EXTRA_ARGS+=(--mirix_no_embeddings)
fi
if [[ "${MIRIX_DEBUG:-0}" =~ ^(1|true|yes)$ ]]; then
    EXTRA_ARGS+=(--mirix_debug)
fi

python ./main.py \
    --dataset_parent_dir "${DATASET_PARENT_DIR}" \
    --save_root_path "${SAVE_ROOT_PATH}" \
    --eval_strategy "${EVAL_STRATEGY:-time_sequential_memory}" \
    --eval_format multiple_choice \
    --eval_rounds "${EVAL_ROUNDS:-1}" \
    --eval_agent mirix \
    --model "${MODEL}" \
    --mirix_api_url "${MIRIX_API_URL:-http://localhost:8531}" \
    --mirix_org_id "${MIRIX_ORG_ID:-}" \
    --mirix_embedding_model "${MIRIX_EMBEDDING_MODEL:-text-embedding-v4}" \
    --mirix_embedding_dim "${MIRIX_EMBEDDING_DIM:-1024}" \
    --mirix_chunk_turns "${MIRIX_CHUNK_TURNS:-6}" \
    --mirix_chunk_max_chars "${MIRIX_CHUNK_MAX_CHARS:-12000}" \
    --mirix_chunk_max_images "${MIRIX_CHUNK_MAX_IMAGES:-8}" \
    --mirix_retrieve_limit "${MIRIX_RETRIEVE_LIMIT:-8}" \
    --mirix_ingest_media_mode "${MIRIX_INGEST_MEDIA_MODE:-original}" \
    "${EXTRA_ARGS[@]}"

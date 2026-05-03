#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"

python ./main.py \
    --dataset_parent_dir "${DATASET_PARENT_DIR}" \
    --save_root_path "${SAVE_ROOT_PATH}" \
    --eval_strategy "${EVAL_STRATEGY:-time_sequential_memory}" \
    --eval_format multiple_choice \
    --eval_rounds "${EVAL_ROUNDS:-1}" \
    --eval_agent memgpt \
    --model "${MODEL}" \
    --memgpt_server_url "${MEMGPT_SERVER_URL:-http://localhost:8283}" \
    --memgpt_server_password "${MEMGPT_SERVER_PASSWORD:-${LETTA_SERVER_PASSWORD:-}}" \
    --memgpt_chunk_turns "${MEMGPT_CHUNK_TURNS:-6}" \
    --memgpt_chunk_max_chars "${MEMGPT_CHUNK_MAX_CHARS:-12000}" \
    --memgpt_chunk_max_images "${MEMGPT_CHUNK_MAX_IMAGES:-8}" \
    --memgpt_context_window "${MEMGPT_CONTEXT_WINDOW:-16000}" \
    --memgpt_embedding_model "${MEMGPT_EMBEDDING_MODEL:-text-embedding-3-small}" \
    --memgpt_embedding_dim "${MEMGPT_EMBEDDING_DIM:-1536}" \
    --memgpt_ingest_mode "${MEMGPT_INGEST_MODE:-archival}" \
    --memgpt_archival_passage_max_chars "${MEMGPT_ARCHIVAL_PASSAGE_MAX_CHARS:-12000}" \
    --memgpt_archival_top_k "${MEMGPT_ARCHIVAL_TOP_K:-12}" \
    --memgpt_archival_tag "${MEMGPT_ARCHIVAL_TAG:-memgpt_eval}" \
    --memgpt_archival_qa_max_images "${MEMGPT_ARCHIVAL_QA_MAX_IMAGES:-8}" \
    --memgpt_ingest_media_mode "${MEMGPT_INGEST_MEDIA_MODE:-original}"

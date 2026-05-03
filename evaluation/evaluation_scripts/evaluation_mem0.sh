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
    --eval_agent mem0 \
    --model "${MODEL}" \
    --mem0_embedding_model "${MEM0_EMBEDDING_MODEL:-text-embedding-v4}" \
    --mem0_embedding_dim "${MEM0_EMBEDDING_DIM:-1024}" \
    --mem0_chunk_turns "${MEM0_CHUNK_TURNS:-6}" \
    --mem0_chunk_max_chars "${MEM0_CHUNK_MAX_CHARS:-12000}" \
    --mem0_chunk_max_images "${MEM0_CHUNK_MAX_IMAGES:-8}" \
    --mem0_retrieve_limit "${MEM0_RETRIEVE_LIMIT:-8}" \
    --mem0_ingest_media_mode "${MEM0_INGEST_MEDIA_MODE:-original}"

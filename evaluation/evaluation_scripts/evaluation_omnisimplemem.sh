#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
OMNISIMPLEMEM_BASELINE_ROOT="${OMNISIMPLEMEM_BASELINE_ROOT:?set OMNISIMPLEMEM_BASELINE_ROOT to the cloned OmniSimpleMem baseline directory}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"

OMNI_EMBEDDING_MODEL="${OMNI_EMBEDDING_MODEL:-BAAI/bge-m3}"
OMNI_EMBEDDING_DIM="${OMNI_EMBEDDING_DIM:-1024}"
OMNI_VISUAL_EMBEDDING_MODEL="${OMNI_VISUAL_EMBEDDING_MODEL:-BAAI/bge-visualized}"
OMNI_VISUAL_EMBEDDING_DIM="${OMNI_VISUAL_EMBEDDING_DIM:-1024}"

python ./main.py \
    --dataset_parent_dir "${DATASET_PARENT_DIR}" \
    --save_root_path "${SAVE_ROOT_PATH}" \
    --eval_strategy "${EVAL_STRATEGY:-time_sequential_memory}" \
    --eval_format multiple_choice \
    --eval_rounds "${EVAL_ROUNDS:-1}" \
    --eval_agent omnisimplemem \
    --model "${MODEL}" \
    --omnisimplemem_baseline_root "${OMNISIMPLEMEM_BASELINE_ROOT}" \
    --omnisimplemem_top_k "${OMNI_TOP_K:-20}" \
    --omnisimplemem_chunk_turns "${OMNI_CHUNK_TURNS:-20}" \
    --omnisimplemem_chunk_max_chars "${OMNI_CHUNK_MAX_CHARS:-6000}" \
    --omnisimplemem_chunk_max_images "${OMNI_CHUNK_MAX_IMAGES:-8}" \
    --omnisimplemem_embedding_model "${OMNI_EMBEDDING_MODEL}" \
    --omnisimplemem_embedding_dim "${OMNI_EMBEDDING_DIM}" \
    --omnisimplemem_visual_embedding_model "${OMNI_VISUAL_EMBEDDING_MODEL}" \
    --omnisimplemem_visual_embedding_dim "${OMNI_VISUAL_EMBEDDING_DIM}"

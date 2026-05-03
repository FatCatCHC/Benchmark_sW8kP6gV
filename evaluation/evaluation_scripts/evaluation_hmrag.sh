#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
HMRAG_EMBEDDING_ROOT="${HMRAG_EMBEDDING_ROOT:?set HMRAG_EMBEDDING_ROOT to the HMRAG embedding output directory}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"

CONVERSATION_DIR_NAME="${CONVERSATION_DIR_NAME:-$(basename "${DATASET_PARENT_DIR}")}"

python ./preprocess/preprocess_hmrag_embedding.py \
    --conversation_dir_name "${CONVERSATION_DIR_NAME}" \
    --output_dir_name "${CONVERSATION_DIR_NAME}" \
    --gpu "${GPU:-0}"

python ./main.py \
    --dataset_parent_dir "${DATASET_PARENT_DIR}" \
    --save_root_path "${SAVE_ROOT_PATH}" \
    --eval_strategy overall_context \
    --eval_format multiple_choice \
    --eval_rounds "${EVAL_ROUNDS:-1}" \
    --eval_agent hmrag \
    --model "${MODEL}" \
    --hmrag_auxiliary_model "${MODEL}" \
    --hmrag_decompose_model "${MODEL}" \
    --hmrag_embedding_root "${HMRAG_EMBEDDING_ROOT}" \
    --hmrag_vector_top_k "${HMRAG_VECTOR_TOP_K:-3}" \
    --hmrag_graph_top_k "${HMRAG_GRAPH_TOP_K:-3}" \
    --hmrag_vector_chunk_tokens "${HMRAG_VECTOR_CHUNK_TOKENS:-256}" \
    --hmrag_graph_chunk_tokens "${HMRAG_GRAPH_CHUNK_TOKENS:-512}"

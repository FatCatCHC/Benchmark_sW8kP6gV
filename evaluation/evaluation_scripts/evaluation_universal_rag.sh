#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
UNIVERSALRAG_EMBEDDING_DIR="${UNIVERSALRAG_EMBEDDING_DIR:?set UNIVERSALRAG_EMBEDDING_DIR to the per-dataset embedding folder}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"
TOP_K="${TOP_K:-10}"

CONVERSATION_DIR_NAME="${CONVERSATION_DIR_NAME:-$(basename "${DATASET_PARENT_DIR}")}"

python ./preprocess/preprocess_universalRAG_embedding.py \
    --conversation_dir_name "${CONVERSATION_DIR_NAME}" \
    --output_dir_name "${CONVERSATION_DIR_NAME}" \
    --gpu "${GPU:-0}"

for cluster in $(ls "${DATASET_PARENT_DIR}"); do
    python ./main.py \
        --dataset_dir_path "${DATASET_PARENT_DIR}/${cluster}" \
        --save_root_path "${SAVE_ROOT_PATH}" \
        --save_dir_name "${cluster}" \
        --eval_strategy "${EVAL_STRATEGY:-time_sequential_memory}" \
        --eval_format multiple_choice \
        --eval_rounds "${EVAL_ROUNDS:-1}" \
        --eval_agent universal_rag \
        --model "${MODEL}" \
        --universal_rag_router_model "${MODEL}" \
        --universal_rag_top_k "${TOP_K}" \
        --universal_rag_text_pkl "${UNIVERSALRAG_EMBEDDING_DIR}/${cluster}/text.pkl" \
        --universal_rag_json_pkl "${UNIVERSALRAG_EMBEDDING_DIR}/${cluster}/json.pkl" \
        --universal_rag_corpus_meta "${UNIVERSALRAG_EMBEDDING_DIR}/${cluster}/corpus_meta.json" \
        --universal_rag_image_pkl "${UNIVERSALRAG_EMBEDDING_DIR}/${cluster}/image.pkl"
done

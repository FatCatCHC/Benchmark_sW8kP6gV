#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${EVAL_DIR}"

DATASET_PARENT_DIR="${DATASET_PARENT_DIR:?set DATASET_PARENT_DIR to the dataset folder containing cluster_* subdirs}"
SAVE_ROOT_PATH="${SAVE_ROOT_PATH:?set SAVE_ROOT_PATH to the output directory}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"
EVAL_AGENT="${EVAL_AGENT:-st_mem}"

case "${EVAL_AGENT}" in
    st_mem|reflexion_mem|mb_mem|gen_agent_mem|lt_mem|mg_mem|sc_mem|mt_mem)
        ;;
    *)
        echo "EVAL_AGENT must be one of: st_mem, reflexion_mem, mb_mem, gen_agent_mem, lt_mem, mg_mem, sc_mem, mt_mem" >&2
        exit 1
        ;;
esac

python ./main.py \
    --dataset_parent_dir "${DATASET_PARENT_DIR}" \
    --save_root_path "${SAVE_ROOT_PATH}" \
    --eval_strategy "${EVAL_STRATEGY:-time_sequential_memory}" \
    --eval_format multiple_choice \
    --eval_rounds "${EVAL_ROUNDS:-1}" \
    --eval_agent "${EVAL_AGENT}" \
    --model "${MODEL}" \
    --mem_engine_retrieval_num "${MEM_ENGINE_RETRIEVAL_NUM:-10}" \
    --mem_engine_gpu "${MEM_ENGINE_GPU:-0}"

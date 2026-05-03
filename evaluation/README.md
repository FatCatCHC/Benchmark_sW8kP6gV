# Evaluation Pipeline

This directory contains the evaluation harness used to benchmark several
multimodal long-context / retrieval / memory agents on the cluster-formatted
QA datasets produced upstream.

## Directory Layout

```
evaluation/
├── main.py                    # entrypoint: argparse + per-cluster dispatch
├── constant.py                # model registry, base URLs, dataset roots
├── dataset_loader.py          # load QA_sample.json + group_chat*.json + user_assistant*.json
├── utils.py                   # API / POST clients, retry helpers, function-call eval
├── candidate_tools.py         # candidate tool catalog used by the function-call task
├── agents/                    # one subpackage per agent family
│   ├── base.py / base_mem_agent.py / base_rag_agent.py
│   ├── prompt.py              # shared system prompts
│   ├── long_context_vlm.py    # baseline: feed full conversation to a VLM
│   ├── universal_rag_agent.py
│   ├── HMRAG/                 # HM-RAG decomposition + dual retrieval + voting
│   ├── VRAG/                  # VRAG-RL style think / search / bbox / answer agent
│   ├── Mem_Engine/            # MemEngine-backed memory agents (st/sc/mb/lt/mg/mt/reflexion/gen)
│   ├── OmniSimpleMem/         # OmniSimpleMem multimodal memory agent
│   ├── MIRIX/                 # MIRIX memory agent
│   ├── Mem0/                  # Mem0 memory agent (Qdrant + API-compatible LLM/embedder)
│   ├── MemVerse/              # MemVerse HTTP /insert + /query memory agent
│   └── MemGPT/                # MemGPT (Letta server) memory agent
├── preprocess/                # offline embedding builders for RAG agents
│   ├── preprocess_universalRAG_embedding.py
│   ├── preprocess_hmrag_embedding.py
│   ├── universal_rag_bge.py
│   └── universal_rag_visual_bge.py
└── evaluation_scripts/        # one canonical shell wrapper per agent
    ├── evaluation_long_context_vlm.sh
    ├── evaluation_mem_engine.sh
    ├── evaluation_universal_rag.sh
    ├── evaluation_universal_rag_caption.sh
    ├── evaluation_hmrag.sh
    ├── evaluation_vrag.sh
    ├── evaluation_omnisimplemem.sh
    ├── evaluation_mirix.sh
    ├── evaluation_mem0.sh
    ├── evaluation_memverse.sh
    └── evaluation_memgpt.sh
```

## Available Agents

`--eval_agent` accepts one of:


| Agent key                                                                                    | Module                                      | Description                                                                                                                           |
| -------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `long_context_vlm`                                                                           | `agents.long_context_vlm`                   | Concatenates the entire conversation and asks the VLM directly. Token budget enforced via `tiktoken`.                                 |
| `universal_rag`                                                                              | `agents.universal_rag_agent`                | LLM router (no/text/json/image) + BGE-large text retrieval + Visualized-BGE image retrieval over precomputed pickles.                 |
| `hmrag`                                                                                      | `agents.HMRAG.hmrag_agent`                  | HM-RAG: query decomposition + parallel vector / graph / web experts + consistency voting + refinement.                                |
| `vrag`                                                                                       | `agents.VRAG.vrag_agent`                    | VRAG-RL style multimodal agent with `<think> / <search> / <bbox> / <answer>` actions over text + image retrieval.                     |
| `omnisimplemem`                                                                              | `agents.OmniSimpleMem.omni_simplemem_agent` | Adapter around the OmniSimpleMem baseline: BGE-M3 + Visualized-BGE in a shared 1024-d space.                                          |
| `mirix`                                                                                      | `agents.MIRIX.mirix_agent`                  | Adapter around a running MIRIX HTTP server (chunked ingest + `retrieve_with_conversation`).                                           |
| `mem0`                                                                                       | `agents.Mem0.mem0_agent`                    | Mem0 memory backend (`Memory.add` / `Memory.search`) backed by a local Qdrant store and an API-compatible LLM / embedder.             |
| `memverse`                                                                                   | `agents.MemVerse.memverse_agent`            | Adapter around a running MemVerse HTTP server (caption-only `/insert` ingest + `/query` retrieval).                                   |
| `memgpt`                                                                                     | `agents.MemGPT.memgpt_agent`                | Adapter around a running MemGPT (Letta) server: chunked archival/messages ingest + archival passage QA.                               |
| `st_mem`, `lt_mem`, `mb_mem`, `sc_mem`, `gen_agent_mem`, `reflexion_mem` | `agents.Mem_Engine.*`                       | MemEngine-backed memory agents (short-term, long-term, MemoryBank, self-controlled, MemoryGraph, multi-trace, generative, reflexion). |


## Evaluation Strategies (`--eval_strategy`)

- `overall_context` – feed the full conversation in file-concat order.
- `only_qa_context` – feed only the QA-relevant evidence.
- `file_sequential_RAG` / `file_sequential_memory` – iterate corpus in file-concat order, retrieving / building memory respectively.
- `time_sequential_RAG` / `time_sequential_memory` – iterate corpus in global timestamp order across all conversation files; required by all RAG / memory agents listed above unless noted otherwise.

## Dataset Layout

Each cluster directory holds:

```
cluster_<index>/
├── QA_sample.json          # required: list of QA items
├── group_chat*.json        # optional: group chat turns
└── user_assistant*.json    # optional: user / assistant turns
```

`--dataset_dir_path` runs a single cluster; `--dataset_parent_dir` iterates
every `cluster_<n>` child of the supplied parent (optionally bounded by
`--start_cluster_index` / `--end_cluster_index`).

## Configuration

All file-system paths are sourced from environment variables and default to
locations relative to this directory; no absolute paths are hard-coded. The
relevant variables are:


| Variable                                              | Default (relative to `evaluation/`)                         | Used by                     |
| ----------------------------------------------------- | ----------------------------------------------------------- | --------------------------- |
| `DATASET_ROOT`                                        | `../../datasets`                                            | `constant.py`               |
| `RAW_DATASET_ROOT`                                    | `../../raw_datasets`                                        | `constant.py`               |
| `EVALUATION_OUTPUT_ROOT`                              | `../../evaluation_output`                                   | `constant.py`               |
| `IMAGE_ROOT_DIR`                                      | `${DATASET_ROOT}/PreFinal_Dataset/MidFinal_Images`          | `constant.py`, agents       |
| `PREFINAL_DATASET_ROOT`                               | `${DATASET_ROOT}/PreFinal_Dataset`                          | preprocess scripts          |
| `UNIVERSALRAG_EMBEDDING_ROOT`                         | `${DATASET_ROOT}/UniversalRAG_embedding`                    | preprocess scripts, VRAG    |
| `UNIVERSALRAG_EMBEDDING_CAPTION_ROOT`                 | `${DATASET_ROOT}/UniversalRAG_embedding_caption`            | preprocess scripts          |
| `HMRAG_EMBEDDING_ROOT`                                | `${DATASET_ROOT}/HMRAG_embedding`                           | preprocess scripts, HMRAG   |
| `BGE_LARGE_MODEL_PATH`                                | HuggingFace `BAAI/bge-large-en-v1.5`                        | preprocess + Mem_Engine SC  |
| `BGE_M3_MODEL_PATH`                                   | HuggingFace `BAAI/bge-m3`                                   | preprocess (Visualized-BGE) |
| `MODEL_API_BASE_URL` / `MODEL_API_KEY`                | API-compatible gateway base URL / api key                   | `constant.py`, `utils.py`   |
| `VLLM_BASE_URL` / `VLLM_API_KEY`                      | local vLLM server                                           | `constant.py`               |


Set these in a `.env` file or your shell before invoking any script;
`main.py` calls `dotenv.load_dotenv()` on startup.

## Running an Evaluation

Each agent has a single canonical script under `evaluation_scripts/`. They
all `cd` into this directory before invoking `python ./main.py`, so they can
be launched from anywhere.

### Long-context VLM baseline

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/long_context_vlm \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_long_context_vlm.sh
```

### MemEngine memory agents (short-term, long-term, ...)

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/st_mem \
EVAL_AGENT=st_mem \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_mem_engine.sh
```

`EVAL_AGENT` is one of `st_mem | reflexion_mem | mb_mem | gen_agent_mem | lt_mem | mt_mem`.

### Universal RAG (text + json + image routes)

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
UNIVERSALRAG_EMBEDDING_DIR=/path/to/UniversalRAG_embedding/MidFinal \
SAVE_ROOT_PATH=/path/to/output/universal_rag \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_universal_rag.sh
```

The script first runs `preprocess/preprocess_universalRAG_embedding.py` to
build / refresh the per-cluster `text.pkl`, `json.pkl`, `image.pkl`, and
`corpus_meta.json`, then iterates over every `cluster_*` and dispatches
`main.py`. The caption-only variant uses
`evaluation_universal_rag_caption.sh` and feeds captions instead of raw images.

### HM-RAG

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/hmrag \
HMRAG_EMBEDDING_ROOT=/path/to/HMRAG_embedding \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_hmrag.sh
```

### VRAG

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/vrag \
UNIVERSALRAG_EMBEDDING_ROOT=/path/to/UniversalRAG_embedding/MidFinal \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_vrag.sh
```

### OmniSimpleMem

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/omnisimplemem \
OMNISIMPLEMEM_BASELINE_ROOT=/path/to/baselines/SimpleMem/OmniSimpleMem \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_omnisimplemem.sh
```

### MIRIX

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/mirix \
MIRIX_API_URL=http://localhost:8531 \
MIRIX_API_KEY=<mirix-server-key> \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_mirix.sh
```

### Mem0

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/mem0 \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_mem0.sh
```

`Mem0Agent` writes its Qdrant store under `${SAVE_ROOT_PATH}/<cluster>/checkpoint/mem0_qdrant`
unless `--mem0_qdrant_path` is passed explicitly.

### MemVerse

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/memverse \
MEMVERSE_API_URL=http://127.0.0.1:8000 \
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_memverse.sh
```

### MemGPT (Letta server)

```bash
DATASET_PARENT_DIR=/path/to/PreFinal_Dataset/MidFinal \
SAVE_ROOT_PATH=/path/to/output/memgpt \
MEMGPT_SERVER_URL=http://localhost:8283 \
MEMGPT_SERVER_PASSWORD=<letta-key>      # or LETTA_SERVER_PASSWORD
MODEL=qwen3-vl-235b-a22b-instruct \
bash evaluation_scripts/evaluation_memgpt.sh
```

`MEMGPT_SERVER_URL` falls back to `LETTA_SERVER_URL`, then `http://localhost:8283`.
The default ingest mode is `archival` (passages + hybrid QA); pass
`MEMGPT_INGEST_MODE=messages` to use the legacy `agents.messages.create` path.

## Output Layout

Each run writes per-cluster JSON files under
`${SAVE_ROOT_PATH}/${cluster_name}/`:

```
<eval_strategy>__<eval_agent>__<model>__round<i>.json           # per-QA result + label
<eval_strategy>__<eval_agent>__<model>__round<i>_readable.json  # full chat trace per QA
checkpoint/                                                      # mem_engine state when applicable
```

## Adding a New Agent

1. Drop a module under `agents/` exposing a class that subclasses
  `BaseAgent`, `BaseRAGAgent`, or `BaseMemoryAgent` and implements
   `evaluate_single_qa` (and optionally `build_memory`).
2. Register it in the `_EVAL_AGENT_MODULE_CLASS` dict at the top of
  `main.py`. The argparse `--eval_agent` choices are derived from this dict.
3. Add a new shell wrapper under `evaluation_scripts/` exposing the same
  environment-variable-driven pattern as the existing ones.


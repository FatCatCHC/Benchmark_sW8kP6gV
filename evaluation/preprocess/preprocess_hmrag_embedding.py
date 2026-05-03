from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from typing import List

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT_PREFINAL_DATASET_ROOT = os.environ.get(
    "PREFINAL_DATASET_ROOT", os.path.join(_REPO_ROOT, "datasets", "PreFinal_Dataset")
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "HMRAG_EMBEDDING_ROOT", os.path.join(_REPO_ROOT, "datasets", "HMRAG_embedding")
)
DEFAULT_BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"

def split_text_into_chunks(tokenizer, text: str, max_tokens: int = 512) -> List[str]:
    token_ids = tokenizer.encode(text, truncation=False)
    tokenized_chunks = [token_ids[i : i + max_tokens] for i in range(0, len(token_ids), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in tokenized_chunks]

def _list_corpus_json_in_cluster(cluster_dir: str) -> list[str]:
    g = sorted(glob.glob(os.path.join(cluster_dir, "group_chat*.json")))
    u = sorted(glob.glob(os.path.join(cluster_dir, "user_assistant*.json")))
    return g + u

def _append_if_nonempty(parts: List[str], text: str | None) -> None:
    if text is None:
        return
    s = str(text).strip()
    if s:
        parts.append(s)

def _turn_to_retrieval_text_segments(turn: dict) -> List[str]:
    sender = turn.get("sender_name", "unknown")
    timestamp = turn.get("timestamp", "")
    content_type = turn.get("content_type")

    parts: List[str] = []
    if content_type == "text":
        _append_if_nonempty(parts, f"{sender} at {timestamp}: {turn.get('content', '')}")
        return parts

    if content_type == "image":
        parts.append(f"{sender} sent an image at {timestamp}")
        _append_if_nonempty(parts, f"Caption: {turn.get('caption', '')}")
        return parts

    if content_type == "json_evidence":
        parts.append(f"{sender} sent a json evidence document at {timestamp}")

        _append_if_nonempty(parts, f"Caption: {turn.get('caption', '')}")
        return parts

    raw_content = turn.get("content", "")
    _append_if_nonempty(parts, f"{sender} at {timestamp}: {raw_content}")
    return parts

def _load_cluster_corpus_text(cluster_dir: str) -> str:
    files = _list_corpus_json_in_cluster(cluster_dir)
    if not files:
        raise FileNotFoundError(
            f"No corpus JSON files found in {cluster_dir} "
            "(expected group_chat*.json and/or user_assistant*.json)."
        )

    all_segments: List[str] = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        conv = data.get("conversation", [])
        if not isinstance(conv, list):
            continue
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            all_segments.extend(_turn_to_retrieval_text_segments(turn))

    corpus_text = "\n\n".join(seg for seg in all_segments if seg.strip())
    if not corpus_text.strip():
        raise FileNotFoundError(f"No usable retrieval text extracted from {cluster_dir}")
    return corpus_text

def _encode_chunk_map(model, chunks: List[str], batch_size: int, prefix: str) -> tuple[dict, dict]:
    meta_map = {}
    feat_map = {}
    if not chunks:
        return meta_map, feat_map

    ids = [f"{prefix}_part{i + 1}" for i in range(len(chunks))]
    for cid, ch in zip(ids, chunks):
        meta_map[cid] = ch

    enc = model.encode(
        chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    enc = enc.detach().cpu().numpy()
    for cid, row in zip(ids, enc):
        feat_map[cid] = row.astype("float32")
    return meta_map, feat_map

def main():
    parser = argparse.ArgumentParser(description="Preprocess conversations -> HMRAG vector/graph BGE pickles + meta JSON.")
    parser.add_argument(
        "--conversation_dir_name",
        type=str,
        default="",
        help="Leaf folder name under DEFAULT_PREFINAL_DATASET_ROOT (e.g. Fact_and_Preference). Empty = error.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="",
        help="Directory to write HMRAG outputs under DEFAULT_OUTPUT_ROOT. Empty defaults to conversation_dir_name.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--vector_chunk_tokens", type=int, default=256)
    parser.add_argument("--graph_chunk_tokens", type=int, default=512)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index for encoding (e.g. 0 -> cuda:0). Requires CUDA and a GPU-enabled PyTorch build.",
    )
    args = parser.parse_args()

    load_dotenv()

    conversation_dir_name = args.conversation_dir_name.strip()
    if not conversation_dir_name:
        print("Error: --conversation_dir_name is required (non-empty).", file=sys.stderr)
        sys.exit(1)

    output_dir_name = args.output_dir_name.strip() or conversation_dir_name
    dataset_leaf_dir = os.path.join(DEFAULT_PREFINAL_DATASET_ROOT, conversation_dir_name)
    output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(dataset_leaf_dir):
        print(f"Error: dataset leaf directory does not exist: {dataset_leaf_dir}", file=sys.stderr)
        sys.exit(1)

    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. Install a CUDA-enabled PyTorch build or use a machine with a GPU.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        print(
            f"Error: --gpu {args.gpu} is invalid (device count: {torch.cuda.device_count()}).",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}")
    device_str = str(device)
    model_path = os.environ.get("BGE_LARGE_MODEL_PATH", "").strip()
    model_id = model_path if model_path else DEFAULT_BGE_MODEL_ID
    print(f"Loading SentenceTransformer: {model_id} -> {device_str}")
    model = SentenceTransformer(model_id, device=device_str)

    cluster_dirs = sorted(
        p for p in glob.glob(os.path.join(dataset_leaf_dir, "cluster_*")) if os.path.isdir(p)
    )
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster_* subdirectories under {dataset_leaf_dir}")

    for cluster_dir in cluster_dirs:
        cluster_name = os.path.basename(cluster_dir)
        corpus_text = _load_cluster_corpus_text(cluster_dir)

        vector_chunks = split_text_into_chunks(
            model.tokenizer,
            corpus_text,
            max_tokens=args.vector_chunk_tokens,
        )
        graph_chunks = split_text_into_chunks(
            model.tokenizer,
            corpus_text,
            max_tokens=args.graph_chunk_tokens,
        )

        vector_meta, vector_feats = _encode_chunk_map(
            model,
            vector_chunks,
            batch_size=args.batch_size,
            prefix="vector",
        )
        graph_meta, graph_feats = _encode_chunk_map(
            model,
            graph_chunks,
            batch_size=args.batch_size,
            prefix="graph",
        )

        output_cluster_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(output_cluster_dir, exist_ok=True)

        vector_path = os.path.join(output_cluster_dir, "vector.pkl")
        graph_path = os.path.join(output_cluster_dir, "graph.pkl")
        meta_path = os.path.join(output_cluster_dir, "hmrag_meta.json")

        with open(vector_path, "wb") as f:
            pickle.dump(vector_feats, f)
        with open(graph_path, "wb") as f:
            pickle.dump(graph_feats, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vector": vector_meta,
                    "graph": graph_meta,
                    "corpus_text": corpus_text,
                    "config": {
                        "conversation_dir_name": conversation_dir_name,
                        "cluster_name": cluster_name,
                        "vector_chunk_tokens": args.vector_chunk_tokens,
                        "graph_chunk_tokens": args.graph_chunk_tokens,
                        "model_id": model_id,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Wrote {vector_path} ({len(vector_feats)} vectors)")
        print(f"Wrote {graph_path} ({len(graph_feats)} vectors)")
        print(f"Wrote {meta_path}")

if __name__ == "__main__":
    main()

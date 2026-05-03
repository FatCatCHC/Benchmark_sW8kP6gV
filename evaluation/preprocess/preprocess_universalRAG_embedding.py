from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

from dotenv import load_dotenv

from universal_rag_bge import DEFAULT_BGE_MODEL_ID, split_text_into_chunks
from universal_rag_visual_bge import DEFAULT_BGE_M3_MODEL_ID
from constant import Image_Root_Dir_Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT_PREFINAL_DATASET_ROOT = os.environ.get(
    "PREFINAL_DATASET_ROOT", os.path.join(_REPO_ROOT, "datasets", "PreFinal_Dataset")
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "UNIVERSALRAG_EMBEDDING_ROOT", os.path.join(_REPO_ROOT, "datasets", "UniversalRAG_embedding")
)

def _list_corpus_json_in_cluster(cluster_dir: str) -> list[str]:
    g = sorted(glob.glob(os.path.join(cluster_dir, "group_chat*.json")))
    u = sorted(glob.glob(os.path.join(cluster_dir, "user_assistant*.json")))
    return g + u

def _load_conversation_turns(cluster_dir: str):

    text_units = []
    all_text_segments = []

    cluster_tag = os.path.basename(cluster_dir)
    files = _list_corpus_json_in_cluster(cluster_dir)
    for filepath in files:
        base = os.path.basename(filepath)
        rel_id = f"{cluster_tag}/{base}"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(filepath)
        conv = data.get("conversation", [])
        for turn_idx, turn in enumerate(conv):
            if turn.get("content_type") != "text":
                continue
            content = turn.get("content") or ""
            sender = turn.get("sender_name", "unknown")
            line = f"{sender} at {turn.get('timestamp', '')} in {rel_id}: {content}"
            tid = f"{rel_id}#t{turn_idx}"
            text_units.append((tid, line))
            all_text_segments.append(line)

    if not text_units and not all_text_segments:
        raise FileNotFoundError(
            f"No text turns found under clusters in {cluster_dir} "
            "(expected group_chat*.json / user_assistant*.json with text turns)."
        )
    return text_units, all_text_segments

def _normalize_local_image_path(image_path: str, json_dir: str) -> str:
    if not image_path or not isinstance(image_path, str):
        return ""
    u = image_path.strip()
    if u.startswith(("http://", "https://")):
        return ""
    if u.startswith("file://"):
        u = u[7:]
    if os.path.isabs(u):
        return u if os.path.isfile(u) else ""
    cand = os.path.normpath(os.path.join(json_dir, u))
    return cand if os.path.isfile(cand) else ""

def _load_image_turns(cluster_dir: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    cluster_tag = os.path.basename(cluster_dir)
    for filepath in _list_corpus_json_in_cluster(cluster_dir):
        json_dir = os.path.dirname(os.path.abspath(filepath))
        base = os.path.basename(filepath)
        rel_id = f"{cluster_tag}/{base}"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for turn_idx, turn in enumerate(data.get("conversation", [])):
            if turn.get("content_type") != "image":
                continue

            path = os.path.join(Image_Root_Dir_Path, turn.get("image_path") or "")

            if not path:
                continue
            sender = turn.get("sender_name", "unknown")
            line = f"{sender} sent an image at {turn.get('timestamp', '')} in {rel_id}"
            cap = turn.get("caption")
            if cap:
                line += f" Caption: {cap}"
            tid = f"{rel_id}#i{turn_idx}"
            out.append((tid, path, line))
    return out

def main():
    parser = argparse.ArgumentParser(description="Preprocess conversations -> BGE text & json pickles + meta JSON.")
    parser.add_argument(
        "--conversation_dir_name",
        type=str,
        default="",
        help="Leaf folder name under DEFAULT_PREFINAL_DATASET_ROOT (e.g. with_noise_inserted). Empty = error.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="",
        help="Directory to write text.pkl, json.pkl, corpus_meta.json under DEFAULT_OUTPUT_ROOT. Empty = error.",
    )
    parser.add_argument(
        "--bge_model_path",
        type=str,
        default="",
        help="Local path to BGE-large weights. If empty, uses BGE_LARGE_MODEL_PATH env or HuggingFace id.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--document_max_tokens", type=int, default=512)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index for encoding (e.g. 0 -> cuda:0). Requires CUDA and a GPU-enabled PyTorch build.",
    )
    parser.add_argument(
        "--visual_bge_weight",
        type=str,
        default="",
        help="Path to Visualized-BGE .pth (e.g. Visualized_m3.pth). If set, writes image.pkl + image entries in corpus_meta.json.",
    )
    args = parser.parse_args()

    load_dotenv()

    conversation_dir_name = args.conversation_dir_name.strip()
    if not conversation_dir_name:
        print("Error: --conversation_dir_name is required (non-empty).", file=sys.stderr)
        sys.exit(1)
    output_dir_name = args.output_dir_name.strip()
    if not output_dir_name:
        output_dir_name = conversation_dir_name
    dataset_leaf_dir = os.path.join(DEFAULT_PREFINAL_DATASET_ROOT, conversation_dir_name)
    output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(dataset_leaf_dir):
        print(f"Error: dataset leaf directory does not exist: {dataset_leaf_dir}", file=sys.stderr)
        sys.exit(1)

    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Install a CUDA-enabled PyTorch build or use a machine with a GPU.", file=sys.stderr)
        sys.exit(1)
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        print(
            f"Error: --gpu {args.gpu} is invalid (device count: {torch.cuda.device_count()}).",
            file=sys.stderr,
        )
        sys.exit(1)
    device = torch.device(f"cuda:{args.gpu}")
    device_str = str(device)

    model_path = args.bge_model_path.strip() or os.environ.get("BGE_LARGE_MODEL_PATH", "").strip()
    model_id = model_path if model_path else DEFAULT_BGE_MODEL_ID
    print(f"Loading SentenceTransformer: {model_id} -> {device_str}")
    model = SentenceTransformer(model_id, device=device_str)

    visual_model = None
    visual_bge_path = args.visual_bge_weight.strip() or os.environ.get("BGE_M3_MODEL_PATH", "").strip()
    visual_bge_weight_path = os.path.join(visual_bge_path, "Visualized_m3.pth")
    if visual_bge_weight_path:
        from universal_rag_visual_bge import load_visualized_bge

        if not os.path.isfile(visual_bge_weight_path):
            print(f"Error: --visual_bge_weight file not found: {visual_bge_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading Visualized-BGE backbone={DEFAULT_BGE_M3_MODEL_ID} weight={visual_bge_weight_path}")
        visual_model = load_visualized_bge(
            DEFAULT_BGE_M3_MODEL_ID,
            visual_bge_weight_path,
        )

    cluster_dirs = sorted(
        p for p in glob.glob(os.path.join(dataset_leaf_dir, "cluster_*")) if os.path.isdir(p)
    )
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster_* subdirectories under {dataset_leaf_dir}")

    for cluster_dir in cluster_dirs:
        text_units, all_text_segments = _load_conversation_turns(cluster_dir)

        meta_text = {}
        text_feats = {}
        if text_units:
            t_ids = [u[0] for u in text_units]
            t_texts = [u[1] for u in text_units]
            for tid, txt in zip(t_ids, t_texts):
                meta_text[tid] = txt
            enc_p = model.encode(
                t_texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
            enc_p = enc_p.detach().cpu().numpy()
            for tid, row in zip(t_ids, enc_p):
                text_feats[tid] = row.astype("float32")

        full_doc = "\n".join(all_text_segments)
        meta_json = {}
        json_feats = {}
        if full_doc.strip():
            chunks = split_text_into_chunks(model.tokenizer, full_doc, max_tokens=args.document_max_tokens)
            for i, ch in enumerate(chunks):
                jid = f"global_json_part{i + 1}"
                meta_json[jid] = ch
            enc_d = model.encode(
                chunks,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
            enc_d = enc_d.detach().cpu().numpy()
            for jid, row in zip(meta_json.keys(), enc_d):
                json_feats[jid] = row.astype("float32")

        meta_image: dict = {}
        image_feats: dict = {}
        if visual_model is not None:
            from tqdm import tqdm

            from universal_rag_visual_bge import encode_visual_image_path

            image_triples = _load_image_turns(cluster_dir)
            for tid, img_path, line in tqdm(image_triples, desc=f"Visual-BGE images {os.path.basename(cluster_dir)}"):
                meta_image[tid] = {"path": img_path, "line": line}
                try:
                    image_feats[tid] = encode_visual_image_path(visual_model, img_path)
                except Exception as e:
                    print(f"[warn] skip image {tid} {img_path}: {e}", file=sys.stderr)
                    meta_image.pop(tid, None)

        output_cluster_dir = os.path.join(output_dir,  os.path.basename(cluster_dir))
        os.makedirs(output_cluster_dir, exist_ok=True)
        text_path = os.path.join(output_cluster_dir, "text.pkl")
        json_path = os.path.join(output_cluster_dir, "json.pkl")
        image_path = os.path.join(output_cluster_dir, "image.pkl")
        meta_path = os.path.join(output_cluster_dir, "corpus_meta.json")

        with open(text_path, "wb") as f:
            pickle.dump(text_feats, f)
        with open(json_path, "wb") as f:
            pickle.dump(json_feats, f)
        meta_obj = {"text": meta_text, "json": meta_json}
        if visual_model is not None:
            with open(image_path, "wb") as f:
                pickle.dump(image_feats, f)
            meta_obj["image"] = meta_image
            print(f"Wrote {image_path} ({len(image_feats)} vectors)")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, ensure_ascii=False, indent=2)

        print(f"Wrote {text_path} ({len(text_feats)} vectors)")
        print(f"Wrote {json_path} ({len(json_feats)} vectors)")
        print(f"Wrote {meta_path}")

if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

def _as_numpy_f32(v) -> np.ndarray:
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass
    return np.asarray(v, dtype=np.float32)

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"

def split_text_into_chunks(tokenizer, text: str, max_tokens: int = 512) -> List[str]:
    token_ids = tokenizer.encode(text, truncation=False)
    tokenized_chunks = [token_ids[i : i + max_tokens] for i in range(0, len(token_ids), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in tokenized_chunks]

def load_corpus_meta(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    text_map = meta.get("text")
    json_map = meta.get("json")

    if text_map is None and isinstance(meta.get("paragraph"), dict):
        text_map = meta["paragraph"]
    if json_map is None and isinstance(meta.get("document"), dict):
        json_map = meta["document"]
    if not isinstance(text_map, dict) or not isinstance(json_map, dict):
        raise ValueError(
            "corpus_meta.json must contain 'text' and 'json' (or legacy 'paragraph' and 'document')."
        )
    image_map = meta.get("image")
    if not isinstance(image_map, dict):
        image_map = {}
    return {"text": text_map, "json": json_map, "image": image_map}

class BGECorpusRetriever:

    def __init__(self, pkl_path: str, id_to_text: Dict[str, str]):
        with open(pkl_path, "rb") as f:
            raw: Dict[str, np.ndarray] = pickle.load(f)
        self.ids: List[str] = []
        self.texts: List[str] = []
        vecs: List[np.ndarray] = []
        for k, v in raw.items():
            if k not in id_to_text:
                continue
            self.ids.append(k)
            self.texts.append(id_to_text[k])
            vecs.append(_as_numpy_f32(v))
        if not vecs:
            self.dim = 1024
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        else:
            self.vectors = np.stack(vecs)
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-12
            self.vectors = self.vectors / norms
            self.dim = self.vectors.shape[1]

    def retrieve(self, query_embedding_normalized: np.ndarray, top_k: int) -> Tuple[List[str], List[str]]:
        if self.vectors.shape[0] == 0:
            return [], []
        q = np.asarray(query_embedding_normalized, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-12)
        sim = self.vectors @ q
        k = min(top_k, len(sim))
        idx = np.argsort(-sim)[:k]
        texts = [self.texts[i] for i in idx]
        ids = [self.ids[i] for i in idx]
        return texts, ids

def encode_query(bge_model, question: str) -> np.ndarray:
    v = bge_model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(v[0], dtype=np.float32)

def encode_passages(bge_model, passages: List[str], batch_size: int = 64) -> np.ndarray:
    if not passages:
        return np.zeros((0, 1024), dtype=np.float32)
    return np.asarray(
        bge_model.encode(passages, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )

def retrieve_from_passage_list(
    bge_model,
    question: str,
    passages: List[str],
    top_k: int,
    similarity_device: str | None = None,
) -> Tuple[List[str], List[int]]:
    if not passages:
        return [], []
    pv = encode_passages(bge_model, passages)
    qv = encode_query(bge_model, question)
    k = min(top_k, len(pv))
    if k <= 0:
        return [], []

    if similarity_device and similarity_device.startswith("cuda") and torch.cuda.is_available():
        try:
            pv_t = torch.from_numpy(pv).to(similarity_device)
            qv_t = torch.from_numpy(qv).to(similarity_device)
            sim_t = pv_t @ qv_t
            idx_t = torch.topk(sim_t, k=k, dim=0).indices
            idx = idx_t.detach().cpu().tolist()
            return [passages[i] for i in idx], idx
        except Exception:
            pass

    sim = pv @ qv
    idx = np.argsort(-sim)[:k]
    return [passages[i] for i in idx], idx.tolist()

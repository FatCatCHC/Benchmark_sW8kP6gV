from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from universal_rag_bge import QUERY_INSTRUCTION

DEFAULT_BGE_M3_MODEL_ID = "BAAI/bge-m3"

def _as_numpy_f32(v) -> np.ndarray:
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass
    return np.asarray(v, dtype=np.float32)

def import_visualized_bge_class():
    try:
        from visual_bge.modeling import Visualized_BGE

        return Visualized_BGE
    except ImportError as e:
        raise ImportError(
            "visual_bge is not installed. Install from FlagEmbedding/research/visual_bge "
            "(pip install -e .) and dependencies (torchvision timm einops ftfy). "
            f"Original error: {e}"
        ) from e

def load_visualized_bge(model_name_bge: str, model_weight: str, device: str | None = None):
    Visualized_BGE = import_visualized_bge_class()
    if not model_weight or not os.path.isfile(model_weight):
        raise FileNotFoundError(f"Visualized-BGE weight not found: {model_weight!r}")
    model = Visualized_BGE(model_name_bge=model_name_bge, model_weight=model_weight)
    if device:
        model = model.to(device)
    model.eval()
    return model

def encode_visual_text_query(model, question: str) -> np.ndarray:
    import torch

    text = QUERY_INSTRUCTION + question
    with torch.no_grad():
        v = model.encode(text=text)
    q = _as_numpy_f32(v).reshape(-1)
    n = np.linalg.norm(q) + 1e-12
    return (q / n).astype(np.float32)

def encode_visual_image_path(model, image_path: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        v = model.encode(image=image_path)
    x = _as_numpy_f32(v).reshape(-1)
    n = np.linalg.norm(x) + 1e-12
    return (x / n).astype(np.float32)

class BGEVisualImageRetriever:

    def __init__(self, pkl_path: str, id_to_record: Dict[str, Dict[str, Any]]):
        with open(pkl_path, "rb") as f:
            raw: Dict[str, np.ndarray] = pickle.load(f)
        self.ids: List[str] = []
        self.records: List[Dict[str, Any]] = []
        vecs: List[np.ndarray] = []
        for k, v in raw.items():
            if k not in id_to_record:
                continue
            rec = id_to_record[k]
            if not isinstance(rec, dict):
                continue
            path = rec.get("path")
            if not path or not os.path.isfile(str(path)):
                continue
            line = str(rec.get("line", ""))
            self.ids.append(k)
            self.records.append({"path": str(path), "line": line})
            vecs.append(_as_numpy_f32(v))
        if not vecs:
            self.dim = 1024
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        else:
            self.vectors = np.stack(vecs)
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-12
            self.vectors = self.vectors / norms
            self.dim = self.vectors.shape[1]
        self._torch_vectors: Dict[str, torch.Tensor] = {}

    def retrieve(
        self, query_embedding_normalized: np.ndarray, top_k: int, similarity_device: str | None = None
    ) -> List[Tuple[str, str, str]]:
        if self.vectors.shape[0] == 0:
            return []
        k = min(top_k, len(self.vectors))
        q = np.asarray(query_embedding_normalized, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-12)

        if similarity_device and similarity_device.startswith("cuda") and torch.cuda.is_available():
            try:
                vectors_t = self._torch_vectors.get(similarity_device)
                if vectors_t is None:
                    vectors_t = torch.from_numpy(self.vectors).to(similarity_device)
                    self._torch_vectors[similarity_device] = vectors_t
                q_t = torch.from_numpy(q).to(similarity_device)
                sim_t = vectors_t @ q_t
                idx = torch.topk(sim_t, k=k, dim=0).indices.detach().cpu().tolist()
            except Exception:
                sim = self.vectors @ q
                idx = np.argsort(-sim)[:k].tolist()
        else:
            sim = self.vectors @ q
            idx = np.argsort(-sim)[:k].tolist()

        out: List[Tuple[str, str, str]] = []
        for i in idx:
            rec = self.records[i]
            out.append((self.ids[i], rec["path"], rec["line"]))
        return out

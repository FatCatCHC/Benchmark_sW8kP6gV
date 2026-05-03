from __future__ import annotations

import json
import os
from typing import List, Tuple

from universal_rag_bge import BGECorpusRetriever, encode_query, retrieve_from_passage_list, split_text_into_chunks

def messages_to_text_blob(messages: list) -> str:
    parts: List[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text") or "")
    return "\n\n".join(x for x in parts if x.strip())

def evidence_to_text_blob(qa_sample: dict) -> str:
    ev = qa_sample.get("evidence") or {}
    texts = list(ev.get("text_evidence") or [])
    tables = ev.get("table_evidence") or []
    blob = "\n".join(texts + [json.dumps(t, ensure_ascii=False) for t in tables])
    return blob.strip()

def load_precomputed_retrievers(
    vector_pkl: str,
    graph_pkl: str,
    meta_path: str,
) -> tuple[BGECorpusRetriever | None, BGECorpusRetriever | None]:
    if not meta_path or not os.path.isfile(meta_path):
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vector_r = None
    graph_r = None
    vector_map = meta.get("vector")
    graph_map = meta.get("graph")
    if vector_pkl and os.path.isfile(vector_pkl) and isinstance(vector_map, dict):
        vector_r = BGECorpusRetriever(vector_pkl, vector_map)
    if graph_pkl and os.path.isfile(graph_pkl) and isinstance(graph_map, dict):
        graph_r = BGECorpusRetriever(graph_pkl, graph_map)
    return vector_r, graph_r

def dual_layer_retrieve(
    bge_model,
    question: str,
    corpus_text: str,
    vector_chunk_tokens: int,
    graph_chunk_tokens: int,
    vector_top_k: int,
    graph_top_k: int,
    vector_retriever: BGECorpusRetriever | None = None,
    graph_retriever: BGECorpusRetriever | None = None,
) -> Tuple[str, str]:
    if vector_retriever is not None or graph_retriever is not None:
        qv = encode_query(bge_model, question)
        vec_passages = []
        graph_passages = []
        if vector_retriever is not None:
            vec_passages, _ = vector_retriever.retrieve(qv, vector_top_k)
        if graph_retriever is not None:
            graph_passages, _ = graph_retriever.retrieve(qv, graph_top_k)
    else:
        if not corpus_text.strip():
            empty = "(No local text passages available for retrieval.)"
            return empty, empty

        tok = bge_model.tokenizer
        vec_chunks = split_text_into_chunks(tok, corpus_text, max_tokens=vector_chunk_tokens)
        graph_chunks = split_text_into_chunks(tok, corpus_text, max_tokens=graph_chunk_tokens)

        vec_passages, _ = retrieve_from_passage_list(bge_model, question, vec_chunks, vector_top_k)
        graph_passages, _ = retrieve_from_passage_list(bge_model, question, graph_chunks, graph_top_k)

    if not vec_passages and not graph_passages and not corpus_text.strip():
        empty = "(No local text passages available for retrieval.)"
        return empty, empty

    def fmt(title: str, passages: List[str]) -> str:
        if not passages:
            return f"{title}\n(no passages scored)"
        lines = [f"[{i + 1}] {p}" for i, p in enumerate(passages)]
        return title + "\n" + "\n\n".join(lines)

    return fmt("Vector-style retrieval (fine-grained chunks):", vec_passages), fmt(
        "Graph-style retrieval (coarse chunks / broader context):", graph_passages
    )

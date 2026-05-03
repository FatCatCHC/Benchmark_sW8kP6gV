import os
import glob
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

TIME_SEQUENTIAL_EVAL_STRATEGIES = frozenset({"time_sequential_RAG", "time_sequential_memory"})

def _parse_turn_timestamp(turn: Dict[str, Any]) -> datetime:
    ts = turn.get("timestamp")
    if isinstance(ts, str) and ts.strip():
        s = ts.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
    return datetime.min

def flat_index_to_stream_coords(streams: List[List[dict]], flat_idx: int) -> Tuple[int, int]:
    if flat_idx < 0:
        raise IndexError("flat_idx must be non-negative")
    offset = 0
    for fi, stream in enumerate(streams):
        n = len(stream)
        if flat_idx < offset + n:
            return fi, flat_idx - offset
        offset += n
    raise IndexError(f"conversation_turn_index {flat_idx} out of range for {offset} total turns")

def build_memory_time_ordered_turns(
    conversation_streams: List[List[dict]],
    flat_anchor_index: int,
) -> List[dict]:
    if not conversation_streams:
        return []
    total = sum(len(s) for s in conversation_streams)
    if total == 0:
        return []
    flat_anchor_index = min(flat_anchor_index, total - 1)
    anchor_fi, anchor_li = flat_index_to_stream_coords(conversation_streams, flat_anchor_index)
    anchor_key = (
        _parse_turn_timestamp(conversation_streams[anchor_fi][anchor_li]),
        anchor_fi,
        anchor_li,
    )
    keyed: List[Tuple[Tuple[datetime, int, int], dict]] = []
    for fi, stream in enumerate(conversation_streams):
        for li, turn in enumerate(stream):
            dt = _parse_turn_timestamp(turn)
            keyed.append(((dt, fi, li), turn))
    keyed.sort(key=lambda x: x[0])
    return [turn for k, turn in keyed if k <= anchor_key]

def sort_conversation_streams_by_global_timestamp(
    conversation_streams: List[List[dict]],
) -> List[dict]:
    if not conversation_streams:
        return []
    keyed: List[Tuple[Tuple[datetime, int, int], dict]] = []
    for fi, stream in enumerate(conversation_streams):
        for li, turn in enumerate(stream):
            dt = _parse_turn_timestamp(turn)
            keyed.append(((dt, fi, li), turn))
    keyed.sort(key=lambda x: x[0])
    sorted_turns = [turn for _, turn in keyed]
    return sorted_turns

def _load_group_chat_stream(filepath: str) -> List[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    conv = data.get("conversation", [])
    if not isinstance(conv, list):
        return []
    return conv

def _list_corpus_json_files(dataset_dir: str) -> List[str]:
    g = sorted(glob.glob(os.path.join(dataset_dir, "group_chat*.json")))
    u = sorted(glob.glob(os.path.join(dataset_dir, "user_assistant*.json")))
    paths = g + u
    if not paths:
        raise FileNotFoundError(
            f"No corpus JSON files (group_chat*.json / user_assistant*.json) in {dataset_dir}"
        )
    return paths

def load_cluster_dataset_dir(dataset_dir_path: str) -> Tuple[List[Dict], List[Dict], List[List[Dict]]]:
    if not os.path.isdir(dataset_dir_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir_path}")

    qa_path = os.path.join(dataset_dir_path, "QA_sample.json")
    if not os.path.isfile(qa_path):
        raise FileNotFoundError(f"QA_sample.json not found in {dataset_dir_path}")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_root = json.load(f)

    if not qa_root:
        raise ValueError(f"No QA entries in {qa_path}")

    corpus_paths = _list_corpus_json_files(dataset_dir_path)
    conversation_streams: List[List[dict]] = []
    for fp in corpus_paths:
        conversation_streams.append(_load_group_chat_stream(fp))

    all_conversations: List[dict] = []
    for conv in conversation_streams:
        all_conversations.extend(conv)

    return all_conversations, qa_root, conversation_streams

def load_qa_dataset(dataset_dir_path: str) -> Tuple[List[Dict], List[Dict], List[List[Dict]]]:
    if not os.path.isdir(dataset_dir_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir_path}")

    qa_sample = os.path.join(dataset_dir_path, "QA_sample.json")
    if os.path.isfile(qa_sample):
        return load_cluster_dataset_dir(dataset_dir_path)
    assert False, f"QA_sample.json not found in {dataset_dir_path}"

def load_qa_dataset_by_strategy(
    dataset_dir_path: str,
    eval_strategy: str,
) -> Dict[str, Any]:
    conversations, qa_list, conversation_streams = load_qa_dataset(dataset_dir_path)
    if eval_strategy in TIME_SEQUENTIAL_EVAL_STRATEGIES:
        sorted_conv = sort_conversation_streams_by_global_timestamp(
            conversation_streams
        )
        return {
            "conversation": sorted_conv,
            "qas": qa_list,
            "conversation_streams": conversation_streams,
        }
    return {
        "conversation": conversations,
        "qas": qa_list,
        "conversation_streams": conversation_streams,
    }

import os
import sys
from datetime import datetime
from typing import Any, List, Optional, cast

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import MBMemory
from default_config.DefaultMemoryConfig import DEFAULT_MBMEMORY

def _coarse_timestamp_10min_bucket(ts: Any) -> Any:
    if not isinstance(ts, str):
        return ts
    s = ts.strip()
    if not s:
        return ts
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    else:
        return ts
    floored = dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)
    return floored.strftime("%Y-%m-%d %H:%M:%S")

def _mb_max_bucketed_wall_time(conversations: list) -> Optional[str]:
    best: Optional[datetime] = None
    for turn in conversations or []:
        ts = turn.get("timestamp")
        if not isinstance(ts, str) or not ts.strip():
            continue
        s = ts.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            continue
        if best is None or dt > best:
            best = dt
    if best is None:
        return None
    return _coarse_timestamp_10min_bucket(best.strftime("%Y-%m-%d %H:%M:%S"))

class MBMemAgent(BaseMemEngineAgent):
    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = MBMemory
        self.memory_config = DEFAULT_MBMEMORY
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["text_retrieval"]["topk"] = _rk
        self.memory_config["recall"]["truncation"]["number"] = 64000
        self._mb_conversations_for_recall: list = []

    def build_memory(
        self,
        conversations: list,
        conversation_streams: Optional[list],
        qa_samples: list,
    ) -> None:
        self._mb_conversations_for_recall = list(conversations) if conversations else []
        super().build_memory(conversations, conversation_streams, qa_samples)

    def _recall_for_question(self, question: str) -> str:
        assert self._memory is not None, "Memory not initialized; call _init_memory first"
        rt = _mb_max_bucketed_wall_time(self._mb_conversations_for_recall)
        if rt is not None:
            result = self._memory.recall({"text": question, "time": rt})
        else:
            result = self._memory.recall(question)
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            items = cast(List[Any], result)
            return "\n".join(str(x) for x in items)
        return str(result)

    def _store_turn_to_memory(self, text: Optional[str], timestamp: Optional[Any] = None) -> None:
        if timestamp is not None:
            timestamp = _coarse_timestamp_10min_bucket(timestamp)
        super()._store_turn_to_memory(text, timestamp)

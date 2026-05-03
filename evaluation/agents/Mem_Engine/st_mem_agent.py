from copy import deepcopy
from datetime import datetime
from typing import Any, Optional

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import STMemory
from default_config.DefaultMemoryConfig import DEFAULT_STMEMORY

def _timestamp_for_time_retrieval(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, bool):
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(s, fmt)
                return float(dt.timestamp())
            except ValueError:
                continue
    return None

class STMemAgent(BaseMemEngineAgent):

    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = STMemory

        self.memory_config = deepcopy(DEFAULT_STMEMORY)
        self.memory_config["store"]["method"] = "STMemoryStore"
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["time_retrieval"]["topk"] = _rk
        self.memory_config["recall"]["truncation"]["number"] = 640000

    def _store_turn_to_memory(self, text: Optional[str], timestamp: Optional[Any] = None) -> None:
        if not text:
            return
        payload: dict[str, Any] = {"text": text}
        tv = _timestamp_for_time_retrieval(timestamp)
        if tv is not None:
            payload["time"] = tv
        assert self._memory is not None
        self._memory.store(payload)

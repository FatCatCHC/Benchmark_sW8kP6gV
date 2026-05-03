import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import SCMemory
from default_config.DefaultMemoryConfig import DEFAULT_SCMEMORY

BGE_LARGE_LOCAL_PATH = os.environ.get("BGE_LARGE_MODEL_PATH", "BAAI/bge-large-en-v1.5")

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

def _apillm_config_for_eval(args) -> dict:
    return {
        "method": "APILLM",
        "name": args.model,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "temperature": args.temperature,
    }

class SCMemAgent(BaseMemEngineAgent):

    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = SCMemory
        self.memory_config = deepcopy(DEFAULT_SCMEMORY)
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["text_retrieval"]["topk"] = _rk

        self.memory_config["recall"]["flash_capacity"] = int(
            getattr(self.args, "sc_mem_flash_capacity", 3)
        )
        self.memory_config["recall"]["activation_topk"] = int(
            getattr(self.args, "sc_mem_activation_topk", 3)
        )
        self.memory_config["recall"]["text_retrieval"]["encoder"]["name"] = "bge-large-en-v1.5"
        self.memory_config["recall"]["text_retrieval"]["encoder"]["dimension"] = 1024
        self.memory_config["recall"]["text_retrieval"]["encoder"]["path"] = BGE_LARGE_LOCAL_PATH
        self.memory_config["recall"]["truncation"]["number"] = 64000

        llm_cfg = _apillm_config_for_eval(args)
        self.memory_config["recall"]["activation_judge"]["LLM_config"] = llm_cfg
        self.memory_config["recall"]["summary_judge"]["LLM_config"] = llm_cfg
        self.memory_config["recall"]["summarizer"]["LLM_config"] = llm_cfg

        gc = self.memory_config.get("global_config")
        if not isinstance(gc, dict):
            gc = {}
            self.memory_config["global_config"] = gc
        gpu = getattr(self.args, "mem_engine_gpu", None)
        gc["usable_gpu"] = str(gpu) if gpu is not None else "0"

        self.counter = 0

    def _store_turn_to_memory(self, text: Optional[str], timestamp: Optional[Any] = None) -> None:
        if not text:
            return
        payload: dict[str, Any] = {"text": text, "summary": text}

        payload["time"] = self.counter
        self.counter += 1
        assert self._memory is not None
        self._memory.store(payload)

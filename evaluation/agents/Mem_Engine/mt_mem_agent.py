import os
from copy import deepcopy

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import MTMemory
from default_config.DefaultMemoryConfig import DEFAULT_MTMEMORY

class MTMemAgent(BaseMemEngineAgent):

    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = MTMemory
        self.memory_config = deepcopy(DEFAULT_MTMEMORY)
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["text_retrieval"]["topk"] = _rk
        self.memory_config["recall"]["truncation"]["number"] = 64000

        self.memory_config["store"]["summarizer"]["LLM_config"] = {
            "method": "APILLM",
            "name": args.model,
            "api_key": args.api_key,
            "base_url": args.base_url,
            "temperature": args.temperature,
        }

        gc = self.memory_config.get("global_config")
        if not isinstance(gc, dict):
            gc = {}
            self.memory_config["global_config"] = gc
        gpu = getattr(self.args, "mem_engine_gpu", None)
        gc["usable_gpu"] = str(gpu) if gpu is not None else "0"

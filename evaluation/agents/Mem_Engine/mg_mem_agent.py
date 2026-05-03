import os
from copy import deepcopy

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import MGMemory
from default_config.DefaultMemoryConfig import DEFAULT_MGMEMORY

def _apillm_config_for_eval(args) -> dict:
    return {
        "method": "APILLM",
        "name": args.model,
        "api_key": os.getenv("MODEL_API_KEY"),
        "base_url": os.getenv("MODEL_API_BASE_URL"),
        "temperature": args.temperature,
    }

class MGMemAgent(BaseMemEngineAgent):

    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = MGMemory
        self.memory_config = deepcopy(DEFAULT_MGMEMORY)
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["recall_retrieval"]["topk"] = _rk
        self.memory_config["recall"]["archival_retrieval"]["topk"] = _rk
        self.memory_config["recall"]["truncation"]["number"] = 64000

        self.memory_config["recall"]["memory_prompt_max_words"] = 4096
        self.memory_config["recall"]["trigger"]["LLM_config"] = _apillm_config_for_eval(args)
        self.memory_config["store"]["summarizer"]["LLM_config"] = _apillm_config_for_eval(args)
        self.memory_config["store"]["flush_checker"]["number"] = 1024
        gc = self.memory_config.get("global_config")
        if not isinstance(gc, dict):
            gc = {}
            self.memory_config["global_config"] = gc
        gpu = getattr(self.args, "mem_engine_gpu", None)
        gc["usable_gpu"] = str(gpu) if gpu is not None else "0"

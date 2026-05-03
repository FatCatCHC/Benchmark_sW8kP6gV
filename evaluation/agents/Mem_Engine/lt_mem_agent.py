from copy import deepcopy

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import LTMemory
from default_config.DefaultMemoryConfig import DEFAULT_LTMEMORY

class LTMemAgent(BaseMemEngineAgent):

    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = LTMemory
        self.memory_config = deepcopy(DEFAULT_LTMEMORY)
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["text_retrieval"]["topk"] = _rk
        self.memory_config["recall"]["truncation"]["number"] = 640000

        gc = self.memory_config.get("global_config")
        if not isinstance(gc, dict):
            gc = {}
            self.memory_config["global_config"] = gc
        gpu = getattr(self.args, "mem_engine_gpu", None)
        gc["usable_gpu"] = str(gpu) if gpu is not None else "0"

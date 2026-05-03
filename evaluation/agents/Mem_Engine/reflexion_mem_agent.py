import sys
import os

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import RFMemory
from default_config.DefaultMemoryConfig import DEFAULT_RFMEMORY

class ReflexionMemAgent(BaseMemEngineAgent):
    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = RFMemory
        self.memory_config = DEFAULT_RFMEMORY
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["top_k"] = _rk

        self.memory_config["recall"]["truncation"]["number"] = 64000

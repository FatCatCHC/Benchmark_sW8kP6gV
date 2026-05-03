import sys
import os

from .base_mem_engine_agent import BaseMemEngineAgent, _ensure_mem_engine_path

_ensure_mem_engine_path()
from memengine import GAMemory
from default_config.DefaultMemoryConfig import DEFAULT_GAMEMORY

class GenAgentMemAgent(BaseMemEngineAgent):
    def __init__(self, client, args):
        super().__init__(client, args)
        self.memory_class = GAMemory
        self.memory_config = DEFAULT_GAMEMORY
        _rk = int(getattr(self.args, "mem_engine_retrieval_num", 10))
        self.memory_config["recall"]["topk"] = _rk

        trunc = self.memory_config["recall"]["truncation"]
        trunc["method"] = "LMTruncation"
        trunc["mode"] = "word"
        trunc["number"] = 64000

        self.memory_config["recall"]["importance_judge"]["LLM_config"] = {
            "method": "APILLM",
            "name": self.args.model,
            "api_key": self.args.api_key,
            "base_url": self.args.base_url,
            "temperature": self.args.temperature,
        }
        print(self.memory_config["recall"]["importance_judge"]["LLM_config"])

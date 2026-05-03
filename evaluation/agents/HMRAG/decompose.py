from __future__ import annotations

import re
from typing import List, Union

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..prompt import (
    HMRAG_DECOMPOSE_COUNT_INTENTS_PROMPT_TEMPLATE,
    HMRAG_DECOMPOSE_SPLIT_QUERY_PROMPT_TEMPLATE,
)

class QueryDecomposer:
    def __init__(self, client, model: str, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(Exception))
    def count_intents(self, query: str) -> int:
        prompt = HMRAG_DECOMPOSE_COUNT_INTENTS_PROMPT_TEMPLATE.format(query=query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=self.temperature,
        )
        raw = (response.choices[0].message.content or "").strip()
        m = re.search(r"-?\d+", raw)
        if not m:
            return 1
        n = int(m.group(0))
        return max(1, min(n, 8))

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(Exception))
    def _split_query(self, query: str) -> List[str]:
        prompt = HMRAG_DECOMPOSE_SPLIT_QUERY_PROMPT_TEMPLATE.format(query=query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=self.temperature,
        )
        raw = (response.choices[0].message.content or "").strip()
        parts = [p.strip() for p in raw.split("||") if p.strip()]
        return parts if parts else [query]

    def decompose(self, query: str) -> Union[str, List[str]]:
        intent_count = self.count_intents(query)
        intent_count = min(intent_count, 3)
        if intent_count > 1:
            return self._split_query(query)
        return query

    def decompose_to_search_string(self, query: str) -> str:
        out = self.decompose(query)
        if isinstance(out, list):
            return " ".join(out)
        return out

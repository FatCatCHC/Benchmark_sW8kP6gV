from __future__ import annotations

import json
import os
from typing import Optional

import urllib.error
import urllib.request

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..prompt import HMRAG_WEB_SUMMARY_PROMPT_TEMPLATE

def _format_serper_payload(data: dict, max_organic: int = 5) -> str:
    lines = []
    if "answerBox" in data and isinstance(data["answerBox"], dict):
        ab = data["answerBox"]
        ans = ab.get("answer") or ab.get("snippet") or ""
        if ans:
            lines.append(f"Direct answer: {ans}")
    organic = data.get("organic") or []
    for item in organic[:max_organic]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        lines.append(f"[{title}]\n{snippet}\n{link}")
    return "\n\n".join(lines) if lines else "No relevant results found."

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(Exception))
def serper_search(query: str, api_key: str, num: int = 5) -> dict:
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=json.dumps({"q": query, "num": num}).encode("utf-8"),
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

class WebRetrieval:
    def __init__(self, client, summarize_model: str, api_key: Optional[str], num_results: int = 5):
        self.client = client
        self.summarize_model = summarize_model
        self.api_key = (api_key or os.environ.get("SERPER_API_KEY") or "").strip()
        self.num_results = num_results

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(Exception))
    def _summarize_snippets(self, formatted: str, query: str) -> str:
        prompt = HMRAG_WEB_SUMMARY_PROMPT_TEMPLATE.format(query=query, snippets=formatted)
        response = self.client.chat.completions.create(
            model=self.summarize_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def find_top_k(self, query: str) -> str:
        if not self.api_key:
            return "(Web retrieval skipped: no Serper API key. Set SERPER_API_KEY or --hmrag_serper_api_key.)"
        try:
            data = serper_search(query, self.api_key, num=self.num_results)
            formatted = _format_serper_payload(data)
            return self._summarize_snippets(formatted, query)
        except urllib.error.HTTPError as e:
            return f"(Web retrieval failed: HTTP {e.code})"
        except Exception as e:
            return f"(Web retrieval failed: {e})"

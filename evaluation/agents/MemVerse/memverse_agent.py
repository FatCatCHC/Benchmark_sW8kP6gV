from __future__ import annotations

import os
from typing import Any, Optional

import requests
from tqdm import tqdm

from ..base_mem_agent import BaseMemoryAgent
from ..prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE,
    MEMORY_INGEST_FINAL_MESSAGE,
    MEMVERSE_FUNCTION_CALL_CONTEXT_NOTE,
    MULTIPLE_CHOICE_OUTPUT_FORMAT,
    OVERALL_EVALUATION_SYSTEM_PROMPT,
    QUESTION_WITH_OPTIONS_TEMPLATE,
)
from utils import (
    build_function_call_prompt_text,
    evaluate_function_call_response,
    load_function_call_candidate_tools,
)

MEMVERSE_AGENT_LABEL = "MemVerse"


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _join_nonempty(parts: list[str], sep: str = "\n") -> str:
    return sep.join(part for part in parts if part and part.strip())


class MemVerseAgent(BaseMemoryAgent):

    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.args = args
        self.eval_format = getattr(args, "eval_format", "multiple_choice")

        self.base_url = (
            _safe_text(getattr(args, "memverse_api_url", "")).strip()
            or os.getenv("MEMVERSE_API_URL", "http://127.0.0.1:8000").strip()
        ).rstrip("/")
        self.insert_timeout_s = max(1.0, float(getattr(args, "memverse_insert_timeout_s", 1800)))
        self.query_timeout_s = max(1.0, float(getattr(args, "memverse_query_timeout_s", 180)))
        self.chunk_turns = max(1, int(getattr(args, "memverse_chunk_turns", 6)))
        self.chunk_max_chars = max(1, int(getattr(args, "memverse_chunk_max_chars", 12000)))
        self.query_mode = _safe_text(getattr(args, "memverse_query_mode", "hybrid")).strip() or "hybrid"
        self.use_pm = bool(getattr(args, "memverse_use_pm", False))

        self._session = requests.Session()
        self._memory_built = False
        self._ingested_turns = 0
        self._ingested_chunks = 0
        self._function_call_candidate_tools = load_function_call_candidate_tools()

    def _qa_system_prompt(self) -> str:
        system_prompt = OVERALL_EVALUATION_SYSTEM_PROMPT
        if self.eval_format == "multiple_choice":
            system_prompt += MULTIPLE_CHOICE_OUTPUT_FORMAT
        elif self.eval_format == "fill_in_the_blank":
            system_prompt += FILL_IN_THE_BLANK_OUTPUT_FORMAT
        else:
            raise ValueError(f"Invalid evaluation format: {self.eval_format}")
        return system_prompt

    def _build_turn_header(self, turn: dict, *, label: str) -> str:
        sender = _safe_text(turn.get("sender_name", "unknown")).strip() or "unknown"
        timestamp = _safe_text(turn.get("timestamp", "")).strip()
        conversation_name = _safe_text(turn.get("conversation_name", "")).strip()
        fields = [f"speaker={sender}", label]
        if timestamp:
            fields.insert(1, f"time={timestamp}")
        if conversation_name:
            fields.append(f"conversation={conversation_name}")
        return "[" + " | ".join(fields) + "]"

    def _turn_to_memverse_text(self, turn: dict) -> str:
        content_type = _safe_text(turn.get("content_type", "text")).strip() or "text"

        if content_type == "text":
            content = _safe_text(turn.get("content", "")).strip()
            if not content:
                return ""
            return f"{self._build_turn_header(turn, label='text')}\n{content}"

        if content_type == "image":
            header = self._build_turn_header(turn, label="image")
            cap = _safe_text(turn.get("caption", "")).strip()
            if cap:
                return f"{header}\n[caption]\n{cap}"
            return f"{header}\n[image; no caption]"

        if content_type == "json_evidence":
            header = self._build_turn_header(turn, label="json_evidence")
            doc_caption = _safe_text(turn.get("caption", "")).strip()
            if doc_caption:
                return f"{header}\n[caption]\n{doc_caption}"
            return f"{header}\n[document; no caption]"

        return ""

    def _build_ingest_chunks(self, conversations: list[dict]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current_lines: list[str] = []
        current_turns = 0
        current_chars = 0

        def flush() -> None:
            nonlocal current_lines, current_turns, current_chars
            chunk_text = _join_nonempty(current_lines, sep="\n\n")
            if chunk_text:
                chunks.append(
                    {
                        "query": chunk_text,
                        "turn_count": current_turns,
                        "char_count": current_chars,
                    }
                )
            current_lines = []
            current_turns = 0
            current_chars = 0

        for turn in conversations:
            line = self._turn_to_memverse_text(turn)
            if not line:
                continue
            line_len = len(line)
            should_flush = False
            if current_lines:
                if current_turns >= self.chunk_turns:
                    should_flush = True
                elif current_chars + line_len > self.chunk_max_chars:
                    should_flush = True
            if should_flush:
                flush()
            current_lines.append(line)
            current_turns += 1
            current_chars += line_len

        flush()
        return chunks

    def _post_insert(self, query: str) -> dict[str, Any]:
        response = self._session.post(
            f"{self.base_url}/insert",
            data={"query": query},
            timeout=self.insert_timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def _post_query(self, query: str, retrieval_query: Optional[str] = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "query": query,
            "mode": self.query_mode,
            "use_pm": str(self.use_pm).lower(),
        }
        stem = (retrieval_query or "").strip()
        if stem:
            data["retrieval_query"] = stem
        response = self._session.post(
            f"{self.base_url}/query",
            data=data,
            timeout=self.query_timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def _post_clear_memory(self) -> dict[str, Any]:
        response = self._session.post(
            f"{self.base_url}/clear_memory",
            timeout=max(60.0, min(600.0, self.insert_timeout_s)),
        )
        response.raise_for_status()
        return response.json()

    def build_memory(
        self,
        conversations: list,
        conversation_streams: Optional[list],
        qa_samples: list[dict],
    ) -> None:
        del conversation_streams, qa_samples
        self._post_clear_memory()
        chunks = self._build_ingest_chunks(conversations)
        self._memory_built = False
        self._ingested_turns = 0
        self._ingested_chunks = 0
        total = len(chunks)
        for idx, chunk in enumerate(tqdm(chunks, desc="Building MemVerse memory"), start=1):
            prefix = MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE.format(
                idx=idx, total=total, agent_label=MEMVERSE_AGENT_LABEL
            ) + "\n"
            self._post_insert(prefix + chunk["query"])
            self._ingested_turns += int(chunk["turn_count"])
            self._ingested_chunks += 1
        if chunks:
            self._post_insert(MEMORY_INGEST_FINAL_MESSAGE)
        self._memory_built = True

    def _question_with_options(self, qa_sample: dict) -> str:
        question = _safe_text(qa_sample.get("question", "")).strip()
        if self.eval_format != "multiple_choice":
            return question
        mc = qa_sample.get("multi_choice_QA") or {}
        options = mc.get("multi_choice_QA_options") or []
        if not options:
            return question
        labels = ["(A)", "(B)", "(C)", "(D)"]
        lines = [f"{labels[i]}: {opt}" for i, opt in enumerate(options[:4])]
        return QUESTION_WITH_OPTIONS_TEMPLATE.format(
            question=question,
            candidate_options="\n".join(lines),
        )

    @staticmethod
    def _qa_retrieval_stem(qa_sample: dict) -> str:
        return _safe_text(qa_sample.get("question", "")).strip()

    @staticmethod
    def _extract_answer_text(query_payload: Any) -> str:
        if isinstance(query_payload, dict):
            for key in ("final_answer", "answer", "response"):
                value = _safe_text(query_payload.get(key, "")).strip()
                if value:
                    return value
        return _safe_text(query_payload).strip()

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: Optional[list] = None,
    ):
        del conversations, conversation_streams
        if not self._memory_built:
            raise RuntimeError("MemVerse memory not built; run evaluate_cluster first.")

        parsed_function_calls = None
        answer_messages: list[dict[str, Any]] = []
        query_payload: Any = None

        stem = self._qa_retrieval_stem(qa_sample)

        if qa_sample.get("category") == "Function_Call":
            full_query = build_function_call_prompt_text(
                qa_sample.get("question", ""),
                MEMVERSE_FUNCTION_CALL_CONTEXT_NOTE,
                self._function_call_candidate_tools,
            )
            answer_messages = [{"role": "user", "content": full_query}]
            query_payload = self._post_query(full_query, retrieval_query=stem)
            raw_response = self._extract_answer_text(query_payload)
            single_qa_result, parsed_function_calls = evaluate_function_call_response(
                raw_response,
                qa_sample.get("answer"),
            )
        else:
            system_prompt = self._qa_system_prompt()
            user_block = self._question_with_options(qa_sample)
            full_query = f"{system_prompt}\n\nQuestion:\n{user_block}"
            answer_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question:\n{user_block}"},
            ]
            query_payload = self._post_query(full_query, retrieval_query=stem)
            raw_response = self._extract_answer_text(query_payload)
            single_qa_result = self._evaluate_answer(raw_response, qa_sample)

        ground_truth = qa_sample.get("answer")
        if (
            self.eval_format == "multiple_choice"
            and "multi_choice_QA" in qa_sample
            and qa_sample.get("category") != "Function_Call"
        ):
            ground_truth = ["(A)", "(B)", "(C)", "(D)"][
                qa_sample["multi_choice_QA"]["multi_choice_QA_answer"]
            ]

        readable_answer = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": qa_sample.get("question"),
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": raw_response,
            "parsed_function_calls": parsed_function_calls,
            "single_qa_result": single_qa_result,
        }
        read_message = {
            **readable_answer,
            "messages": answer_messages,
            "recall_context": None,
            "memverse_query_payload": query_payload,
            "memverse_api_url": self.base_url,
            "memverse_query_mode": self.query_mode,
            "memverse_retrieval_query": stem or None,
            "memverse_use_pm": self.use_pm,
            "ingested_turn_count": self._ingested_turns,
            "ingested_chunk_count": self._ingested_chunks,
            "chunk_turns": self.chunk_turns,
            "chunk_max_chars": self.chunk_max_chars,
        }
        return readable_answer, read_message

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from ..base_mem_agent import BaseMemoryAgent
from ..prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    MEM0_USER_PROMPT_TEMPLATE,
    MEMORY_CAPTION_ONLY_QUERY_IMAGES_PREFIX,
    MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE,
    MEMORY_INGEST_FINAL_MESSAGE,
    MULTIPLE_CHOICE_OUTPUT_FORMAT,
    OVERALL_EVALUATION_SYSTEM_PROMPT,
    QUESTION_WITH_OPTIONS_TEMPLATE,
)
from constant import Image_Root_Dir_Path
from utils import (
    build_function_call_messages,
    evaluate_function_call_response,
    get_response,
    image_path_to_data_url,
    load_function_call_candidate_tools,
)

MEM0_AGENT_LABEL = "Mem0"


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _join_nonempty(parts: list[str], sep: str = "\n") -> str:
    return sep.join(part for part in parts if part and part.strip())


def _extract_answer_text(raw_response: Any) -> str:
    if isinstance(raw_response, tuple):
        raw_response = raw_response[0]
    return _safe_text(raw_response).strip()


def _resolve_memory_modal_mode(args: Any) -> str:
    override = _safe_text(getattr(args, "mem0_memory_modal_mode", "")).strip().lower()
    if override in ("original", "caption"):
        return override
    fallback = _safe_text(getattr(args, "mem0_ingest_media_mode", "original")).strip().lower() or "original"
    if fallback not in ("original", "caption"):
        raise ValueError(f"mem0 memory modal mode must be 'original' or 'caption', got {fallback!r}")
    return fallback


class Mem0Agent(BaseMemoryAgent):

    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.args = args
        self.eval_format = getattr(args, "eval_format", "multiple_choice")
        self.dataset_dir = Path(args.dataset_dir_path).resolve()
        self.memory_modal_mode = _resolve_memory_modal_mode(args)

        self.chunk_turns = max(1, int(getattr(args, "mem0_chunk_turns", 6)))
        self.chunk_max_chars = max(1, int(getattr(args, "mem0_chunk_max_chars", 12000)))
        self.chunk_max_images = max(1, int(getattr(args, "mem0_chunk_max_images", 8)))
        self.retrieve_limit = max(1, int(getattr(args, "mem0_retrieve_limit", 8)))
        self.mem0_collection = (
            _safe_text(getattr(args, "mem0_collection_name", "")).strip()
            or f"mem0_eval_{(_safe_text(getattr(args, 'save_dir_name', '')).strip() or 'cluster')}"
        )
        self.mem0_qdrant_path = (
            _safe_text(getattr(args, "mem0_qdrant_path", "")).strip()
            or str((Path(getattr(args, "checkpoint_dir", ".")).resolve() / "mem0_qdrant"))
        )

        base_user_id = _safe_text(getattr(args, "mem0_user_id", "")).strip()
        if not base_user_id:
            cluster_name = _safe_text(getattr(args, "save_dir_name", "")).strip() or self.dataset_dir.name or "cluster"
            base_user_id = f"mem0-eval-{cluster_name}-{uuid.uuid4().hex[:10]}"
        self.user_id = base_user_id
        self.run_id = f"mem0-run-{uuid.uuid4().hex[:8]}"

        self._memory = None
        self._memory_built = False
        self._ingested_turns = 0
        self._ingested_chunks = 0
        self._function_call_candidate_tools = load_function_call_candidate_tools()

    def _bootstrap_mem0(self):
        if self._memory is not None:
            return self._memory
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise ImportError("Install mem0 first: pip install mem0ai") from exc

        api_key = _safe_text(getattr(self.args, "api_key", "")).strip()
        api_base = _safe_text(getattr(self.args, "base_url", "")).strip()
        if not api_key or not api_base:
            raise RuntimeError("Mem0 agent requires args.api_key and args.base_url for OpenAI-compatible backend.")

        os.environ["MODEL_API_KEY"] = api_key
        os.environ["MODEL_API_BASE_URL"] = api_base

        llm_model = _safe_text(getattr(self.args, "model", "qwen3-vl-plus")).strip() or "qwen3-vl-plus"
        embedding_model = _safe_text(getattr(self.args, "mem0_embedding_model", "text-embedding-v4")).strip()
        embedding_dim = int(getattr(self.args, "mem0_embedding_dim", 1024))
        qdrant_path = Path(self.mem0_qdrant_path).resolve()
        qdrant_path.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": api_key,
                    "openai_base_url": api_base.rstrip("/"),
                    "model": llm_model,
                    "temperature": float(getattr(self.args, "temperature", 0.01)),
                    "enable_vision": True,
                    "vision_details": "auto",
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": api_key,
                    "openai_base_url": api_base.rstrip("/"),
                    "model": embedding_model,
                    "embedding_dims": embedding_dim,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.mem0_collection,
                    "path": str(qdrant_path),
                    "embedding_model_dims": embedding_dim,
                },
            },
        }
        self._memory = Memory.from_config(config)
        return self._memory

    def _reset_mem0_memory(self) -> None:
        if self._memory is None:
            return
        try:
            self._memory.reset()
            return
        except Exception:
            try:
                self._memory.delete_all(user_id=self.user_id)
                return
            except Exception:
                pass

        try:
            qdrant_path = Path(self.mem0_qdrant_path).resolve()
            if qdrant_path.exists():
                shutil.rmtree(qdrant_path)
        except Exception:
            pass
        self._memory = None
        self._bootstrap_mem0()

    def _resolve_image_to_data_url(self, raw: str) -> str:
        ref = _safe_text(raw).strip()
        if not ref:
            return ""
        if ref.startswith("data:"):
            return ref
        if ref.startswith(("http://", "https://")):
            return ref
        if ref.startswith("file://"):
            ref = ref[7:]

        path = Path(ref)
        if path.is_file():
            return image_path_to_data_url(str(path.resolve()))

        candidate = (self.dataset_dir / ref).resolve()
        if candidate.is_file():
            return image_path_to_data_url(str(candidate))

        rooted = Path(Image_Root_Dir_Path) / ref.lstrip("/\\")
        if rooted.is_file():
            return image_path_to_data_url(str(rooted.resolve()))
        return ""

    def _message_char_and_image_count(self, message: dict[str, Any]) -> tuple[int, int]:
        chars = 0
        images = 0
        content = message.get("content")
        if isinstance(content, str):
            return len(content), 0
        if isinstance(content, dict):
            content = [content]
        for item in content or []:
            if not isinstance(item, dict):
                chars += len(_safe_text(item))
                continue
            if item.get("type") == "text":
                chars += len(_safe_text(item.get("text", "")))
            elif item.get("type") == "image_url":
                images += 1
        return chars, images

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

    def _turn_to_mem0_message(self, turn: dict) -> Optional[dict[str, Any]]:
        content_type = _safe_text(turn.get("content_type", "text")).strip() or "text"

        if content_type == "text":
            text = _safe_text(turn.get("content", "")).strip()
            if not text:
                return None
            rendered = f"{self._build_turn_header(turn, label='text')}\n{text}"
            if self.memory_modal_mode == "caption":
                return {"role": "user", "content": rendered}
            return {
                "role": "user",
                "content": [{"type": "text", "text": rendered}],
            }

        if content_type == "image":
            header = self._build_turn_header(turn, label="image")
            if self.memory_modal_mode == "caption":
                caption = _safe_text(turn.get("caption", "")).strip()
                text = f"{header}\n[caption]\n{caption}" if caption else f"{header}\n[image; no caption]"
                return {"role": "user", "content": text}

            content: list[dict[str, Any]] = [{"type": "text", "text": header}]
            image_path = _safe_text(turn.get("image_path", "")).strip()
            image_url = self._resolve_image_to_data_url(image_path)
            if image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            caption = _safe_text(turn.get("caption", "")).strip()
            if caption:
                content.append({"type": "text", "text": f"[caption]\n{caption}"})
            return {"role": "user", "content": content}

        if content_type == "json_evidence":
            header = self._build_turn_header(turn, label="json_evidence")
            top_caption = _safe_text(turn.get("caption", "")).strip()

            if self.memory_modal_mode == "caption":
                text = f"{header}\n[caption]\n{top_caption}" if top_caption else f"{header}\n[document; no caption]"
                return {"role": "user", "content": text}

            content: list[dict[str, Any]] = [{"type": "text", "text": header}]
            if top_caption:
                content.append({"type": "text", "text": f"[caption]\n{top_caption}"})
            for item in turn.get("content") or []:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                inner = item.get("content") or {}
                if not isinstance(inner, dict):
                    inner = {}
                if item_type == "text":
                    text = _safe_text(inner.get("text", "")).strip()
                    if text:
                        content.append({"type": "text", "text": text})
                elif item_type == "image":
                    image_path = _safe_text(inner.get("image_path", "")).strip()
                    image_url = self._resolve_image_to_data_url(image_path)
                    if image_url:
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
            return {"role": "user", "content": content}

        return None

    def _build_ingest_chunks(self, conversations: list[dict]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current_messages: list[dict[str, Any]] = []
        current_turns = 0
        current_chars = 0
        current_images = 0

        def flush() -> None:
            nonlocal current_messages, current_turns, current_chars, current_images
            if current_messages:
                chunks.append(
                    {
                        "messages": list(current_messages),
                        "turn_count": current_turns,
                        "char_count": current_chars,
                        "image_count": current_images,
                    }
                )
            current_messages = []
            current_turns = 0
            current_chars = 0
            current_images = 0

        for turn in conversations:
            message = self._turn_to_mem0_message(turn)
            if message is None:
                continue
            msg_chars, msg_images = self._message_char_and_image_count(message)
            should_flush = False
            if current_messages:
                if current_turns >= self.chunk_turns:
                    should_flush = True
                elif current_chars + msg_chars > self.chunk_max_chars:
                    should_flush = True
                elif current_images + msg_images > self.chunk_max_images:
                    should_flush = True
            if should_flush:
                flush()

            current_messages.append(message)
            current_turns += 1
            current_chars += msg_chars
            current_images += msg_images

        flush()
        return chunks

    def _mem0_add(self, messages: list[dict[str, Any]]) -> Any:
        if self._memory is None:
            raise RuntimeError("Mem0 memory backend is not initialized.")
        try:
            return self._memory.add(
                messages,
                user_id=self.user_id,
                run_id=self.run_id,
                infer=False,
            )
        except TypeError:
            return self._memory.add(messages, user_id=self.user_id)

    def _mem0_search(self, query: str) -> Any:
        if self._memory is None:
            raise RuntimeError("Mem0 memory backend is not initialized.")
        try:
            return self._memory.search(
                query=query,
                filters={"user_id": self.user_id},
                top_k=self.retrieve_limit,
            )
        except TypeError:
            pass
        try:
            return self._memory.search(
                query,
                user_id=self.user_id,
                run_id=self.run_id,
                limit=self.retrieve_limit,
            )
        except TypeError:
            return self._memory.search(query, user_id=self.user_id, limit=self.retrieve_limit)

    def build_memory(
        self,
        conversations: list,
        conversation_streams: Optional[list],
        qa_samples: list[dict],
    ) -> None:
        del conversation_streams, qa_samples
        self._bootstrap_mem0()
        self.run_id = f"mem0-run-{uuid.uuid4().hex[:8]}"
        self._reset_mem0_memory()

        self._memory_built = False
        self._ingested_turns = 0
        self._ingested_chunks = 0
        chunks = self._build_ingest_chunks(conversations)
        total = len(chunks)

        for idx, chunk in enumerate(tqdm(chunks, desc="Building Mem0 memory", mininterval=0.5), start=1):
            msgs = list(chunk["messages"])
            prefix = MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE.format(
                idx=idx, total=total, agent_label=MEM0_AGENT_LABEL
            )
            if msgs:
                first = msgs[0]
                first_content = first.get("content")
                if isinstance(first_content, str):
                    msgs[0] = {**first, "content": prefix + first_content}
                elif isinstance(first_content, list):
                    first_parts = list(first_content)
                    if first_parts and isinstance(first_parts[0], dict) and first_parts[0].get("type") == "text":
                        first_parts[0] = {
                            **first_parts[0],
                            "text": prefix + _safe_text(first_parts[0].get("text", "")),
                        }
                        msgs[0] = {**first, "content": first_parts}
                    else:
                        msgs[0] = {**first, "content": [{"type": "text", "text": prefix}] + first_parts}
                else:
                    msgs[0] = {**first, "content": prefix + _safe_text(first_content)}
            self._mem0_add(msgs)
            self._ingested_turns += int(chunk["turn_count"])
            self._ingested_chunks += 1

        if chunks:
            self._mem0_add([{"role": "user", "content": MEMORY_INGEST_FINAL_MESSAGE}])
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
    def _qa_image_refs(qa_sample: dict) -> list[str]:
        top = qa_sample.get("images")
        if isinstance(top, list) and top:
            return [_safe_text(x).strip() for x in top if _safe_text(x).strip()]
        evidence = qa_sample.get("evidence")
        if isinstance(evidence, dict):
            imgs = evidence.get("image_evidence")
            if isinstance(imgs, list) and imgs:
                return [_safe_text(x).strip() for x in imgs if _safe_text(x).strip()]
        return []

    def _memory_item_to_text(self, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, (int, float, bool)):
            return str(item)
        if isinstance(item, list):
            return _join_nonempty([self._memory_item_to_text(x) for x in item], sep="\n\n")
        if isinstance(item, dict):
            parts: list[str] = []
            for key in ("memory", "text", "summary", "value", "answer", "snippet", "context", "results", "items", "data"):
                value = item.get(key)
                if value is None:
                    continue
                rendered = self._memory_item_to_text(value)
                if rendered:
                    parts.append(rendered)
            content = item.get("content")
            if isinstance(content, list):
                rendered_content: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = _safe_text(part.get("text", "")).strip()
                        if text:
                            rendered_content.append(text)
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        rendered_content.append("[image evidence]")
                    else:
                        text = self._memory_item_to_text(part)
                        if text:
                            rendered_content.append(text)
                if rendered_content:
                    parts.append(_join_nonempty(rendered_content, sep="\n"))
            return _join_nonempty(parts, sep="\n")
        return str(item)

    def _retrieve_context(self, qa_sample: dict) -> tuple[str, Any]:
        question = _safe_text(qa_sample.get("question", "")).strip()
        refs = self._qa_image_refs(qa_sample)
        if self.memory_modal_mode == "caption" and refs:
            question = (
                question
                + MEMORY_CAPTION_ONLY_QUERY_IMAGES_PREFIX
                + "\n".join(f"- [image evidence] {r}" for r in refs)
            )
        payload = self._mem0_search(question)
        context = self._memory_item_to_text(payload).strip()
        return context, payload

    def _answer_question_with_context(
        self,
        qa_sample: dict,
        context: str,
    ) -> tuple[str, list[dict[str, Any]], int]:
        system_prompt = OVERALL_EVALUATION_SYSTEM_PROMPT
        if self.eval_format == "multiple_choice":
            system_prompt += MULTIPLE_CHOICE_OUTPUT_FORMAT
        elif self.eval_format == "fill_in_the_blank":
            system_prompt += FILL_IN_THE_BLANK_OUTPUT_FORMAT
        else:
            raise ValueError(f"Invalid evaluation format: {self.eval_format}")

        question_text = self._question_with_options(qa_sample)
        user_text = MEM0_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question_text,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        query_image_count = 0
        refs = self._qa_image_refs(qa_sample)

        if self.memory_modal_mode == "caption":
            if refs:
                content[0]["text"] += (
                    MEMORY_CAPTION_ONLY_QUERY_IMAGES_PREFIX
                    + "\n".join(f"- [image evidence] {r}" for r in refs)
                )
        else:
            for raw in refs:
                data_url = self._resolve_image_to_data_url(raw)
                if not data_url:
                    continue
                content.append({"type": "image_url", "image_url": {"url": data_url}})
                query_image_count += 1

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        raw_response = get_response(self.client, messages, self.args)
        return _extract_answer_text(raw_response), messages, query_image_count

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: Optional[list] = None,
    ):
        del conversations, conversation_streams
        if not self._memory_built or self._memory is None:
            raise RuntimeError("Mem0 memory is not initialized. build_memory must run before QA evaluation.")

        parsed_function_calls = None
        answer_messages: list[dict[str, Any]] = []
        context, search_payload = self._retrieve_context(qa_sample)
        raw_response = ""
        answer_query_image_count = 0

        if qa_sample.get("category") == "Function_Call":
            answer_messages = build_function_call_messages(
                qa_sample.get("question", ""),
                context,
                self._function_call_candidate_tools,
            )
            raw = get_response(self.client, answer_messages, self.args)
            raw_response = _extract_answer_text(raw)
            single_qa_result, parsed_function_calls = evaluate_function_call_response(
                raw_response,
                qa_sample.get("answer"),
            )
        else:
            raw_response, answer_messages, answer_query_image_count = self._answer_question_with_context(
                qa_sample,
                context,
            )
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
            "messages": {
                "answer_messages": answer_messages,
                "retrieved_context": context,
                "mem0_search_payload": search_payload,
                "mem0_user_id": self.user_id,
                "mem0_run_id": self.run_id,
                "mem0_collection_name": self.mem0_collection,
                "mem0_qdrant_path": self.mem0_qdrant_path,
                "ingested_turn_count": self._ingested_turns,
                "ingested_chunk_count": self._ingested_chunks,
                "chunk_turns": self.chunk_turns,
                "chunk_max_chars": self.chunk_max_chars,
                "chunk_max_images": self.chunk_max_images,
                "retrieve_limit": self.retrieve_limit,
                "answer_query_image_count": answer_query_image_count,
                "memory_modal_mode": self.memory_modal_mode,
            },
        }
        return readable_answer, read_message

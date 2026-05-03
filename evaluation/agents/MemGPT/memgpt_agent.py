from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm

from ..base_mem_agent import BaseMemoryAgent
from ..prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    MEMGPT_ARCHIVAL_HUMAN_BLOCK,
    MEMGPT_ARCHIVAL_PERSONA_BLOCK,
    MEMGPT_ARCHIVAL_QA_BRIDGE_TEMPLATE,
    MEMGPT_ARCHIVAL_QA_INTRO,
    MEMGPT_IMAGE_REF_MARKER_TAG,
    MEMGPT_MESSAGES_HUMAN_BLOCK,
    MEMGPT_MESSAGES_PERSONA_BLOCK,
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
    guess_mime_type,
    image_path_to_data_url,
    load_function_call_candidate_tools,
)

MEMGPT_AGENT_LABEL = "MemGPT"
_IMAGE_REF_RE = re.compile(rf"\[{re.escape(MEMGPT_IMAGE_REF_MARKER_TAG)}:([^\]]+)\]")


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _data_url_to_letta_image_part(data_url: str) -> Optional[dict[str, Any]]:
    ref = _safe_text(data_url).strip()
    if not ref.startswith("data:"):
        return None
    try:
        header, b64 = ref.split(",", 1)
    except ValueError:
        return None
    hl = header.lower()
    if "image/jpeg" in hl or "image/jpg" in hl:
        media_type = "image/jpeg"
    elif "image/webp" in hl:
        media_type = "image/webp"
    elif "image/gif" in hl:
        media_type = "image/gif"
    else:
        media_type = "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64},
    }


def _path_to_letta_image_part(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    mime = guess_mime_type(str(path))
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": b64},
    }


def _letta_image_part_from_ref(ref: str, dataset_dir: Path) -> Optional[dict[str, Any]]:
    ref = _safe_text(ref).strip()
    if not ref:
        return None
    if ref.startswith("data:"):
        return _data_url_to_letta_image_part(ref)
    if ref.startswith(("http://", "https://")):
        return {
            "type": "image",
            "source": {"type": "url", "url": ref},
        }
    p = Path(ref)
    if not p.is_file():
        p = (dataset_dir / ref).resolve()
    return _path_to_letta_image_part(p)


def _letta_message_size(msg: dict[str, Any]) -> tuple[int, int]:
    chars = 0
    images = 0
    content = msg.get("content")
    if isinstance(content, str):
        return len(content), 0
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            chars += len(_safe_text(part.get("text", "")))
        elif part.get("type") == "image":
            images += 1
            raw = _safe_text((part.get("source") or {}).get("data", ""))
            chars += min(len(raw), 10_000)
    return chars, images


def _letta_message_to_plaintext(msg: dict[str, Any]) -> str:
    role = _safe_text(msg.get("role", "user")).strip() or "user"
    content = msg.get("content")
    lines: list[str] = []
    if isinstance(content, str):
        t = content.strip()
        if t:
            lines.append(t)
    else:
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                t = _safe_text(part.get("text", "")).strip()
                if t:
                    lines.append(t)
    body = "\n".join(lines).strip()
    if not body:
        return ""
    return f"{role.upper()}:\n{body}"


def _split_archival_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    max_chars = max(1024, int(max_chars))
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _extract_image_refs_ordered(text: str, max_refs: int) -> list[str]:
    if max_refs <= 0 or not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _IMAGE_REF_RE.finditer(text):
        ref = m.group(1).strip()
        if not ref or ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
        if len(out) >= max_refs:
            break
    return out


def _resolve_memory_modal_mode(args: Any) -> str:
    override = _safe_text(getattr(args, "memgpt_memory_modal_mode", "")).strip().lower()
    if override in ("original", "caption"):
        return override
    fallback = _safe_text(getattr(args, "memgpt_ingest_media_mode", "original")).strip().lower() or "original"
    if fallback not in ("original", "caption"):
        raise ValueError(f"memory_modal_mode must be 'original' or 'caption', got {fallback!r}")
    return fallback


class MemGPTAgent(BaseMemoryAgent):

    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.args = args
        self.eval_format = getattr(args, "eval_format", "multiple_choice")
        self.dataset_dir = Path(args.dataset_dir_path).resolve()

        self.chunk_turns = max(1, int(getattr(args, "memgpt_chunk_turns", 6)))
        self.chunk_max_chars = max(1, int(getattr(args, "memgpt_chunk_max_chars", 12000)))
        self.chunk_max_images = max(1, int(getattr(args, "memgpt_chunk_max_images", 8)))
        self.memory_modal_mode = _resolve_memory_modal_mode(args)

        self.server_url = (
            _safe_text(getattr(args, "memgpt_server_url", "")).strip()
            or os.getenv("MEMGPT_SERVER_URL", "").strip()
            or os.getenv("LETTA_SERVER_URL", "").strip()
            or "http://localhost:8283"
        ).rstrip("/")

        self.context_window = max(4096, int(getattr(args, "memgpt_context_window", 16000)))
        self.embedding_model = (
            _safe_text(getattr(args, "memgpt_embedding_model", "")).strip()
            or "text-embedding-3-small"
        )
        self.embedding_dim = int(getattr(args, "memgpt_embedding_dim", 1536))

        ingest_mode = _safe_text(getattr(args, "memgpt_ingest_mode", "archival")).strip().lower() or "archival"
        if ingest_mode not in ("archival", "messages"):
            raise ValueError("memgpt_ingest_mode must be 'archival' or 'messages'")
        self.ingest_mode = ingest_mode
        self.archival_passage_max_chars = max(1024, int(getattr(args, "memgpt_archival_passage_max_chars", 12000)))
        self.archival_top_k = max(1, int(getattr(args, "memgpt_archival_top_k", 12)))
        self.archival_ingest_tag = (
            _safe_text(getattr(args, "memgpt_archival_tag", "memgpt_eval")).strip() or "memgpt_eval"
        )
        self.archival_qa_max_images = max(0, int(getattr(args, "memgpt_archival_qa_max_images", 8)))

        self._letta = None
        self._agent_id: Optional[str] = None
        self._memory_built = False
        self._function_call_candidate_tools = load_function_call_candidate_tools()
        self._fc_transcript_lines: list[str] = []

    def _bootstrap_letta(self):
        if self._letta is not None:
            return self._letta
        try:
            from letta_client import Letta
        except ImportError as exc:
            raise ImportError(
                "Install letta_client for MemGPT eval: pip install letta-client"
            ) from exc

        pw = (
            _safe_text(getattr(self.args, "memgpt_server_password", "")).strip()
            or os.getenv("MEMGPT_SERVER_PASSWORD", "").strip()
            or os.getenv("LETTA_SERVER_PASSWORD", "").strip()
        )
        if pw:
            self._letta = Letta(base_url=self.server_url, api_key=pw)
        else:
            self._letta = Letta(base_url=self.server_url)
        return self._letta

    def _llm_model_name(self) -> str:
        return _safe_text(getattr(self.args, "model", "")).strip() or "gpt-4o-mini"

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

    def _turn_to_letta_message(self, turn: dict) -> Optional[dict[str, Any]]:
        content_type = _safe_text(turn.get("content_type", "text")).strip() or "text"

        if content_type == "text":
            text = _safe_text(turn.get("content", "")).strip()
            if not text:
                return None
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{self._build_turn_header(turn, label='text')}\n{text}"}
                ],
            }

        if content_type == "image":
            header = self._build_turn_header(turn, label="image")
            if self.memory_modal_mode == "caption":
                cap = _safe_text(turn.get("caption", "")).strip()
                text = f"{header}\n[caption]\n{cap}" if cap else f"{header}\n[image; no caption]"
                return {"role": "user", "content": [{"type": "text", "text": text}]}

            image_path = _safe_text(turn.get("image_path", "")).strip()
            ref_line = f"\n[{MEMGPT_IMAGE_REF_MARKER_TAG}:{image_path}]" if image_path else ""
            content: list[dict[str, Any]] = [{"type": "text", "text": f"{header}{ref_line}"}]

            abs_image_path = os.path.join(Image_Root_Dir_Path, image_path)
            image_url = image_path_to_data_url(abs_image_path)
            content.append(_letta_image_part_from_ref(image_url, self.dataset_dir))
            return {"role": "user", "content": content}

        if content_type == "json_evidence":
            header = self._build_turn_header(turn, label="json_evidence")
            cap = _safe_text(turn.get("caption", "")).strip()
            if self.memory_modal_mode == "caption":
                text = f"{header}\n[caption]\n{cap}" if cap else f"{header}\n[document; no caption]"
                return {"role": "user", "content": [{"type": "text", "text": text}]}

            content: list[dict[str, Any]] = [{"type": "text", "text": header}]
            for item in turn.get("content") or []:
                if not isinstance(item, dict):
                    continue
                it = item.get("type")
                inner = item.get("content") or {}
                if not isinstance(inner, dict):
                    inner = {}
                if it == "text":
                    t = _safe_text(inner.get("text", "")).strip()
                    if t:
                        content.append({"type": "text", "text": t})
                elif it == "image":
                    image_path = _safe_text(inner.get("image_path", "")).strip()
                    if image_path:
                        content.append({"type": "text", "text": f"[{MEMGPT_IMAGE_REF_MARKER_TAG}:{image_path}]"})
                    abs_image_path = os.path.join(Image_Root_Dir_Path, image_path)
                    image_url = image_path_to_data_url(abs_image_path)
                    content.append(_letta_image_part_from_ref(image_url, self.dataset_dir))

            return {"role": "user", "content": content} if len(content) > 1 or header.strip() else None

        return None

    def _build_ingest_chunks(self, conversations: list) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        current_turns = 0
        current_chars = 0
        current_images = 0

        def flush() -> None:
            nonlocal current, current_turns, current_chars, current_images
            if current:
                chunks.append(
                    {
                        "messages": list(current),
                        "turn_count": current_turns,
                        "char_count": current_chars,
                        "image_count": current_images,
                    }
                )
            current = []
            current_turns = 0
            current_chars = 0
            current_images = 0

        for turn in conversations:
            msg = self._turn_to_letta_message(turn)
            if msg is None:
                continue
            msg_chars, msg_images = _letta_message_size(msg)
            should_flush = False
            if current:
                if current_turns >= self.chunk_turns:
                    should_flush = True
                elif current_chars + msg_chars > self.chunk_max_chars:
                    should_flush = True
                elif current_images + msg_images > self.chunk_max_images:
                    should_flush = True
            if should_flush:
                flush()
            current.append(msg)
            current_turns += 1
            current_chars += msg_chars
            current_images += msg_images

        flush()
        return chunks

    def _create_memgpt_agent(self) -> str:
        letta = self._bootstrap_letta()
        api_key = _safe_text(getattr(self.args, "api_key", "")).strip()
        endpoint = _safe_text(getattr(self.args, "base_url", "")).strip().rstrip("/")
        if not api_key:
            raise RuntimeError("args.api_key is empty; MemGPT agent llm_config needs provider API key.")
        if not endpoint:
            raise RuntimeError("args.base_url is empty; MemGPT agent llm_config needs model endpoint.")
        if self.ingest_mode == "archival":
            memory_blocks: list[dict[str, Any]] = [
                {
                    "label": "persona",
                    "value": MEMGPT_ARCHIVAL_PERSONA_BLOCK,
                    "read_only": True,
                    "limit": 2500,
                },
                {
                    "label": "human",
                    "value": MEMGPT_ARCHIVAL_HUMAN_BLOCK,
                    "read_only": True,
                    "limit": 1500,
                },
            ]
        else:
            memory_blocks = [
                {
                    "label": "persona",
                    "value": MEMGPT_MESSAGES_PERSONA_BLOCK,
                },
                {
                    "label": "human",
                    "value": MEMGPT_MESSAGES_HUMAN_BLOCK,
                },
            ]
        agent = letta.agents.create(
            llm_config={
                "model": self._llm_model_name(),
                "model_endpoint_type": "openai",
                "model_endpoint": endpoint,
                "api_key": api_key,
                "context_window": self.context_window,
            },
            embedding_config={
                "embedding_model": self.embedding_model,
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": endpoint,
                "embedding_dim": self.embedding_dim,
                "api_key": api_key,
            },
            memory_blocks=memory_blocks,
            context_window_limit=self.context_window,
        )
        return agent.id

    def _clear_existing_agent_memory(self) -> None:
        if not self._agent_id:
            return
        letta = self._bootstrap_letta()
        old_agent_id = self._agent_id
        try:
            letta.agents.delete(old_agent_id)
            print(f"Cleared previous Letta agent memory: {old_agent_id}")
        except Exception as exc:
            print(f"Warning: failed to clear previous Letta agent {old_agent_id}: {exc}")
        finally:
            self._agent_id = None

    @staticmethod
    def _extract_assistant_text(response: Any) -> str:
        for msg in getattr(response, "messages", None) or []:
            if getattr(msg, "message_type", None) == "assistant_message":
                return _safe_text(getattr(msg, "content", None)).strip()
        return _safe_text(response).strip()

    def build_memory(
        self,
        conversations: list,
        conversation_streams: Optional[list],
        qa_samples: List[dict],
    ) -> None:
        del conversation_streams, qa_samples

        letta = self._bootstrap_letta()
        print("Loading Letta")
        self._clear_existing_agent_memory()
        self._agent_id = self._create_memgpt_agent()
        self._memory_built = False

        chunks = self._build_ingest_chunks(conversations)
        total = len(chunks)
        desc = (
            "Building MemGPT (Letta) archival memory"
            if self.ingest_mode == "archival"
            else "Building MemGPT (Letta) memory"
        )
        tag = self.archival_ingest_tag
        for idx, chunk in enumerate(tqdm(chunks, desc=desc), start=1):
            msgs = list(chunk["messages"])
            prefix = MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE.format(
                idx=idx, total=total, agent_label=MEMGPT_AGENT_LABEL
            )
            if self.ingest_mode == "archival":
                lines = [_letta_message_to_plaintext(m) for m in msgs]
                body = "\n\n".join(line for line in lines if line)
                full_text = f"{prefix}{body}".strip()
                base_tags = [tag, f"chunk_{idx}_of_{total}"]
                segments = _split_archival_text(full_text, self.archival_passage_max_chars)
                for part_i, seg in enumerate(segments, start=1):
                    seg_tags = list(base_tags)
                    if len(segments) > 1:
                        seg_tags.append(f"part_{part_i}_of_{len(segments)}")
                    letta.agents.passages.create(self._agent_id, text=seg, tags=seg_tags)
                continue

            if msgs and isinstance(msgs[0].get("content"), list):
                parts = list(msgs[0]["content"])
                if parts and isinstance(parts[0], dict) and parts[0].get("type") == "text":
                    parts[0] = {
                        **parts[0],
                        "text": prefix + _safe_text(parts[0].get("text", "")),
                    }
                    msgs[0] = {**msgs[0], "content": parts}
                else:
                    msgs[0] = {
                        **msgs[0],
                        "content": [{"type": "text", "text": prefix}] + parts,
                    }
            letta.agents.messages.create(self._agent_id, messages=msgs)

        if self.ingest_mode == "messages":
            letta.agents.messages.create(
                self._agent_id,
                input=MEMORY_INGEST_FINAL_MESSAGE,
            )

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

    def _archival_search_excerpts(self, query: str) -> str:
        assert self._letta is not None and self._agent_id
        resp = self._letta.agents.passages.search(
            self._agent_id,
            query=query,
            top_k=self.archival_top_k,
            tags=[self.archival_ingest_tag],
            tag_match_mode="any",
        )
        parts: list[str] = []
        for i, r in enumerate(resp.results or [], start=1):
            parts.append(f"[{i}] {_safe_text(getattr(r, 'content', None))}")
        return "\n\n".join(parts).strip()

    def _eval_image_ref_to_letta_part(self, ref: str) -> Optional[dict[str, Any]]:
        ref = _safe_text(ref).strip()
        if not ref:
            return None
        if ref.startswith(("http://", "https://")):
            return _letta_image_part_from_ref(ref, self.dataset_dir)
        abs_image_path = os.path.join(Image_Root_Dir_Path, ref)
        if not os.path.isfile(abs_image_path):
            return None
        image_url = image_path_to_data_url(abs_image_path)
        return _letta_image_part_from_ref(image_url, self.dataset_dir)

    def _archival_qa_user_content(
        self, recall_excerpts: str, user_text: str
    ) -> tuple[str | list[dict[str, Any]], bool]:
        intro = MEMGPT_ARCHIVAL_QA_INTRO
        core = recall_excerpts.strip() if recall_excerpts else "(no passages retrieved)"

        if self.archival_qa_max_images <= 0:
            body = intro + core + "\n\n---\n\nQuestion:\n" + user_text
            return body, False

        refs = _extract_image_refs_ordered(intro + core, self.archival_qa_max_images)
        image_parts: list[dict[str, Any]] = []
        for r in refs:
            p = self._eval_image_ref_to_letta_part(r)
            if p:
                image_parts.append(p)
        if not image_parts:
            body = intro + core + "\n\n---\n\nQuestion:\n" + user_text
            return body, False

        bridge = MEMGPT_ARCHIVAL_QA_BRIDGE_TEMPLATE.format(attached_count=len(image_parts))
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": intro + core + bridge},
        ]
        parts.extend(image_parts)
        parts.append({"type": "text", "text": f"---\n\nQuestion:\n{user_text}"})
        return parts, True

    def _answer_via_memgpt(
        self, qa_sample: dict
    ) -> tuple[str, list[dict[str, Any]], Optional[str], bool]:
        assert self._letta is not None and self._agent_id
        user_text = self._question_with_options(qa_sample)

        system_prompt = OVERALL_EVALUATION_SYSTEM_PROMPT
        if self.eval_format == "multiple_choice":
            system_prompt += MULTIPLE_CHOICE_OUTPUT_FORMAT
        elif self.eval_format == "fill_in_the_blank":
            system_prompt += FILL_IN_THE_BLANK_OUTPUT_FORMAT
        else:
            raise ValueError(f"Invalid evaluation format: {self.eval_format}")

        recall_excerpts: Optional[str] = None
        used_multimodal_qa = False
        if self.ingest_mode == "archival":
            recall_excerpts = self._archival_search_excerpts(user_text)
            user_content, used_multimodal_qa = self._archival_qa_user_content(recall_excerpts, user_text)
        else:
            user_content = user_text

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = self._letta.agents.messages.create(self._agent_id, messages=messages)
        return self._extract_assistant_text(resp), messages, recall_excerpts, used_multimodal_qa

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: Optional[list] = None,
    ):
        del conversations, conversation_streams
        if not self._memory_built or not self._agent_id:
            raise RuntimeError("MemGPT memory not built; run evaluate_cluster first.")

        parsed_function_calls = None
        answer_messages: list[dict[str, Any]] = []
        recall_excerpts: Optional[str] = None
        archival_qa_multimodal = False

        if qa_sample.get("category") == "Function_Call":
            answer_messages = build_function_call_messages(
                qa_sample.get("question", ""),
                "No Extra Context",
                self._function_call_candidate_tools,
            )
            raw = get_response(self.client, answer_messages, self.args)
            if isinstance(raw, tuple):
                raw = raw[0]
            raw_response = raw if isinstance(raw, str) else str(raw)
            print(raw_response)
            single_qa_result, parsed_function_calls = evaluate_function_call_response(
                raw_response,
                qa_sample.get("answer"),
            )
        else:
            raw_response, answer_messages, recall_excerpts, archival_qa_multimodal = self._answer_via_memgpt(
                qa_sample
            )
            print(raw_response)
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

        recall_context: Optional[str] = None
        if qa_sample.get("category") != "Function_Call" and self.ingest_mode == "archival":
            recall_context = recall_excerpts

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
            "recall_context": recall_context,
            "memgpt_server_url": self.server_url,
            "memgpt_agent_id": self._agent_id,
            "memory_modal_mode": self.memory_modal_mode,
            "memgpt_ingest_mode": self.ingest_mode,
            "memgpt_archival_qa_multimodal": archival_qa_multimodal,
        }

        return readable_answer, read_message

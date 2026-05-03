from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import requests
from tqdm import tqdm

from ..base_mem_agent import BaseMemoryAgent
from ..prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    MEMORY_CAPTION_ONLY_QUERY_IMAGES_PREFIX,
    MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE,
    MEMORY_INGEST_FINAL_MESSAGE,
    MIRIX_USER_PROMPT_TEMPLATE,
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

DEFAULT_MIRIX_API_URL = "http://localhost:8531"
DEFAULT_MIRIX_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
MIRIX_AGENT_LABEL = "MIRIX"


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _join_nonempty(parts: list[str], sep: str = "\n") -> str:
    return sep.join(part for part in parts if part and part.strip())


def _extract_answer_text(raw_response: Any) -> str:
    if isinstance(raw_response, tuple):
        raw_response = raw_response[0]
    if not isinstance(raw_response, str):
        return str(raw_response)
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return raw_response.strip()
    if isinstance(parsed, dict) and "answer" in parsed:
        return _safe_text(parsed.get("answer")).strip()
    return raw_response.strip()


def _resolve_memory_modal_mode(args: Any) -> str:
    override = _safe_text(getattr(args, "mirix_memory_modal_mode", "")).strip().lower()
    if override in ("original", "caption"):
        return override
    fallback = _safe_text(getattr(args, "mirix_ingest_media_mode", "original")).strip().lower() or "original"
    if fallback not in ("original", "caption"):
        raise ValueError(f"mirix memory modal mode must be 'original' or 'caption', got {fallback!r}")
    return fallback


def fetch_queue_trace_detail(client, trace_id: str, headers: dict[str, str]) -> dict[str, Any]:
    url = f"{client.base_url}/memory/queue-traces/{trace_id}"
    response = client.session.get(url, headers=headers, timeout=client.timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("detail", str(exc))
        except Exception:
            detail = response.text or str(exc)
        raise requests.HTTPError(f"API request failed: {detail}") from exc
    return response.json()


def delete_user_memories_for_eval(client, user_id: str, headers: dict[str, str]) -> bool:
    url = f"{client.base_url}/users/{quote(user_id, safe='')}/memories"
    try:
        response = client.session.delete(url, headers=headers, timeout=client.timeout)
    except requests.RequestException as exc:
        print(f"[MIRIX] delete_user_memories request failed: {exc}", flush=True)
        return False
    if response.status_code in (200, 204):
        return True
    if response.status_code == 404:
        return True
    snippet = (response.text or "")[:500]
    print(
        f"[MIRIX] delete_user_memories HTTP {response.status_code}: {snippet!r}",
        flush=True,
    )
    return False


def request_queue_trace_interrupt(
    client,
    trace_id: str,
    headers: dict[str, str],
    *,
    reason: Optional[str] = None,
) -> None:
    url = f"{client.base_url}/memory/queue-traces/{trace_id}/interrupt"
    payload = {"reason": reason or "Evaluation client: interrupt after wait timeout"}
    response = client.session.post(url, headers=headers, json=payload, timeout=client.timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("detail", str(exc))
        except Exception:
            detail = response.text or str(exc)
        raise requests.HTTPError(f"Interrupt API failed: {detail}") from exc


def wait_for_queue_trace_completed(
    client,
    trace_id: str,
    headers: dict[str, str],
    *,
    timeout_sec: float | None = None,
    poll_interval_sec: float | None = None,
    interrupt_drain_sec: float | None = None,
    skip_on_failure: bool = False,
) -> dict[str, Any] | None:
    if timeout_sec is None:
        timeout_sec = float(os.getenv("MIRIX_QUEUE_WAIT_TIMEOUT_SEC", "600"))
    if poll_interval_sec is None:
        poll_interval_sec = float(os.getenv("MIRIX_QUEUE_POLL_INTERVAL_SEC", "10"))
    if interrupt_drain_sec is None:
        interrupt_drain_sec = float(os.getenv("MIRIX_QUEUE_INTERRUPT_DRAIN_SEC", "180"))

    def _fail(msg: str, *, exc: Optional[BaseException] = None) -> None:
        print(f"[MIRIX queue trace {trace_id}] {msg}", flush=True)
        if exc is not None:
            print(exc, flush=True)

    def _poll_once() -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            data = fetch_queue_trace_detail(client, trace_id, headers)
        except (requests.HTTPError, requests.RequestException) as exc:
            if skip_on_failure:
                _fail("fetch_queue_trace_detail failed; skipping.", exc=exc)
                return None, "error"
            raise
        status = data["trace"]["status"]
        if status == "completed":
            return data, "completed"
        if status == "failed":
            err = data["trace"].get("error_message") or "unknown error"
            if skip_on_failure:
                _fail(f"status=failed ({err}); skipping.")
                return None, "failed"
            raise RuntimeError(f"Memory queue trace {trace_id} failed: {err}")
        if status not in ("queued", "processing"):
            msg = f"Unexpected queue trace status {status!r} for trace {trace_id}"
            if skip_on_failure:
                _fail(f"{msg}; skipping.")
                return None, "unexpected"
            raise RuntimeError(msg)
        return data, None

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        data, terminal = _poll_once()
        if terminal == "completed" and data is not None:
            return data
        if terminal in ("failed", "error", "unexpected"):
            return None
        print(
            f"Queue trace {trace_id} status: processing/queued at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            flush=True,
        )
        time.sleep(poll_interval_sec)

    print(
        f"[MIRIX queue trace {trace_id}] timeout after {timeout_sec}s; requesting interrupt.",
        flush=True,
    )
    try:
        request_queue_trace_interrupt(
            client,
            trace_id,
            headers,
            reason="Evaluation client: primary wait timeout",
        )
    except (requests.HTTPError, requests.RequestException) as exc:
        if skip_on_failure:
            _fail("request_queue_trace_interrupt failed; skipping.", exc=exc)
            return None
        raise

    drain_deadline = time.monotonic() + interrupt_drain_sec
    while time.monotonic() < drain_deadline:
        data, terminal = _poll_once()
        if terminal == "completed" and data is not None:
            return data
        if terminal in ("failed", "error", "unexpected"):
            return None
        time.sleep(poll_interval_sec)

    if skip_on_failure:
        _fail(
            f"still not terminal after interrupt + {interrupt_drain_sec}s drain; skipping."
        )
        return None
    raise TimeoutError(
        f"Queue trace {trace_id} did not reach status 'completed' within "
        f"{timeout_sec}s (+ {interrupt_drain_sec}s post-interrupt drain)"
    )


class MIRIXAgent(BaseMemoryAgent):

    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.args = args
        self.eval_format = getattr(args, "eval_format", "multiple_choice")
        self.dataset_dir = Path(args.dataset_dir_path).resolve()
        self.memory_modal_mode = _resolve_memory_modal_mode(args)

        self.chunk_turns = max(1, int(getattr(args, "mirix_chunk_turns", 6)))
        self.chunk_max_chars = max(1, int(getattr(args, "mirix_chunk_max_chars", 12000)))
        self.chunk_max_images = max(1, int(getattr(args, "mirix_chunk_max_images", 8)))
        self.retrieve_limit = max(1, int(getattr(args, "mirix_retrieve_limit", 8)))
        self.mirix_api_url = (
            (_safe_text(getattr(args, "mirix_api_url", "")).strip())
            or os.getenv("MIRIX_API_URL", DEFAULT_MIRIX_API_URL).strip()
        )
        self.mirix_org_id = (
            (_safe_text(getattr(args, "mirix_org_id", "")).strip())
            or os.getenv("MIRIX_ORG_ID", DEFAULT_MIRIX_ORG_ID).strip()
            or DEFAULT_MIRIX_ORG_ID
        )
        self.mirix_debug = bool(getattr(args, "mirix_debug", False))

        base_user_id = _safe_text(getattr(args, "mirix_user_id", "")).strip()
        if not base_user_id:
            cluster_name = (_safe_text(getattr(args, "save_dir_name", "")).strip() or self.dataset_dir.name or "cluster")
            base_user_id = f"mirix-eval-{cluster_name}-{uuid.uuid4().hex[:10]}"
        self.user_id = base_user_id

        self._mirix_client = None
        self._headers: dict[str, str] = {}
        self._memory_built = False
        self._ingested_turns = 0
        self._ingested_chunks = 0
        self._function_call_candidate_tools = load_function_call_candidate_tools()

    def _bootstrap_mirix_client(self):
        if self._mirix_client is not None:
            return self._mirix_client

        api_key = os.getenv("MIRIX_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set MIRIX_API_KEY before using the MIRIX evaluation agent.")

        model_api_base = self.args.base_url
        model_api_key = self.args.api_key
        if not model_api_base or not model_api_key:
            raise RuntimeError("Set args.base_url and args.api_key for MIRIX meta-agent initialization.")

        try:
            from mirix import MirixClient
        except ImportError as exc:
            raise ImportError(
                "Failed to import MIRIX client. Install it with `pip install mirix python-dotenv`."
            ) from exc

        self._headers = {"X-Org-Id": self.mirix_org_id}
        init_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "headers": self._headers,
            "debug": self.mirix_debug,
        }

        self._mirix_client = MirixClient(**init_kwargs)
        self._mirix_client.initialize_meta_agent(
            config=self._meta_agent_config(model_api_base, model_api_key),
            update_agents=True,
            headers=self._headers,
        )
        return self._mirix_client

    def _reset_mirix_client(self):
        self._mirix_client = None

    def _use_memory_embeddings(self) -> bool:
        if bool(getattr(self.args, "mirix_no_embeddings", False)):
            return False
        env = os.getenv("MIRIX_NO_EMBEDDINGS", "").strip().lower()
        if env in ("1", "true", "yes"):
            return False
        return True

    def _meta_agent_config(self, api_base: str, api_key: str) -> dict[str, Any]:
        llm_model = _safe_text(getattr(self.args, "model", "qwen3-vl-plus")).strip()
        embedding_model = _safe_text(getattr(self.args, "mirix_embedding_model", "text-embedding-v4")).strip()
        embedding_dim = int(getattr(self.args, "mirix_embedding_dim", 1024))
        use_embeddings = self._use_memory_embeddings()

        llm_block = {
            "model": llm_model,
            "model_endpoint_type": "openai",
            "model_endpoint": api_base.rstrip("/"),
            "api_key": api_key,
            "context_window": 131072,
            "max_tokens": max(1024, int(getattr(self.args, "max_tokens", 1024))),
            "temperature": float(getattr(self.args, "temperature", 0.01)),
        }
        agents_block = {
            "agents": [
                "core_memory_agent",
                "semantic_memory_agent",
            ],
        }

        if not use_embeddings:
            return {
                "build_embeddings_for_memory": False,
                "llm_config": llm_block,
                "meta_agent_config": agents_block,
            }

        return {
            "build_embeddings_for_memory": True,
            "llm_config": llm_block,
            "embedding_config": {
                "embedding_model": embedding_model,
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": api_base.rstrip("/"),
                "embedding_dim": embedding_dim,
                "api_key": api_key,
            },
            "meta_agent_config": agents_block,
        }

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
        return ""

    def _message_char_and_image_count(self, message: dict[str, Any]) -> tuple[int, int]:
        chars = 0
        images = 0
        for item in message.get("content") or []:
            if not isinstance(item, dict):
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

    def _resolve_turn_image_to_data_url(self, raw: str) -> str:
        ref = _safe_text(raw).strip()
        if not ref:
            return ""
        resolved = self._resolve_image_to_data_url(ref)
        if resolved:
            return resolved
        if not ref.startswith(("http://", "https://", "data:", "file://")):
            rooted = Path(Image_Root_Dir_Path) / ref.lstrip("/\\")
            if rooted.is_file():
                return image_path_to_data_url(str(rooted.resolve()))
        return ""

    def _turn_to_mirix_message(self, turn: dict, turn_idx: int) -> Optional[dict[str, Any]]:
        del turn_idx
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

            content: list[dict[str, Any]] = [{"type": "text", "text": header}]
            image_path = _safe_text(turn.get("image_path", "")).strip()
            abs_image_path = os.path.join(Image_Root_Dir_Path, image_path)
            image_url = image_path_to_data_url(abs_image_path)
            content.append({"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}})
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
                    abs_image_path = os.path.join(Image_Root_Dir_Path, image_path)
                    image_url = image_path_to_data_url(abs_image_path)
                    content.append({"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}})

            return {"role": "user", "content": content} if len(content) > 1 or header.strip() else None

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

        for turn_idx, turn in enumerate(conversations):
            message = self._turn_to_mirix_message(turn, turn_idx)
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

    def build_memory(
        self,
        conversations: list,
        conversation_streams: Optional[list],
        qa_samples: list[dict],
    ) -> None:
        del conversation_streams, qa_samples

        mirix_client = self._bootstrap_mirix_client()
        if not delete_user_memories_for_eval(mirix_client, self.user_id, self._headers):
            print("[MIRIX] Server purge unavailable or failed; re-creating MirixClient.", flush=True)
            self._reset_mirix_client()
            mirix_client = self._bootstrap_mirix_client()

        self._ingested_turns = 0
        self._ingested_chunks = 0
        self._memory_built = False

        chunks = self._build_ingest_chunks(conversations)
        total = len(chunks)
        for idx, chunk in enumerate(tqdm(chunks, desc="Building MIRIX memory", mininterval=0.5), start=1):
            msgs = list(chunk["messages"])
            prefix = MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE.format(
                idx=idx, total=total, agent_label=MIRIX_AGENT_LABEL
            )
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
            try:
                memory_result = mirix_client.add(
                    user_id=self.user_id,
                    messages=msgs,
                    headers=self._headers,
                )
            except Exception as exc:
                print(f"[MIRIX ingest chunk {idx}/{total}] add() failed: {exc}", flush=True)
                continue

            trace_id_1 = memory_result.get("trace_id")
            if not trace_id_1:
                print(
                    f"[MIRIX ingest chunk {idx}/{total}] add() response missing trace_id: {memory_result!r}",
                    flush=True,
                )
                continue
            detail = wait_for_queue_trace_completed(
                mirix_client,
                trace_id_1,
                self._headers,
                skip_on_failure=True,
            )
            if detail is None:
                continue
            self._ingested_turns += int(chunk["turn_count"])
            self._ingested_chunks += 1

        try:
            memory_result = mirix_client.add(
                user_id=self.user_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": MEMORY_INGEST_FINAL_MESSAGE}
                        ],
                    }
                ],
                headers=self._headers,
            )
        except Exception as exc:
            print(f"[MIRIX] final ingest add() failed: {exc}", flush=True)
        else:
            trace_id_2 = memory_result.get("trace_id")
            if not trace_id_2:
                print(
                    f"[MIRIX] final ingest add() response missing trace_id: {memory_result!r}",
                    flush=True,
                )
            else:
                wait_for_queue_trace_completed(
                    mirix_client,
                    trace_id_2,
                    self._headers,
                    skip_on_failure=True,
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
        option_lines = [f"{labels[i]}: {opt}" for i, opt in enumerate(options[:4])]
        return QUESTION_WITH_OPTIONS_TEMPLATE.format(
            question=question,
            candidate_options="\n".join(option_lines),
        )

    def _qa_query_messages(self, qa_sample: dict, *, include_options: bool) -> tuple[list[dict[str, Any]], int]:
        question = (
            self._question_with_options(qa_sample)
            if include_options
            else _safe_text(qa_sample.get("question", "")).strip()
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": question}]
        query_image_count = 0
        refs = self._qa_image_refs(qa_sample)

        if self.memory_modal_mode == "caption":
            if refs:
                lines = "\n".join(f"- [image evidence] {r}" for r in refs)
                content[0]["text"] = (
                    question
                    + MEMORY_CAPTION_ONLY_QUERY_IMAGES_PREFIX
                    + lines
                )
            return [{"role": "user", "content": content}], 0

        for raw in refs:
            data_url = self._resolve_turn_image_to_data_url(raw)
            if not data_url:
                continue
            content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}})
            query_image_count += 1
        return [{"role": "user", "content": content}], query_image_count

    def _memory_item_to_text(self, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, (int, float, bool)):
            return str(item)
        if isinstance(item, list):
            parts = [self._memory_item_to_text(part) for part in item]
            return _join_nonempty(parts, sep="\n\n")
        if isinstance(item, dict):
            summary_parts: list[str] = []
            for key in (
                "memory",
                "text",
                "summary",
                "value",
                "answer",
                "snippet",
                "context",
                "memories",
                "results",
                "items",
                "data",
            ):
                value = item.get(key)
                if value is None:
                    continue
                rendered = self._memory_item_to_text(value)
                if rendered:
                    summary_parts.append(rendered)

            content = item.get("content")
            if isinstance(content, list):
                rendered_content: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        rendered = self._memory_item_to_text(part)
                    elif part.get("type") == "text":
                        rendered = _safe_text(part.get("text", "")).strip()
                    elif part.get("type") == "image_url":
                        rendered = "[image evidence]"
                    else:
                        rendered = self._memory_item_to_text(part)
                    if rendered:
                        rendered_content.append(rendered)
                if rendered_content:
                    summary_parts.append(_join_nonempty(rendered_content, sep="\n"))

            if summary_parts:
                prefix_fields: list[str] = []
                for key in ("id", "memory_id", "score"):
                    value = item.get(key)
                    if value is not None and _safe_text(value).strip():
                        prefix_fields.append(f"{key}={value}")
                body = _join_nonempty(summary_parts, sep="\n")
                if prefix_fields:
                    return f"[{' | '.join(prefix_fields)}]\n{body}"
                return body

            try:
                return json.dumps(item, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(item)
        return str(item)

    def _retrieve_context(self, qa_sample: dict) -> tuple[str, Any, list[dict[str, Any]], int]:
        if self._mirix_client is None:
            raise RuntimeError("MIRIX client is not initialized. build_memory must run before QA evaluation.")
        retrieve_messages, query_image_count = self._qa_query_messages(qa_sample, include_options=False)
        raw_memories = self._mirix_client.retrieve_with_conversation(
            user_id=self.user_id,
            messages=retrieve_messages,
            limit=self.retrieve_limit,
            headers=self._headers,
        )
        context = self._memory_item_to_text(raw_memories).strip()
        return context, raw_memories, retrieve_messages, query_image_count

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
        user_text = MIRIX_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question_text,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        query_image_count = 0
        refs = self._qa_image_refs(qa_sample)

        if self.memory_modal_mode == "caption":
            if refs:
                lines = "\n".join(f"- [image evidence] {r}" for r in refs)
                content[0]["text"] += (
                    MEMORY_CAPTION_ONLY_QUERY_IMAGES_PREFIX
                    + lines
                )
        else:
            for raw in refs:
                data_url = self._resolve_turn_image_to_data_url(raw)
                if not data_url:
                    continue
                content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}})
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
        conversation_streams: list | None = None,
    ):
        del conversations, conversation_streams
        if not self._memory_built or self._mirix_client is None:
            raise RuntimeError("MIRIX memory is not initialized. build_memory must run before QA evaluation.")

        parsed_function_calls = None
        answer_messages: list[dict[str, Any]] = []
        context, raw_memories, retrieve_messages, retrieve_query_image_count = self._retrieve_context(qa_sample)

        if qa_sample.get("category") == "Function_Call":
            answer_messages = build_function_call_messages(
                qa_sample.get("question", ""),
                context,
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
            answer_query_image_count = 0
        else:
            raw_response, answer_messages, answer_query_image_count = self._answer_question_with_context(
                qa_sample,
                context,
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
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": qa_sample.get("question"),
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": raw_response,
            "parsed_function_calls": parsed_function_calls,
            "single_qa_result": single_qa_result,
            "messages": {
                "retrieve_messages": retrieve_messages,
                "answer_messages": answer_messages,
                "retrieved_context": context,
                "retrieved_memories_raw": raw_memories,
                "mirix_api_url": self.mirix_api_url,
                "mirix_org_id": self.mirix_org_id,
                "mirix_user_id": self.user_id,
                "ingested_turn_count": self._ingested_turns,
                "ingested_chunk_count": self._ingested_chunks,
                "chunk_turns": self.chunk_turns,
                "chunk_max_chars": self.chunk_max_chars,
                "chunk_max_images": self.chunk_max_images,
                "retrieve_limit": self.retrieve_limit,
                "retrieve_query_image_count": retrieve_query_image_count,
                "answer_query_image_count": answer_query_image_count,
                "memory_modal_mode": self.memory_modal_mode,
                "mirix_use_memory_embeddings": self._use_memory_embeddings(),
            },
        }
        return readable_answer, read_message

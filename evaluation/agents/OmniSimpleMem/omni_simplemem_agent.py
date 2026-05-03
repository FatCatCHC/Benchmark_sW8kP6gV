from __future__ import annotations

import shutil
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from utils import data_url_to_image_path

from ..base_mem_agent import BaseMemoryAgent
from ..prompt import OMNISIMPLEMEM_MULTIPLE_CHOICE_QUESTION_TEMPLATE

DEFAULT_OMNI_SIMPLEMEM_BASELINE_ROOT = (
    Path(__file__).resolve().parents[3] / "baselines" / "SimpleMem" / "OmniSimpleMem"
)
DEFAULT_TEXT_CHUNK_MAX_TURNS = 20
DEFAULT_TEXT_CHUNK_MAX_CHARS = 6000

def _safe_text(value) -> str:
    return "" if value is None else str(value)

def _question_with_options(qa_sample: dict, eval_format: str) -> str:
    question = _safe_text(qa_sample.get("question", "")).strip()
    if eval_format != "multiple_choice":
        return question

    mc = qa_sample.get("multi_choice_QA") or {}
    options = mc.get("multi_choice_QA_options") or []
    if not options:
        return question

    labels = ["(A)", "(B)", "(C)", "(D)"]
    option_lines = [f"{labels[i]}: {opt}" for i, opt in enumerate(options[:4])]
    return OMNISIMPLEMEM_MULTIPLE_CHOICE_QUESTION_TEMPLATE.format(
        question=question,
        candidate_options="\n".join(option_lines),
    )

def _uses_visualized_bge(text_model: str, visual_model: str) -> bool:
    text_value = (text_model or "").strip().lower().replace("\\", "/")
    visual_value = (visual_model or "").strip().lower().replace("\\", "/")
    if "bge-m3" not in text_value:
        return False
    return (
        "bge-visualized" in visual_value
        or "visualized_m3" in visual_value
        or visual_value.endswith(".pth")
    )

class OmniSimpleMemAgent(BaseMemoryAgent):

    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.args = args
        self.eval_format = args.eval_format
        self.dataset_dir = Path(args.dataset_dir_path).resolve()

        raw_root = (getattr(args, "omnisimplemem_baseline_root", "") or "").strip()
        self.omni_baseline_root = Path(raw_root or DEFAULT_OMNI_SIMPLEMEM_BASELINE_ROOT).resolve()
        self.runtime_root = Path(
            (getattr(args, "omnisimplemem_runtime_root", "") or "").strip()
            or (Path(args.output_dir).resolve() / "checkpoint" / "omnisimplemem")
        )
        self.top_k = max(1, int(getattr(args, "omnisimplemem_top_k", 5)))
        self.embedding_model = (
            getattr(args, "omnisimplemem_embedding_model", "") or "all-MiniLM-L6-v2"
        ).strip()
        self.embedding_dim = int(getattr(args, "omnisimplemem_embedding_dim", 384))
        self.visual_embedding_model = (
            getattr(args, "omnisimplemem_visual_embedding_model", "")
            or "UCSC-VLAA/openvision-vit-large-patch14-224"
        ).strip()
        self.visual_embedding_dim = int(getattr(args, "omnisimplemem_visual_embedding_dim", 768))

        self._orchestrator = None
        self._config_cls = None
        self._retrieval_level_cls = None
        self._expansion_request_cls = None
        self._ingested_turns = 0
        self._stored_text_items = 0
        self._stored_image_items = 0
        self._ingested_chunks = 0
        self._session_id = self._conversation_id()
        self._text_chunk_max_turns = max(
            1, int(getattr(args, "omnisimplemem_chunk_turns", DEFAULT_TEXT_CHUNK_MAX_TURNS))
        )
        self._text_chunk_max_chars = max(
            1, int(getattr(args, "omnisimplemem_chunk_max_chars", DEFAULT_TEXT_CHUNK_MAX_CHARS))
        )
        self._chunk_max_images = max(
            1, int(getattr(args, "omnisimplemem_chunk_max_images", 8))
        )
        self._progress_bar = None
        self._current_turn_type = "idle"

    def _conversation_id(self) -> str:
        return f"{self.args.save_dir_name or 'cluster'}__omnisimplemem"

    def _bootstrap_omni(self):
        root_candidate = self.omni_baseline_root
        if root_candidate.name != "OmniSimpleMem" and (root_candidate / "OmniSimpleMem").is_dir():
            root_candidate = (root_candidate / "OmniSimpleMem").resolve()

        if root_candidate.is_dir():
            root_str = str(root_candidate)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            self.omni_baseline_root = root_candidate

        try:
            from omni_memory import OmniMemoryConfig, OmniMemoryOrchestrator
            from omni_memory.retrieval.pyramid_retriever import (
                ExpansionRequest,
                RetrievalLevel,
            )
        except ImportError as exc:
            raise ImportError(
                "Failed to import Omni-SimpleMem. Install it with "
                "`cd baselines/SimpleMem/OmniSimpleMem && pip install -e .[all]`, "
                "or pass --omnisimplemem_baseline_root to a checked-out OmniSimpleMem repo. "
                f"Resolved baseline root: {self.omni_baseline_root}. Original error: {exc}"
            ) from exc

        self._config_cls = OmniMemoryConfig
        self._retrieval_level_cls = RetrievalLevel
        self._expansion_request_cls = ExpansionRequest
        return OmniMemoryOrchestrator, OmniMemoryConfig

    def _reset_runtime_state(self) -> None:
        if self._orchestrator is not None:
            try:
                self._orchestrator.close()
            except Exception:
                pass
            self._orchestrator = None
        shutil.rmtree(self.runtime_root, ignore_errors=True)
        self.runtime_root.mkdir(parents=True, exist_ok=True)

    def _build_config(self):
        _, config_cls = self._bootstrap_omni()
        if hasattr(config_cls, "create_default"):
            config = config_cls.create_default()
        else:
            config = config_cls()

        use_visualized_bge = _uses_visualized_bge(
            self.embedding_model,
            self.visual_embedding_model,
        )

        if hasattr(config, "embedding"):
            if hasattr(config.embedding, "model_name"):
                config.embedding.model_name = self.embedding_model
            if hasattr(config.embedding, "embedding_dim"):
                config.embedding.embedding_dim = self.embedding_dim
            if hasattr(config.embedding, "visual_embedding_model"):
                config.embedding.visual_embedding_model = self.visual_embedding_model
            if hasattr(config.embedding, "visual_embedding_dim"):
                config.embedding.visual_embedding_dim = self.visual_embedding_dim

        if hasattr(config, "entropy_trigger"):
            if use_visualized_bge:

                if hasattr(config.entropy_trigger, "enable_visual_trigger"):
                    config.entropy_trigger.enable_visual_trigger = True
            else:
                if hasattr(config.entropy_trigger, "visual_encoder"):
                    config.entropy_trigger.visual_encoder = "clip"
                if hasattr(config.entropy_trigger, "visual_model_name"):
                    config.entropy_trigger.visual_model_name = self.visual_embedding_model
                if hasattr(config.entropy_trigger, "enable_visual_trigger"):
                    config.entropy_trigger.enable_visual_trigger = True

        if hasattr(config, "llm"):
            llm_model = self.args.model

            if hasattr(config.llm, "api_key"):
                config.llm.api_key = getattr(self.args, "api_key", None)
            if hasattr(config.llm, "api_base_url"):
                config.llm.api_base_url = getattr(self.args, "base_url", None)
            if hasattr(config.llm, "summary_model"):
                config.llm.summary_model = llm_model
            if hasattr(config.llm, "query_model"):
                config.llm.query_model = llm_model
            if hasattr(config.llm, "temperature"):
                config.llm.temperature = float(getattr(self.args, "temperature", 0.0))

        if hasattr(config, "retrieval"):
            if hasattr(config.retrieval, "default_top_k"):
                config.retrieval.default_top_k = self.top_k

        return config

    def _materialize_image_ref(self, raw: str, prefix: str) -> str:
        if not raw or not isinstance(raw, str):
            return ""

        ref = raw.strip()
        if not ref:
            return ""

        input_dir = self.runtime_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        if ref.startswith("data:"):
            out_path = input_dir / f"{prefix}_{uuid.uuid4().hex}.png"
            data_url_to_image_path(ref, str(out_path))
            return str(out_path)

        if ref.startswith("file://"):
            ref = ref[7:]

        if ref.startswith(("http://", "https://")):
            suffix = Path(ref.split("?", 1)[0]).suffix or ".img"
            out_path = input_dir / f"{prefix}_{uuid.uuid4().hex}{suffix}"
            urllib.request.urlretrieve(ref, out_path)
            return str(out_path)

        ref_path = Path(ref)
        if ref_path.is_file():
            return str(ref_path.resolve())

        candidate = (self.dataset_dir / ref).resolve()
        if candidate.is_file():
            return str(candidate)

        return ""

    def _make_tags(
        self,
        timestamp: str,
        sender: str,
        conversation_name: str,
        content_type: str,
    ) -> list[str]:
        tags = [f"content_type:{content_type}", f"sender:{sender or 'unknown'}"]
        if timestamp:
            tags.append(f"timestamp:{timestamp}")
        if conversation_name:
            tags.append(f"conversation:{conversation_name}")
        return tags

    def _store_text_memory(
        self,
        text: str,
        *,
        tags: Optional[list[str]] = None,
    ) -> bool:
        if not text.strip():
            return False
        if self._orchestrator is None:
            raise RuntimeError("OmniSimpleMem runtime is not initialized.")

        result = self._orchestrator.add_text(
            text,
            session_id=self._session_id,
            tags=tags,
            force=True,
        )
        if not result.success:
            return False
        self._stored_text_items += 1
        return True

    def _store_image_memory(
        self,
        image_path: str,
        *,
        tags: Optional[list[str]] = None,
        caption: str = "",
    ) -> bool:
        if not image_path:
            return False
        if self._orchestrator is None:
            raise RuntimeError("OmniSimpleMem runtime is not initialized.")

        image_tags = list(tags or [])
        if "vision_on_demand" not in image_tags:
            image_tags.append("vision_on_demand")
        clean_caption = caption.strip()
        image_result = self._orchestrator.add_image(
            image_path,
            session_id=self._session_id,
            tags=image_tags,
            force=True,
            generate_caption=not bool(clean_caption),
            caption_text=clean_caption or None,
        )
        if not image_result.success:
            return False
        self._stored_image_items += 1
        return True

    @staticmethod
    def _merge_tags(existing: list[str], new_tags: Optional[list[str]]) -> list[str]:
        merged = list(existing)
        for tag in new_tags or []:
            if tag and tag not in merged:
                merged.append(tag)
        return merged

    def _refresh_progress(self, *, note: str = "") -> None:
        if self._progress_bar is None:
            return
        postfix = {
            "type": self._current_turn_type,
            "chunks": self._ingested_chunks,
            "text": self._stored_text_items,
            "images": self._stored_image_items,
        }
        if note:
            postfix["note"] = note
        self._progress_bar.set_postfix(postfix, refresh=True)

    def _format_text_turn(
        self,
        sender: str,
        timestamp: str,
        conversation_name: str,
        content: str,
    ) -> str:
        prefix = f"{sender} at {timestamp}".strip()
        if conversation_name:
            prefix += f" in {conversation_name}"
        return f"{prefix}: {content}".strip()

    def _format_image_text(
        self,
        sender: str,
        timestamp: str,
        conversation_name: str,
        caption: str,
        *,
        label: str = "image",
    ) -> str:
        prefix = f"{sender} shared a {label} at {timestamp}".strip()
        if conversation_name:
            prefix += f" in {conversation_name}"
        if caption:
            return f"{prefix}. Image caption: {caption}"
        return f"{prefix}."

    def _json_evidence_to_text(self, turn: dict) -> str:
        sender = _safe_text(turn.get("sender_name", "unknown")).strip() or "unknown"
        timestamp = _safe_text(turn.get("timestamp", "")).strip()
        conversation_name = _safe_text(turn.get("conversation_name", "")).strip()
        first_line = f"{sender} shared a json evidence document at {timestamp}".strip()
        if conversation_name:
            first_line += f" in {conversation_name}"

        parts: list[str] = [first_line]
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
                    parts.append(text)
            elif item_type == "image":
                caption = _safe_text(item.get("caption", "")).strip()
                if caption:
                    parts.append(f"Embedded image caption: {caption}")

        caption = _safe_text(turn.get("caption", "")).strip()
        if caption:
            parts.append(f"Document caption: {caption}")
        return "\n".join(part for part in parts if part.strip())

    def _turn_to_chunk_item(self, turn: dict, turn_idx: int) -> Optional[dict]:
        sender = _safe_text(turn.get("sender_name", "unknown")).strip() or "unknown"
        timestamp = _safe_text(turn.get("timestamp", "")).strip()
        conversation_name = _safe_text(turn.get("conversation_name", "")).strip()
        content_type = _safe_text(turn.get("content_type", "text")).strip() or "text"
        base_tags = self._make_tags(timestamp, sender, conversation_name, content_type)

        text_parts: list[str] = []
        image_items: list[dict] = []

        if content_type == "text":
            content = _safe_text(turn.get("content", "")).strip()
            if content:
                text_parts.append(
                    self._format_text_turn(sender, timestamp, conversation_name, content)
                )

        elif content_type == "image":
            caption = _safe_text(turn.get("caption", "")).strip()
            image_path = self._materialize_image_ref(
                _safe_text(turn.get("image_path") or turn.get("image_url") or turn.get("content")),
                f"turn_{turn_idx}",
            )
            if caption:
                text_parts.append(
                    self._format_image_text(
                        sender,
                        timestamp,
                        conversation_name,
                        caption,
                    )
                )
            elif image_path:
                text_parts.append(
                    self._format_image_text(
                        sender,
                        timestamp,
                        conversation_name,
                        "",
                    )
                )
            if image_path:
                image_tags = list(base_tags)
                if caption:
                    image_tags.append(f"caption:{caption}")
                image_items.append(
                    {
                        "image_path": image_path,
                        "caption": caption,
                        "tags": image_tags,
                    }
                )

        elif content_type == "json_evidence":
            document_text = self._json_evidence_to_text(turn)
            if document_text:
                text_parts.append(document_text)

            for item_idx, item in enumerate(turn.get("content") or []):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                inner = item.get("content") or {}
                if not isinstance(inner, dict):
                    continue
                image_path = self._materialize_image_ref(
                    _safe_text(inner.get("image_url") or inner.get("image_path")),
                    f"turn_{turn_idx}_{item_idx}",
                )
                if not image_path:
                    continue
                caption = _safe_text(item.get("caption", "")).strip()
                image_tags = list(base_tags)
                if caption:
                    image_tags.append(f"caption:{caption}")
                image_items.append(
                    {
                        "image_path": image_path,
                        "caption": caption,
                        "tags": image_tags,
                    }
                )

        else:
            fallback_text = _safe_text(turn.get("content", "")).strip()
            if fallback_text:
                text_parts.append(
                    self._format_text_turn(
                        sender,
                        timestamp,
                        conversation_name,
                        fallback_text,
                    )
                )

        if not text_parts and not image_items:
            return None

        merged_tags = list(base_tags)
        for image_item in image_items:
            merged_tags = self._merge_tags(merged_tags, image_item.get("tags"))

        return {
            "turn_count": 1,
            "char_count": sum(len(part) for part in text_parts),
            "image_count": len(image_items),
            "text_parts": text_parts,
            "image_items": image_items,
            "tags": merged_tags,
            "content_type": content_type,
        }

    def _build_ingest_chunks(self, conversations: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        current_text_parts: list[str] = []
        current_image_items: list[dict] = []
        current_tags: list[str] = []
        current_turns = 0
        current_chars = 0
        current_images = 0

        def flush() -> None:
            nonlocal current_text_parts, current_image_items, current_tags
            nonlocal current_turns, current_chars, current_images
            if current_turns <= 0 and not current_text_parts and not current_image_items:
                return
            chunks.append(
                {
                    "text": "\n\n".join(part for part in current_text_parts if part and part.strip()).strip(),
                    "image_items": list(current_image_items),
                    "tags": list(current_tags),
                    "turn_count": current_turns,
                    "char_count": current_chars,
                    "image_count": current_images,
                }
            )
            current_text_parts = []
            current_image_items = []
            current_tags = []
            current_turns = 0
            current_chars = 0
            current_images = 0

        for turn_idx, turn in enumerate(conversations):
            item = self._turn_to_chunk_item(turn, turn_idx)
            if item is None:
                continue

            should_flush = False
            if current_turns > 0:
                if current_turns >= self._text_chunk_max_turns:
                    should_flush = True
                elif current_chars + int(item["char_count"]) > self._text_chunk_max_chars:
                    should_flush = True
                elif current_images + int(item["image_count"]) > self._chunk_max_images:
                    should_flush = True

            if should_flush:
                flush()

            current_text_parts.extend(item["text_parts"])
            current_image_items.extend(item["image_items"])
            current_tags = self._merge_tags(current_tags, item["tags"])
            current_turns += int(item["turn_count"])
            current_chars += int(item["char_count"])
            current_images += int(item["image_count"])

        flush()
        return chunks
    def build_memory(
        self,
        conversations: list,
        conversation_streams: Optional[list],
        qa_samples: list[dict],
    ) -> None:
        del conversation_streams, qa_samples

        self._reset_runtime_state()
        orchestrator_cls, _ = self._bootstrap_omni()
        config = self._build_config()
        self._orchestrator = orchestrator_cls(config=config, data_dir=str(self.runtime_root))
        self._orchestrator.start_session(self._session_id)

        self._ingested_turns = 0
        self._stored_text_items = 0
        self._stored_image_items = 0
        self._ingested_chunks = 0
        self._current_turn_type = "startup"
        chunks = self._build_ingest_chunks(conversations)
        self._progress_bar = tqdm(chunks, desc="Building OmniSimpleMem memory", mininterval=0.5)
        self._refresh_progress(note="init")
        try:
            for chunk_idx, chunk in enumerate(self._progress_bar, start=1):
                self._current_turn_type = "chunk"
                chunk_note = (
                    f"chunk={chunk_idx} turns={chunk['turn_count']} "
                    f"chars={chunk['char_count']} images={chunk['image_count']}"
                )
                self._refresh_progress(note=chunk_note)

                text = _safe_text(chunk.get("text", "")).strip()
                if text:
                    chunk_tags = self._merge_tags(chunk.get("tags") or [], ["chunked_text"])
                    self._store_text_memory(text, tags=chunk_tags)

                for image_item in chunk.get("image_items") or []:
                    self._store_image_memory(
                        _safe_text(image_item.get("image_path")),
                        tags=image_item.get("tags") or chunk.get("tags"),
                        caption=_safe_text(image_item.get("caption", "")).strip(),
                    )

                self._ingested_turns += int(chunk.get("turn_count") or 0)
                self._ingested_chunks += 1
                self._refresh_progress(note="stored")
        finally:
            if self._progress_bar is not None:
                self._progress_bar.close()
            self._progress_bar = None

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
    ):
        del conversations, conversation_streams

        if self._orchestrator is None:
            raise RuntimeError("OmniSimpleMem runtime is not initialized. build_memory must run before QA.")

        question = _question_with_options(qa_sample, self.eval_format)
        raw_images = qa_sample.get("images") or []
        query_image_used = False
        if raw_images:
            query_image_used = bool(
                self._materialize_image_ref(_safe_text(raw_images[0]), "qa_image")
            )

        result = self._orchestrator.answer(
            question=question,
            top_k=self.top_k,
            include_sources=True,
            include_on_demand_images=True,
        )
        raw_response = result["answer"]
        single_qa_result = self._evaluate_answer(raw_response, qa_sample)

        ground_truth = qa_sample.get("answer")
        if (
            self.eval_format == "multiple_choice"
            and qa_sample.get("category") != "Function_Call"
        ):
            mc = qa_sample.get("multi_choice_QA") or {}
            ans_idx = mc.get("multi_choice_QA_answer")
            if isinstance(ans_idx, int) and 0 <= ans_idx < 4:
                ground_truth = ["(A)", "(B)", "(C)", "(D)"][ans_idx]

        readable_answer = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": qa_sample.get("question"),
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": raw_response,
            "single_qa_result": single_qa_result,
        }
        read_message = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": qa_sample.get("question"),
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": raw_response,
            "single_qa_result": single_qa_result,
            "messages": {
                "question": question,
                "top_k": self.top_k,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "visual_embedding_model": self.visual_embedding_model,
                "visual_embedding_dim": self.visual_embedding_dim,
                "baseline_root": str(self.omni_baseline_root),
                "runtime_root": str(self.runtime_root),
                "session_id": self._session_id,
                "ingested_turn_count": self._ingested_turns,
                "stored_text_items": self._stored_text_items,
                "stored_image_items": self._stored_image_items,
                "query_image_used": query_image_used,
                "query_image_supported_by_adapter": False,
                "sources": result.get("sources", []),
                "retrieval_result": result.get("retrieval_result"),
            },
        }
        return readable_answer, read_message

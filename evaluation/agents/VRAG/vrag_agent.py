from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Optional

import requests
from PIL import Image
import time
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import torch

from constant import Image_Root_Dir_Path, VLM_METAINFO_MAP
from dataset_loader import build_memory_time_ordered_turns
from universal_rag_bge import retrieve_from_passage_list
from universal_rag_visual_bge import BGEVisualImageRetriever, encode_visual_text_query, load_visualized_bge

from ..base_rag_agent import BaseRAGAgent
from ..prompt import (
    VRAG_FILL_IN_THE_BLANK_ANSWER_FORMAT_INSTRUCTION,
    VRAG_FORCE_ANSWER_MESSAGE,
    VRAG_FUNCTION_CALL_USER_SUFFIX,
    VRAG_INVALID_ACTION_MESSAGE,
    VRAG_MULTIPLE_CHOICE_ANSWER_FORMAT_INSTRUCTION,
    VRAG_NO_ANSWER_TEXT,
    VRAG_SEARCH_FAILED_MESSAGE,
    VRAG_SYSTEM_PROMPT,
)
from utils import evaluate_function_call_response

DEFAULT_VRAG_EMBEDDING_ROOT = os.environ.get(
    "UNIVERSALRAG_EMBEDDING_ROOT",
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..",
            "datasets", "UniversalRAG_embedding",
        )
    ),
)
DEFAULT_VISUAL_BGE_BACKBONE = "BAAI/bge-m3"

def _safe_text(value: Any) -> str:
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
    return f"{question}\n\nCandidate options:\n" + "\n".join(option_lines)

def _answer_format_instruction(eval_format: str) -> str:
    if eval_format == "multiple_choice":
        return VRAG_MULTIPLE_CHOICE_ANSWER_FORMAT_INSTRUCTION
    return VRAG_FILL_IN_THE_BLANK_ANSWER_FORMAT_INSTRUCTION

def _normalize_visual_weight_path(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if os.path.isdir(raw):
        return os.path.join(raw, "Visualized_m3.pth")
    return raw

def _normalize_image_ref(raw: str, base_dir: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    ref = raw.strip()
    if not ref:
        return ""
    if ref.startswith("data:"):
        return ref
    if ref.startswith(("http://", "https://")):
        return ref
    if ref.startswith("file://"):
        ref = ref[7:]
    if os.path.isabs(ref):
        return ref if os.path.isfile(ref) else ""

    basename = os.path.basename(ref)
    candidates: list[str] = []
    base_dir = (base_dir or "").strip()
    if base_dir:
        candidates.append(os.path.normpath(os.path.join(base_dir, ref)))
        candidates.append(os.path.normpath(os.path.join(base_dir, "images", ref)))
        candidates.append(os.path.normpath(os.path.join(base_dir, "images", basename)))

        grandparent = os.path.dirname(os.path.dirname(base_dir))
        if grandparent:
            dataset_name = os.path.basename(os.path.dirname(base_dir))
            if dataset_name:
                candidates.append(
                    os.path.normpath(os.path.join(grandparent, f"{dataset_name}_Images", basename))
                )
            candidates.append(os.path.normpath(os.path.join(grandparent, "MidFinal_Images", basename)))
    candidates.append(os.path.normpath(os.path.join(Image_Root_Dir_Path, basename)))

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            return cand
    return ""

def _json_evidence_to_text(turn: dict) -> str:
    sender = _safe_text(turn.get("sender_name", "unknown"))
    timestamp = _safe_text(turn.get("timestamp", ""))
    conv_name = _safe_text(turn.get("conversation_name", ""))
    parts: list[str] = [f"{sender} sent a json evidence document at {timestamp}"]
    if conv_name:
        parts[0] += f" in {conv_name}"

    caption = _safe_text(turn.get("caption", "")).strip()
    if caption:
        parts.append(f"Document caption: {caption}")
    return "\n".join(part for part in parts if part.strip())

def _image_ref_to_pil(image_ref: str) -> Image.Image:
    if image_ref.startswith("data:"):
        _, b64_data = image_ref.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image_ref.startswith(("http://", "https://")):
        response = requests.get(image_ref, timeout=20)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(image_ref).convert("RGB")

def _pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def _lexical_rank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    q_tokens = _tokenize_for_overlap(question)
    scored = []
    for item in candidates:
        line = _safe_text(item.get("line", ""))
        t_tokens = _tokenize_for_overlap(line)
        overlap = len(q_tokens & t_tokens)
        scored.append((overlap, line, item))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [item for _, _, item in scored[:top_k]]

@dataclass
class ObservedImage:
    source_ref: str
    raw_image: Image.Image
    input_image: Image.Image
    context_line: str

class VRAGAgent(BaseRAGAgent):
    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.eval_strategy = args.eval_strategy
        self.eval_format = args.eval_format
        self.model = args.model
        self.max_tokens = int(getattr(args, "max_tokens", 512))
        self.temperature = float(getattr(args, "temperature", 0.01))
        self.vrag_top_k = int(getattr(args, "vrag_top_k", 3))
        self.vrag_max_steps = int(getattr(args, "vrag_max_steps", 6))
        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.repeated_nums = 1
        self.token_limit = VLM_METAINFO_MAP[args.model]["token_limit"]
        self._retrieval_device = self._resolve_retrieval_device(getattr(args, "vrag_gpu", None))

        self._image_retriever: Optional[BGEVisualImageRetriever] = None
        self._vbge_model = None
        self._text_bge = None
        self._load_precomputed_retriever()

    @staticmethod
    def _resolve_retrieval_device(gpu_arg: Optional[str]) -> str:
        if gpu_arg is None or str(gpu_arg).strip() == "":
            return "cuda" if torch.cuda.is_available() else "cpu"
        raw = str(gpu_arg).strip().lower()
        if raw == "cpu":
            return "cpu"
        if raw.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError(f"VRAG retrieval requested GPU device {raw}, but CUDA is not available.")
            return raw
        if not raw.isdigit():
            raise ValueError(
                f"Invalid --vrag_gpu value: {gpu_arg!r}. Expected 'cpu', 'cuda', 'cuda:N', or a GPU index like '0'."
            )
        if not torch.cuda.is_available():
            raise ValueError(f"VRAG retrieval requested GPU device cuda:{raw}, but CUDA is not available.")
        gpu_index = int(raw)
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid --vrag_gpu value: {gpu_arg!r}. Available CUDA device count: {torch.cuda.device_count()}."
            )
        return f"cuda:{gpu_index}"

    def _infer_vrag_index_paths(self) -> tuple[str, str]:
        dataset_dir = (getattr(self.args, "dataset_dir_path", "") or "").strip()
        if not dataset_dir:
            return "", ""
        cluster_name = os.path.basename(os.path.normpath(dataset_dir))
        dataset_name = os.path.basename(os.path.dirname(os.path.normpath(dataset_dir)))
        embedding_root = (
            getattr(self.args, "vrag_embedding_root", "") or DEFAULT_VRAG_EMBEDDING_ROOT
        ).strip()
        base_dir = os.path.join(embedding_root, dataset_name, cluster_name)
        return os.path.join(base_dir, "image.pkl"), os.path.join(base_dir, "corpus_meta.json")

    def _visual_weight_path(self) -> str:
        explicit = _normalize_visual_weight_path(getattr(self.args, "vrag_visual_bge_weight", ""))
        if explicit:
            return explicit
        env_path = _normalize_visual_weight_path(os.environ.get("BGE_M3_MODEL_PATH", ""))
        return env_path

    def _load_precomputed_retriever(self) -> None:
        image_pkl = (getattr(self.args, "vrag_image_pkl", "") or "").strip()
        meta_path = (getattr(self.args, "vrag_corpus_meta", "") or "").strip()

        if not image_pkl:
            image_pkl = (getattr(self.args, "universal_rag_image_pkl", "") or "").strip()
        if not meta_path:
            meta_path = (getattr(self.args, "universal_rag_corpus_meta", "") or "").strip()
        if not image_pkl and not meta_path:
            image_pkl, meta_path = self._infer_vrag_index_paths()

        if not (image_pkl and meta_path and os.path.isfile(image_pkl) and os.path.isfile(meta_path)):
            return

        try:
            from universal_rag_bge import load_corpus_meta

            meta = load_corpus_meta(meta_path)
            image_meta = meta.get("image") or {}
            if image_meta:
                self._image_retriever = BGEVisualImageRetriever(image_pkl, image_meta)
        except Exception as e:
            print(f"[VRAGAgent] failed to load image retriever: {e}", file=sys.stderr, flush=True)
            self._image_retriever = None

    def _get_visual_retriever_model(self):
        if self._vbge_model is None:
            weight_path = self._visual_weight_path()
            if not weight_path or not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    "VRAG visual retrieval requires Visualized-BGE weights. "
                    "Set --vrag_visual_bge_weight or BGE_M3_MODEL_PATH."
                )
            self._vbge_model = load_visualized_bge(
                DEFAULT_VISUAL_BGE_BACKBONE,
                weight_path,
                device=self._retrieval_device if self._retrieval_device != "cpu" else None,
            )
        return self._vbge_model

    def _get_text_bge(self):
        if self._text_bge is None:
            from sentence_transformers import SentenceTransformer

            model_path = (getattr(self.args, "vrag_bge_model_path", "") or "").strip()
            if not model_path:
                model_path = (getattr(self.args, "universal_rag_bge_model_path", "") or "").strip()
            if not model_path:
                model_path = os.environ.get("BGE_LARGE_MODEL_PATH", "").strip() or "BAAI/bge-large-en-v1.5"
            self._text_bge = SentenceTransformer(model_path, device=self._retrieval_device)
        return self._text_bge

    def _build_image_candidates(self, turns: list[dict]) -> list[dict]:
        candidates: list[dict] = []
        for idx, turn in enumerate(turns):
            if turn.get("content_type") != "image":
                continue

            image_path = os.path.join(Image_Root_Dir_Path, _safe_text(turn.get("image_path")))
            ref = image_path
            if not os.path.isfile(ref):
                print(
                    "[VRAGAgent] skip image turn: could not resolve "
                    f"image_path={image_path!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            sender = _safe_text(turn.get("sender_name", "unknown"))
            timestamp = _safe_text(turn.get("timestamp", ""))
            conv_name = _safe_text(turn.get("conversation_name", ""))

            line = f"{sender} sent an image at {timestamp}"
            if conv_name:
                line += f" in {conv_name}"

            candidates.append(
                {
                    "id": f"turn_image_{idx}",
                    "ref": ref,
                    "line": line,
                }
            )
        return candidates

    def _build_text_candidates(self, turns: list[dict]) -> list[dict]:
        candidates: list[dict] = []
        for idx, turn in enumerate(turns):
            if turn.get("content_type") != "text":
                continue
            sender = _safe_text(turn.get("sender_name", "unknown"))
            timestamp = _safe_text(turn.get("timestamp", ""))
            conv_name = _safe_text(turn.get("conversation_name", ""))
            content = _safe_text(turn.get("content", "")).strip()
            if not content:
                continue
            line = f"{sender} at {timestamp}"
            if conv_name:
                line += f" in {conv_name}"
            line += f": {content}"
            candidates.append(
                {
                    "id": f"turn_text_{idx}",
                    "text": line,
                    "line": line,
                }
            )
        return candidates

    def _build_json_candidates(self, turns: list[dict]) -> list[dict]:
        candidates: list[dict] = []
        for idx, turn in enumerate(turns):
            if turn.get("content_type") != "json_evidence":
                continue
            text = _json_evidence_to_text(turn).strip()
            if not text:
                continue
            candidates.append(
                {
                    "id": f"turn_json_{idx}",
                    "text": text,
                    "line": text,
                }
            )
        return candidates

    def _rank_images(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        allowed_refs = {item["ref"] for item in candidates}
        if self._image_retriever is not None and self._image_retriever.vectors.shape[0] > 0:
            try:
                model = self._get_visual_retriever_model()
                qv = encode_visual_text_query(model, question)
                ranked = self._image_retriever.retrieve(
                    qv,
                    max(self.vrag_top_k * 3, self.vrag_top_k),
                    similarity_device=self._retrieval_device,
                )
                out = []
                for tid, image_path, line in ranked:
                    if image_path not in allowed_refs:
                        continue
                    out.append({"id": tid, "ref": image_path, "line": line})
                    if len(out) >= self.vrag_top_k:
                        return out
            except Exception as e:
                print(f"[VRAGAgent] visual retrieval failed, fallback to text ranking: {e}", file=sys.stderr, flush=True)

        try:
            text_retriever = self._get_text_bge()
            passages = [item["line"] for item in candidates]
            _, indices = retrieve_from_passage_list(
                text_retriever,
                question,
                passages,
                self.vrag_top_k,
                similarity_device=self._retrieval_device,
            )
            if indices:
                return [candidates[i] for i in indices]
        except Exception as e:
            print(f"[VRAGAgent] text ranking unavailable, fallback to lexical ranking: {e}", file=sys.stderr, flush=True)

        return _lexical_rank(question, candidates, self.vrag_top_k)

    def _rank_texts(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        try:
            text_retriever = self._get_text_bge()
            passages = [item["text"] for item in candidates]
            _, indices = retrieve_from_passage_list(
                text_retriever,
                question,
                passages,
                self.vrag_top_k,
                similarity_device=self._retrieval_device,
            )
            if indices:
                return [candidates[i] for i in indices]
        except Exception as e:
            print(
                f"[VRAGAgent] text retrieval unavailable, fallback to lexical ranking: {e}",
                file=sys.stderr,
                flush=True,
            )

        return _lexical_rank(question, candidates, self.vrag_top_k)

    def _format_text_observation(self, text_items: list[dict]) -> str:
        if not text_items:
            return ""
        lines = [f"[{i + 1}] {item['text']}" for i, item in enumerate(text_items)]
        return "Retrieved text passages:\n" + "\n\n".join(lines)

    def _format_json_observation(self, json_items: list[dict]) -> str:
        if not json_items:
            return ""
        lines = [f"[{i + 1}] {item['text']}" for i, item in enumerate(json_items)]
        return "Retrieved json evidence:\n" + "\n\n".join(lines)

    def _process_image(self, image: Image.Image) -> tuple[Image.Image, str]:
        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image, _pil_to_data_url(image)

    def _build_initial_messages(self, qa_sample: dict) -> list[dict]:
        question = _question_with_options(qa_sample, self.eval_format)
        user_tail = f"Question: {question}\n\n{_answer_format_instruction(self.eval_format)}"
        if qa_sample.get("category") == "Function_Call":
            user_tail = user_tail + "\n\n" + VRAG_FUNCTION_CALL_USER_SUFFIX
        return [
            {"role": "system", "content": VRAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_tail},
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(Exception))
    def _chat(self, messages: list[dict]) -> str:
        time.sleep(2)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        print(response.choices[0].message.content)
        return _safe_text(response.choices[0].message.content)

    def _parse_response(self, text: str) -> tuple[Optional[str], str, str]:
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""
        action_match = re.search(r"<(search|answer|bbox)>(.*?)</\1>", text, re.DOTALL)
        if not action_match:
            return None, "", thought
        return action_match.group(1), action_match.group(2).strip(), thought

    def _crop_latest_image(self, observed: ObservedImage, bbox_content: str) -> tuple[Optional[ObservedImage], Optional[str]]:
        try:
            bbox = json.loads(bbox_content)
        except Exception:
            return None, VRAG_INVALID_ACTION_MESSAGE

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(v, (int, float)) for v in bbox)
            or any(v < 0 for v in bbox)
        ):
            return None, VRAG_INVALID_ACTION_MESSAGE

        x1, y1, x2, y2 = bbox
        input_w, input_h = observed.input_image.size
        raw_w, raw_h = observed.raw_image.size
        crop_region = [
            raw_w * x1 / max(input_w, 1),
            raw_h * y1 / max(input_h, 1),
            raw_w * x2 / max(input_w, 1),
            raw_h * y2 / max(input_h, 1),
        ]
        pad_size = 56
        crop_region = [
            max(int(crop_region[0] - pad_size), 0),
            max(int(crop_region[1] - pad_size), 0),
            min(int(crop_region[2] + pad_size), raw_w),
            min(int(crop_region[3] + pad_size), raw_h),
        ]
        if crop_region[2] <= crop_region[0] or crop_region[3] <= crop_region[1]:
            return None, VRAG_INVALID_ACTION_MESSAGE

        crop_box = (crop_region[0], crop_region[1], crop_region[2], crop_region[3])
        cropped = observed.raw_image.crop(crop_box)
        processed, _ = self._process_image(cropped)
        return ObservedImage(
            source_ref=observed.source_ref,
            raw_image=cropped,
            input_image=processed,
            context_line=f"Cropped view of: {observed.context_line}",
        ), None

    def _result_pair(
        self,
        qa_sample: dict,
        answer_text: str,
        messages: list[dict],
        trajectory: list[dict],
        parsed_function_calls: list[dict] | None = None,
        single_qa_result_override: bool | None = None,
    ):
        final_text = answer_text.strip() or VRAG_NO_ANSWER_TEXT
        if single_qa_result_override is None:
            single_qa_result = self._evaluate_answer(final_text, qa_sample)
        else:
            single_qa_result = single_qa_result_override
        ground_truth = qa_sample.get("answer")
        if self.eval_format == "multiple_choice" and qa_sample.get("category") != "Function_Call":
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
            "raw_response": final_text,
            "parsed_function_calls": parsed_function_calls,
            "single_qa_result": single_qa_result,
        }
        read_message = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": qa_sample.get("question"),
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": final_text,
            "parsed_function_calls": parsed_function_calls,
            "single_qa_result": single_qa_result,
            "messages": messages,
            "trajectory": trajectory,
        }
        return readable_answer, read_message

    def _finalize_vrag_trajectory(
        self,
        qa_sample: dict,
        answer_text: str,
        messages: list,
        trajectory: list,
    ):
        if qa_sample.get("category") == "Function_Call":
            single_qa_result, parsed_function_calls = evaluate_function_call_response(
                answer_text,
                qa_sample.get("answer"),
            )
            return self._result_pair(
                qa_sample,
                answer_text,
                messages,
                trajectory,
                parsed_function_calls=parsed_function_calls,
                single_qa_result_override=single_qa_result,
            )
        return self._result_pair(qa_sample, answer_text, messages, trajectory)

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
    ):
        turns = conversations
        image_candidates = self._build_image_candidates(turns)
        text_candidates = self._build_text_candidates(turns)
        json_candidates = self._build_json_candidates(turns)

        messages = self._build_initial_messages(qa_sample)
        seen_image_refs: dict[str, int] = {}
        seen_text_ids: set[str] = set()
        seen_json_ids: set[str] = set()
        observed_images: list[ObservedImage] = []
        trajectory: list[dict] = []

        for step in range(self.vrag_max_steps):
            raw_response = self._chat(messages)
            messages.append({"role": "assistant", "content": raw_response})
            action, content, thought = self._parse_response(raw_response)
            trace_item = {
                "step": step + 1,
                "thought": thought,
                "action": action,
                "content": content,
            }

            if action == "answer":
                trajectory.append(trace_item)
                return self._finalize_vrag_trajectory(
                    qa_sample,
                    content or raw_response,
                    messages,
                    trajectory,
                )

            if action == "search":
                ranked = self._rank_images(content, image_candidates)
                ranked_texts = self._rank_texts(content, text_candidates)
                ranked_jsons = self._rank_texts(content, json_candidates)
                selected = None
                selected_texts: list[dict] = []
                selected_jsons: list[dict] = []
                for item in ranked:
                    ref = item["ref"]
                    if seen_image_refs.get(ref, 0) < self.repeated_nums:
                        selected = item
                        break

                for item in ranked_texts:
                    tid = item["id"]
                    if tid in seen_text_ids:
                        continue
                    selected_texts.append(item)
                    seen_text_ids.add(tid)
                    if len(selected_texts) >= self.vrag_top_k:
                        break

                for item in ranked_jsons:
                    tid = item["id"]
                    if tid in seen_json_ids:
                        continue
                    selected_jsons.append(item)
                    seen_json_ids.add(tid)
                    if len(selected_jsons) >= self.vrag_top_k:
                        break

                user_content: list[dict] = []
                observation_parts: list[str] = []

                if selected_texts:
                    user_content.append(
                        {
                            "type": "text",
                            "text": self._format_text_observation(selected_texts),
                        }
                    )
                    observation_parts.append(
                        "text=" + "; ".join(item["text"] for item in selected_texts)
                    )

                if selected_jsons:
                    user_content.append(
                        {
                            "type": "text",
                            "text": self._format_json_observation(selected_jsons),
                        }
                    )
                    observation_parts.append(
                        "json=" + "; ".join(item["text"] for item in selected_jsons)
                    )

                if selected is not None:
                    try:
                        raw_image = _image_ref_to_pil(selected["ref"])
                        input_image, data_url = self._process_image(raw_image)
                    except Exception as e:
                        selected = None
                        observation_parts.append(f"image_load_failed={e}")
                    else:
                        observed = ObservedImage(
                            source_ref=selected["ref"],
                            raw_image=raw_image,
                            input_image=input_image,
                            context_line=selected["line"],
                        )
                        observed_images.append(observed)
                        seen_image_refs[selected["ref"]] = seen_image_refs.get(selected["ref"], 0) + 1
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                        )
                        observation_parts.append(f"image={selected['line']}")

                if not user_content:
                    trace_item["observation"] = "search_failed"
                    trajectory.append(trace_item)
                    messages.append({"role": "user", "content": VRAG_SEARCH_FAILED_MESSAGE})
                    continue

                trace_item["observation"] = " | ".join(observation_parts)
                trajectory.append(trace_item)
                messages.append({"role": "user", "content": user_content})
                continue

            if action == "bbox" and observed_images:
                cropped, error_message = self._crop_latest_image(observed_images[-1], content)
                if cropped is None:
                    trace_item["observation"] = "invalid_bbox"
                    trajectory.append(trace_item)
                    messages.append({"role": "user", "content": error_message or VRAG_INVALID_ACTION_MESSAGE})
                    continue

                observed_images.append(cropped)
                _, data_url = self._process_image(cropped.raw_image)
                trace_item["observation"] = cropped.context_line
                trajectory.append(trace_item)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                        ],
                    }
                )
                continue

            trace_item["observation"] = "invalid_action"
            trajectory.append(trace_item)
            messages.append({"role": "user", "content": VRAG_INVALID_ACTION_MESSAGE})

        messages.append({"role": "user", "content": VRAG_FORCE_ANSWER_MESSAGE})
        final_response = self._chat(messages)
        messages.append({"role": "assistant", "content": final_response})
        final_action, final_content, final_thought = self._parse_response(final_response)
        trajectory.append(
            {
                "step": self.vrag_max_steps + 1,
                "thought": final_thought,
                "action": final_action,
                "content": final_content,
                "observation": "forced_final_answer",
            }
        )
        final_answer = final_content if final_action == "answer" and final_content else final_response
        return self._finalize_vrag_trajectory(qa_sample, final_answer, messages, trajectory)

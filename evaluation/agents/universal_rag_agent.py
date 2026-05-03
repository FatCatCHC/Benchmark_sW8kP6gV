from __future__ import annotations

import json
import os
import random
import sys
from typing import List, Optional, Tuple

from copy import deepcopy
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import tiktoken

from constant import API_VLMS, VLM_METAINFO_MAP, Image_Root_Dir_Path
from universal_rag_visual_bge import DEFAULT_BGE_M3_MODEL_ID
from .base_rag_agent import BaseRAGAgent
from .prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    MULTIPLE_CHOICE_OUTPUT_FORMAT,
    OVERALL_EVALUATION_SYSTEM_PROMPT,
    UNIVERSAL_RAG_ROUTER_PROMPT,
)
from utils import (
    image_path_to_data_url,
    get_response_universal_rag as get_response,
    build_function_call_messages,
    evaluate_function_call_response,
    load_function_call_candidate_tools,
    messages_to_text_context,
)

from universal_rag_bge import (
    BGECorpusRetriever,
    DEFAULT_BGE_MODEL_ID,
    encode_query,
    load_corpus_meta,
    retrieve_from_passage_list,
    split_text_into_chunks,
)
from universal_rag_visual_bge import BGEVisualImageRetriever, encode_visual_text_query, load_visualized_bge

RETRIEVAL_METHODS = ("no", "text", "json", "image", "error")

def _normalize_route(raw: str | None) -> str:
    if not raw:
        return "error"
    text = raw.strip().lower()
    first = text.split()[0].strip(".,;:\"'") if text else ""
    legacy_first = {"paragraph": "text", "document": "json"}
    if first in legacy_first:
        first = legacy_first[first]
    if first in RETRIEVAL_METHODS and first != "error":
        return first
    for m in ("text", "json", "image"):
        if m in text:
            return m
    if "paragraph" in text:
        return "text"
    if "document" in text:
        return "json"
    if text == "no" or first == "no":
        return "no"
    return "error"

def _evidence_items_to_content(items: list) -> list:
    parts: list = []
    if not isinstance(items, list):
        return parts
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        inner = item.get("content")
        if not isinstance(inner, dict):
            inner = {}
        if itype == "text":
            txt = inner.get("text")
            if txt is not None and str(txt).strip():
                parts.append({"type": "text", "text": str(txt)})
        elif itype == "image":

            raw = inner.get("image_path")
            if raw:
                raw = os.path.join(Image_Root_Dir_Path, raw)
            else:
                raw = ""
            if raw:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path_to_data_url(raw)},
                    }
                )
    return parts

def _filter_messages_text_only(conv_messages: list) -> list:
    out = []
    for m in conv_messages:
        if m.get("role") != "user":
            continue
        if m.get("content_type") == "text":

            new_m = deepcopy(m)
            new_m.pop("content_type", None)
            out.append(new_m)
    return out

def _filter_messages_image_only(conv_messages: list) -> list:
    out = []
    for m in conv_messages:
        if m.get("role") != "user":
            continue
        if m.get("content_type") == "image":
            new_m = deepcopy(m)
            new_m.pop("content_type", None)
            out.append(new_m)
    return out

def _filter_messages_json_only(conv_messages: list) -> list:
    out = []
    for m in conv_messages:
        if m.get("role") != "user":
            continue
        if m.get("content_type") == "json":
            new_m = deepcopy(m)
            new_m.pop("content_type", None)
            raw_content = new_m.get("content")
            flat: list = []
            if isinstance(raw_content, list):
                for part in raw_content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "json" and "json" in part:
                        flat.extend(_evidence_items_to_content(part["json"]))
                    elif part.get("type") == "text" and isinstance(part.get("text"), str):
                        flat.append({"type": "text", "text": part["text"]})
                    else:
                        raise ValueError(f"Invalid content type: {part.get('type')}")
                new_m["content"] = flat
            out.append(new_m)
    return out

class UniversalRAGAgent(BaseRAGAgent):
    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.eval_strategy = args.eval_strategy
        self.eval_format = args.eval_format
        self.router_model = getattr(args, "universal_rag_router_model", "gpt-4.1")
        self.bge_model_path = (getattr(args, "universal_rag_bge_model_path", "") or "").strip() or (os.environ.get("BGE_LARGE_MODEL_PATH", "") or "").strip()
        self.universal_rag_top_k = int(getattr(args, "universal_rag_top_k", 3))
        self._st = None
        self._text_r: Optional[BGECorpusRetriever] = None
        self._json_r: Optional[BGECorpusRetriever] = None
        self._image_r: Optional[BGEVisualImageRetriever] = None
        self._vbge_model = None
        self.universal_rag_visual_bge_weight = os.path.join(os.environ.get("BGE_M3_MODEL_PATH", ""), "Visualized_m3.pth")
        self.universal_rag_visual_bge_backbone = DEFAULT_BGE_M3_MODEL_ID
        self.encoder = tiktoken.encoding_for_model("gpt-4.1")
        self.token_limit = VLM_METAINFO_MAP[args.model].get("token_limit", 108000)
        self._function_call_candidate_tools = load_function_call_candidate_tools()

        text_pkl = getattr(args, "universal_rag_text_pkl", "").strip()
        json_pkl = getattr(args, "universal_rag_json_pkl", "").strip()
        image_pkl = getattr(args, "universal_rag_image_pkl", "").strip()
        meta_path = getattr(args, "universal_rag_corpus_meta", "").strip()

        if meta_path and os.path.isfile(meta_path):
            meta = load_corpus_meta(meta_path)
            if text_pkl and os.path.isfile(text_pkl):
                self._text_r = BGECorpusRetriever(text_pkl, meta["text"])
            if json_pkl and os.path.isfile(json_pkl):
                self._json_r = BGECorpusRetriever(json_pkl, meta["json"])
            if (
                image_pkl
                and os.path.isfile(image_pkl)
                and meta.get("image")
                and len(meta["image"]) > 0
            ):
                self._image_r = BGEVisualImageRetriever(image_pkl, meta["image"])

    def _get_bge(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            mid = self.bge_model_path if self.bge_model_path else DEFAULT_BGE_MODEL_ID
            self._st = SentenceTransformer(mid)
        return self._st

    def _get_visualized_bge(self):
        if self._vbge_model is None:
            if not self.universal_rag_visual_bge_weight or not os.path.isfile(
                self.universal_rag_visual_bge_weight
            ):
                raise FileNotFoundError(
                    "universal_rag_visual_bge_weight must be a valid .pth path when using image.pkl retrieval."
                )
            self._vbge_model = load_visualized_bge(
                self.universal_rag_visual_bge_backbone,
                self.universal_rag_visual_bge_weight,
            )
        return self._vbge_model

    def _bge_retrieve(
        self, retriever: Optional[BGECorpusRetriever], question: str
    ) -> Tuple[List[str], List[str]]:
        if retriever is None or retriever.vectors.shape[0] == 0:
            return [], []
        st = self._get_bge()
        qv = encode_query(st, question)
        return retriever.retrieve(qv, self.universal_rag_top_k)

    def _retrieved_texts_block(self, passages: List[str]) -> str:
        if not passages:
            return ""
        lines = [f"[{i + 1}] {p}" for i, p in enumerate(passages)]
        return "Retrieved passages (BGE top-k):\n" + "\n\n".join(lines)

    def _retrieved_json_block(self, json_evidence: List[str]) -> str:
        if not json_evidence:
            return ""
        lines = [f"[{i + 1}] {j}" for i, j in enumerate(json_evidence)]
        return "Retrieved json evidence (BGE top-k):\n" + "\n\n".join(lines)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(Exception))
    def _route_with_router_model(self, question: str) -> str:
        prompt_text = UNIVERSAL_RAG_ROUTER_PROMPT.format(query=question)
        response = self.client.chat.completions.create(
            model=self.router_model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=16,
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        route = _normalize_route(raw)
        if route == "error":
            route = random.choice(["no", "text", "json", "image"])
        return route

    def _append_multiple_choice_options(self, messages: list, qa_sample: dict) -> None:
        if "multi_choice_QA" not in qa_sample or "multi_choice_QA_options" not in qa_sample["multi_choice_QA"]:
            return
        if self.eval_format != "multiple_choice":
            return
        options = ["(A)", "(B)", "(C)", "(D)"]
        multi_choice_QA_options = qa_sample["multi_choice_QA"]["multi_choice_QA_options"]
        candidate_options = [f"{options[i]}: {opt}" for i, opt in enumerate(multi_choice_QA_options)]
        options_text = "\n".join(candidate_options)
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": f"Here are the candidate options: {options_text}"}]}
        )

    def _corpus_messages_for_rag(self, conversations: list) -> list:

        overall_conversations_message = []
        for conversation in conversations:
            preprocessed_conversation = []

            if conversation["content_type"] == "text":
                content = conversation["content"]
                preprocessed_conversation.append({
                    "role": "user",
                    "content_type": "text",
                    "content": [{"type": "text", "text": f"{conversation['sender_name']} at {conversation['timestamp']} in {conversation['conversation_name']}: {content}"}],
                })

            elif conversation["content_type"] == "image":
                preprocessed_conversation.append({
                    "role": "user",
                    "content_type": "image",
                    "content": [
                        {"type": "text", "text": f"{conversation['sender_name']} sent an image at {conversation['timestamp']} in {conversation['conversation_name']}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_path_to_data_url(os.path.join(Image_Root_Dir_Path, conversation["image_path"]))
                            }
                        }
                    ],
                })
            elif conversation["content_type"] == "json_evidence":
                preprocessed_conversation.append({
                    "role": "user",
                    "content_type": "json",
                    "content": [
                        {"type": "text", "text": f"{conversation['sender_name']} sent a json evidence at {conversation['timestamp']} in {conversation['conversation_name']}"},
                        {
                            "type": "json",
                            "json": conversation["content"],
                        }
                        ],
                })
            else:
                raise ValueError(f"Invalid content type: {conversation['content_type']}")

            overall_conversations_message.extend(preprocessed_conversation)
        return overall_conversations_message

    def _construct_conversation_messages( self, qa_sample: dict, conversations: list, route: str) -> list:

        overall_conversations_message = self._corpus_messages_for_rag(
            conversations,
        )
        system_prompt = OVERALL_EVALUATION_SYSTEM_PROMPT
        if self.eval_format == "multiple_choice":
            system_prompt += MULTIPLE_CHOICE_OUTPUT_FORMAT
        elif self.eval_format == "fill_in_the_blank":
            system_prompt += FILL_IN_THE_BLANK_OUTPUT_FORMAT
        else:
            raise ValueError(f"Invalid evaluation format: {self.eval_format}")

        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        retrieved_texts: List[str] = []
        retrieved_ids: List[str] = []

        if route == "no":

            pass
        elif route == "text":

            retrieved_texts, retrieved_ids = self._bge_retrieve(self._text_r, qa_sample["question"])
            if retrieved_texts:
                block = self._retrieved_texts_block(retrieved_texts)
                messages.append({"role": "user", "content": [{"type": "text", "text": block}]})
            else:
                print("Text Fallback to text messages")
                fallback = _filter_messages_text_only(overall_conversations_message)
                messages.extend(fallback if fallback else overall_conversations_message)

        elif route == "json":

            retrieved_texts, retrieved_ids = self._bge_retrieve(self._json_r, qa_sample["question"])
            if retrieved_texts:
                block = self._retrieved_texts_block(retrieved_texts)
                messages.append({"role": "user", "content": [{"type": "text", "text": block}]})
            else:
                print("Json Fallback to json messages")
                fallback = _filter_messages_json_only(overall_conversations_message)
                messages.extend(fallback if fallback else overall_conversations_message)
        elif route == "image":
            ranked: list[tuple[str, str, str]] = []
            if (
                self._image_r is not None
                and self._image_r.vectors.shape[0] > 0
                and self.universal_rag_visual_bge_weight
                and os.path.isfile(self.universal_rag_visual_bge_weight)
            ):
                try:
                    vb = self._get_visualized_bge()
                    qv = encode_visual_text_query(vb, qa_sample["question"])
                    ranked = self._image_r.retrieve(qv, self.universal_rag_top_k)
                except Exception as e:
                    print(
                        f"[UniversalRAGAgent] Visualized-BGE image retrieve failed: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
            if ranked:
                intro = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Retrieved images (Visualized-BGE top-{len(ranked)}): answer using the following evidence.",
                        }
                    ],
                }
                messages.append(intro)
                for _tid, img_path, line in ranked:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": line},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_path_to_data_url(img_path)},
                                },
                            ],
                        }
                    )
            else:
                img_msgs = _filter_messages_image_only(overall_conversations_message)
                if img_msgs:
                    messages.extend(img_msgs[: self.universal_rag_top_k])
                else:
                    messages.extend(overall_conversations_message)

        else:
            messages.extend(overall_conversations_message)

        messages.append({"role": "user", "content": qa_sample["question"]})
        self._append_multiple_choice_options(messages, qa_sample)
        self._last_retrieved_texts = retrieved_texts
        self._last_retrieved_ids = retrieved_ids
        return messages

    def construct_routed_messages(self, qa_sample: dict, conversations: list, route: str,) -> list:
        self._last_retrieved_texts = []
        self._last_retrieved_ids = []
        return self._construct_conversation_messages(qa_sample, conversations, route,)

    def count_tokens(self, messages):
        total = 0
        for m in messages:
            content = m.get("content", "")

            if isinstance(content, str):
                total += len(self.encoder.encode(content))

            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += len(self.encoder.encode(item.get("text", "")))
                    elif isinstance(item, dict) and item.get("type") == "image":
                        total += 1024 * 4

            elif isinstance(content, dict):
                text = content.get("text", "")
                total += len(self.encoder.encode(text))

            elif isinstance(content, dict) and content.get("type") == "image":
                total += 1024 * 4
            else:
                continue
        return total

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
    ):
        route = self._route_with_router_model(qa_sample["question"])
        messages = self.construct_routed_messages(qa_sample,conversations,route)
        parsed_function_calls: Optional[List[dict]] = None

        if self.encoder is not None:

            input_tokens = self.count_tokens(messages)
            print("input_tokens: ", input_tokens)
            while input_tokens > self.token_limit:
                for i in range(1, len(messages) - 1):
                    if messages[i]["role"] == "user":
                        message_tokens = self.count_tokens([messages[i]])
                        input_tokens -= message_tokens
                        del messages[i]
                        break

        try:
            if qa_sample.get("category") == "Function_Call":
                if self.eval_format == "multiple_choice":
                    context_messages = messages[1:-2]
                else:
                    context_messages = messages[1:-1]
                context = messages_to_text_context(context_messages)
                function_call_messages = build_function_call_messages(
                    qa_sample.get("question", ""),
                    context,
                    self._function_call_candidate_tools,
                )
                raw_response = get_response(self.client, function_call_messages, self.args)
                messages = function_call_messages
                single_qa_result, parsed_function_calls = evaluate_function_call_response(
                    raw_response,
                    qa_sample.get("answer"),
                )
            else:
                raw_response = get_response(self.client, messages, self.args)
                single_qa_result = self._evaluate_answer(str(raw_response), qa_sample)
        except Exception as e:
            raw_response = f"Error: {e}"
            single_qa_result = False
            print(f"{self.__class__.__name__} error: {e}")
        print("raw_response: ", raw_response)

        ground_truth = qa_sample.get("answer")
        if self.eval_format == "multiple_choice" and "multi_choice_QA" in qa_sample and qa_sample.get("category") != "Function_Call":
            ground_truth = ["(A)", "(B)", "(C)", "(D)"][qa_sample["multi_choice_QA"]["multi_choice_QA_answer"]]

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
            "messages": messages,
        }

        return readable_answer, read_message

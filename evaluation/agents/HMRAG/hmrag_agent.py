from __future__ import annotations

import math
import os
import re
import sys
import traceback
from itertools import combinations
from typing import Dict, List

import tiktoken
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from constant import VLM_METAINFO_MAP
from ..base_rag_agent import BaseRAGAgent
from .decompose import QueryDecomposer
from .retrieval_bge import dual_layer_retrieve, load_precomputed_retrievers, messages_to_text_blob
from .web_retrieval import WebRetrieval
from utils import (
    build_function_call_messages,
    evaluate_function_call_response,
    load_function_call_candidate_tools,
)

DEFAULT_HMRAG_EMBEDDING_ROOT = os.environ.get(
    "HMRAG_EMBEDDING_ROOT",
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..",
            "datasets", "HMRAG_embedding",
        )
    ),
)

_OPTION_RE = re.compile(r"\(([A-D])\)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"the answer is\s*\(?([A-D])\)?", re.IGNORECASE)

from ..prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    HMRAG_DECISION_REFINEMENT_CONFLICTING,
    HMRAG_DECISION_REFINEMENT_CONSISTENT,
    HMRAG_EXPERT_SYSTEM_TEMPLATE,
    HMRAG_EXPERT_USER_TEMPLATE,
    HMRAG_FUNCTION_CALL_FORMAT_INSTRUCTION,
    HMRAG_MULTIPLE_CHOICE_FORMAT_INSTRUCTION_TEMPLATE,
    HMRAG_SHORT_PHRASE_FORMAT_INSTRUCTION,
    HMRAG_SUMMARIZE_PROMPT_TEMPLATE,
    MULTIPLE_CHOICE_OUTPUT_FORMAT,
    OVERALL_EVALUATION_SYSTEM_PROMPT,
)
def _safe_tiktoken_encoder(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", _normalize_text(text))

def _extract_option_letter(text: str) -> str | None:
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    matches = _OPTION_RE.findall(text)
    if matches:
        return matches[0].upper()
    token_matches = re.findall(r"\b([A-D])\b", text.upper())
    if len(token_matches) == 1:
        return token_matches[0]
    return None

def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for tok_a in a:
        prev = 0
        for j, tok_b in enumerate(b, start=1):
            cur = dp[j]
            if tok_a == tok_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]

def _rouge_l(summary_a: str, summary_b: str) -> float:
    a = _word_tokens(summary_a)
    b = _word_tokens(summary_b)
    if not a or not b:
        return 0.0
    return _lcs_len(a, b) / max(len(a), len(b))

def _bleu1(reference: str, candidate: str) -> float:
    ref = _word_tokens(reference)
    cand = _word_tokens(candidate)
    if not ref or not cand:
        return 0.0

    ref_counts: Dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    overlap = 0
    used: Dict[str, int] = {}
    for tok in cand:
        used[tok] = used.get(tok, 0) + 1
        if used[tok] <= ref_counts.get(tok, 0):
            overlap += 1

    precision = overlap / len(cand)
    brevity_penalty = math.exp(min(0.0, 1.0 - (len(ref) / max(1, len(cand)))))
    return precision * brevity_penalty

def _pairwise_similarity(summary_a: str, summary_b: str, rouge_weight: float) -> float:
    rouge = _rouge_l(summary_a, summary_b)
    bleu = (_bleu1(summary_a, summary_b) + _bleu1(summary_b, summary_a)) / 2.0
    return rouge_weight * rouge + (1.0 - rouge_weight) * bleu

def _json_evidence_to_message_parts(turn: dict) -> list:
    parts: List[dict] = []
    sender = turn.get("sender_name", "unknown")
    timestamp = turn.get("timestamp", "")
    parts.append(
        {
            "type": "text",
            "text": f"{sender} sent a json evidence document at {timestamp}",
        }
    )

    content = turn.get("content") or []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            inner = item.get("content") or {}
            if item_type == "text":
                txt = inner.get("text")
                if txt:
                    parts.append({"type": "text", "text": str(txt)})
            elif item_type == "image":
                raw = inner.get("image_url")
                if raw:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": raw},
                        }
                    )
    caption = turn.get("caption")
    if caption:
        parts.append({"type": "text", "text": f"Caption: {caption}"})
    return parts

def _turn_dicts_to_messages(turns: list) -> list:
    messages = []
    for conversation in turns:
        content_type = conversation.get("content_type")
        sender = conversation.get("sender_name", "unknown")
        timestamp = conversation.get("timestamp", "")

        if content_type == "text":
            content = conversation.get("content", "")
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"{sender} at {timestamp}: {content}"}],
                }
            )
        elif content_type == "image":
            image_url = conversation.get("image_url")
            content_parts: List[dict] = [{"type": "text", "text": f"{sender} sent an image at {timestamp}"}]
            if image_url:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )
            caption = conversation.get("caption")
            if caption:
                content_parts.append({"type": "text", "text": f"Caption: {caption}"})
            messages.append({"role": "user", "content": content_parts})
        elif content_type == "json_evidence":
            messages.append({"role": "user", "content": _json_evidence_to_message_parts(conversation)})
        else:
            content = conversation.get("content", "")
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"{sender} at {timestamp}: {content}"}],
                }
            )
    return messages

def _hmrag_system_prompt_for_eval(eval_format: str) -> str:
    system_prompt = OVERALL_EVALUATION_SYSTEM_PROMPT
    if eval_format == "multiple_choice":
        system_prompt += MULTIPLE_CHOICE_OUTPUT_FORMAT
    elif eval_format == "fill_in_the_blank":
        system_prompt += FILL_IN_THE_BLANK_OUTPUT_FORMAT
    else:
        raise ValueError(f"Invalid evaluation format: {eval_format}")
    return system_prompt

def _hmrag_question_tail_messages(qa_sample: dict, eval_format: str) -> list:
    messages: List[dict] = [{"role": "user", "content": qa_sample.get("question", "")}]
    if eval_format == "multiple_choice":
        mc = qa_sample.get("multi_choice_QA") or {}
        raw_options = mc.get("multi_choice_QA_options")
        if not isinstance(raw_options, list) or not raw_options:
            return messages
        options = ["(A)", "(B)", "(C)", "(D)"]
        candidate_options = [f"{options[index]}: {option}" for index, option in enumerate(raw_options)]
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "Here are the candidate options: " + "\n".join(candidate_options)}],
            }
        )
    return messages

def _construct_hmrag_messages(
    qa_sample: dict,
    conversations: list,
    eval_format: str,
) -> list:
    return [
        {"role": "system", "content": _hmrag_system_prompt_for_eval(eval_format)},
        *_turn_dicts_to_messages(conversations),
        *_hmrag_question_tail_messages(qa_sample, eval_format),
    ]

def _construct_hmrag_messages_without_conversation(qa_sample: dict, eval_format: str) -> list:
    return [
        {"role": "system", "content": _hmrag_system_prompt_for_eval(eval_format)},
        *_hmrag_question_tail_messages(qa_sample, eval_format),
    ]

class HMRAGAgent(BaseRAGAgent):
    def __init__(self, client, args):
        super().__init__(args)
        self.client = client
        self.args = args
        self._max_tokens = args.max_tokens
        self.eval_strategy = args.eval_strategy
        self.eval_format = args.eval_format
        self.model = args.model
        self.encoder = _safe_tiktoken_encoder(args.model)
        self.token_limit = VLM_METAINFO_MAP[args.model]["token_limit"]

        decomp_model = args.model
        aux_model = args.model
        self.aux_model = aux_model
        self.decomposer = QueryDecomposer(
            self.client,
            decomp_model,
            temperature=float(getattr(args, "hmrag_decompose_temperature", 0.0)),
        )

        serper_key = (getattr(args, "hmrag_serper_api_key", "") or "").strip()
        self.web = WebRetrieval(
            self.client,
            summarize_model=aux_model,
            api_key=serper_key,
            num_results=int(getattr(args, "hmrag_serper_num", 5)),
        )

        self.hmrag_vector_top_k = int(getattr(args, "hmrag_vector_top_k", 4))
        self.hmrag_graph_top_k = int(getattr(args, "hmrag_graph_top_k", 4))
        self.hmrag_vector_chunk_tokens = int(getattr(args, "hmrag_vector_chunk_tokens", 256))
        self.hmrag_graph_chunk_tokens = int(getattr(args, "hmrag_graph_chunk_tokens", 512))
        self.hmrag_consistency_threshold = float(getattr(args, "hmrag_consistency_threshold", 0.55))
        self.hmrag_vote_rouge_weight = float(getattr(args, "hmrag_vote_rouge_weight", 0.6))
        self.hmrag_expert_max_tokens = int(
            getattr(args, "hmrag_expert_max_tokens", min(max(args.max_tokens, 256), 1024))
        )
        self.hmrag_summary_max_tokens = int(getattr(args, "hmrag_summary_max_tokens", 96))
        self._function_call_candidate_tools = load_function_call_candidate_tools()

        self._bge = SentenceTransformer(os.environ.get("BGE_LARGE_MODEL_PATH", ""))
        self._vector_retriever = None
        self._graph_retriever = None
        self._load_precomputed_retrievers()

    def _infer_hmrag_index_paths(self) -> tuple[str, str, str]:
        dataset_dir = (getattr(self.args, "dataset_dir_path", "") or "").strip()
        if not dataset_dir:
            return "", "", ""

        cluster_name = os.path.basename(os.path.normpath(dataset_dir))
        dataset_name = os.path.basename(os.path.dirname(os.path.normpath(dataset_dir)))
        embedding_root = (
            getattr(self.args, "hmrag_embedding_root", "") or DEFAULT_HMRAG_EMBEDDING_ROOT
        ).strip()
        base_dir = os.path.join(embedding_root, dataset_name, cluster_name)
        return (
            os.path.join(base_dir, "vector.pkl"),
            os.path.join(base_dir, "graph.pkl"),
            os.path.join(base_dir, "hmrag_meta.json"),
        )

    def _load_precomputed_retrievers(self) -> None:
        vector_pkl = (getattr(self.args, "hmrag_vector_pkl", "") or "").strip()
        graph_pkl = (getattr(self.args, "hmrag_graph_pkl", "") or "").strip()
        meta_path = (getattr(self.args, "hmrag_corpus_meta", "") or "").strip()

        if not (vector_pkl or graph_pkl or meta_path):
            vector_pkl, graph_pkl, meta_path = self._infer_hmrag_index_paths()

        self._vector_retriever, self._graph_retriever = load_precomputed_retrievers(
            vector_pkl,
            graph_pkl,
            meta_path,
        )

    def _failed_qa_pair(self, qa_sample: dict, error_log: str):
        question = qa_sample.get("question", "")
        readable_answer = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": question,
            "answer": qa_sample.get("answer"),
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": error_log,
            "single_qa_result": False,
        }
        read_message = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": question,
            "answer": qa_sample.get("answer"),
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": error_log,
            "single_qa_result": False,
            "messages": error_log,
        }
        return readable_answer, read_message

    def corpus_messages_for_rag(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
    ) -> list:
        del qa_sample, conversation_streams
        return _turn_dicts_to_messages(conversations)

    def construct_messages(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
    ) -> list:
        del conversation_streams
        return _construct_hmrag_messages(
            qa_sample,
            conversations,
            self.eval_format,
        )

    def _context_messages_for_retrieval(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None,
    ) -> str:
        del qa_sample
        del conversation_streams
        ctx_messages = self.corpus_messages_for_rag(
            {},
            conversations,
        )
        return messages_to_text_blob(ctx_messages)

    def _is_usable_web_block(self, web_block: str) -> bool:
        if not web_block:
            return False
        normalized = web_block.strip().lower()
        if not normalized:
            return False
        bad_prefixes = (
            "(web retrieval skipped:",
            "(web retrieval failed:",
            "no relevant results found",
        )
        return not normalized.startswith(bad_prefixes)

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
            elif isinstance(content, dict):
                total += len(self.encoder.encode(content.get("text", "")))
        return total

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(30), retry=retry_if_exception_type(Exception))
    def _chat_text(
        self,
        model: str,
        messages: list,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def _answer_format_instruction(self, qa_sample: dict) -> str:
        if qa_sample.get("category") == "Function_Call":
            return HMRAG_FUNCTION_CALL_FORMAT_INSTRUCTION
        if self.eval_format == "multiple_choice":
            mc = qa_sample.get("multi_choice_QA") or {}
            raw_options = mc.get("multi_choice_QA_options")
            if not isinstance(raw_options, list) or not raw_options:
                return HMRAG_SHORT_PHRASE_FORMAT_INSTRUCTION
            labels = ["(A)", "(B)", "(C)", "(D)"]
            lines = [f"{labels[i]} {opt}" for i, opt in enumerate(raw_options)]
            return HMRAG_MULTIPLE_CHOICE_FORMAT_INSTRUCTION_TEMPLATE.format(
                candidate_options="\n".join(lines)
            )
        return HMRAG_SHORT_PHRASE_FORMAT_INSTRUCTION

    def _expert_answer(self, expert_name: str, search_q: str, evidence_block: str, qa_sample: dict) -> str:
        system = HMRAG_EXPERT_SYSTEM_TEMPLATE.format(expert_name=expert_name)
        user = HMRAG_EXPERT_USER_TEMPLATE.format(
            search_q=search_q,
            question=qa_sample.get("question", ""),
            format_instruction=self._answer_format_instruction(qa_sample),
            expert_name=expert_name,
            evidence_block=evidence_block,
        )
        try:
            return self._chat_text(
                self.aux_model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.hmrag_expert_max_tokens,
                temperature=0.0,
            )
        except Exception as e:
            return f"(Expert {expert_name} failed: {e})"

    def _summarize_answer(self, expert_name: str, answer: str, qa_sample: dict) -> str:
        letter = _extract_option_letter(answer)
        if self.eval_format == "multiple_choice" and letter:
            return f"Selected option {letter}."

        prompt = HMRAG_SUMMARIZE_PROMPT_TEMPLATE.format(
            expert_name=expert_name,
            question=qa_sample.get("question", ""),
            answer=answer,
        )
        try:
            return self._chat_text(
                self.aux_model,
                [{"role": "user", "content": prompt}],
                max_tokens=self.hmrag_summary_max_tokens,
                temperature=0.0,
            )
        except Exception:
            return answer.strip()

    def _consistency_vote(self, expert_answers: Dict[str, str], qa_sample: dict) -> dict:
        summaries = {
            name: self._summarize_answer(name, answer, qa_sample)
            for name, answer in expert_answers.items()
        }

        pair_scores = []
        for left, right in combinations(expert_answers.keys(), 2):
            left_letter = _extract_option_letter(expert_answers[left])
            right_letter = _extract_option_letter(expert_answers[right])
            if left_letter and right_letter and left_letter == right_letter:
                score = 1.0
                rouge = 1.0
                bleu = 1.0
            else:
                rouge = _rouge_l(summaries[left], summaries[right])
                bleu = (_bleu1(summaries[left], summaries[right]) + _bleu1(summaries[right], summaries[left])) / 2.0
                score = _pairwise_similarity(
                    summaries[left],
                    summaries[right],
                    self.hmrag_vote_rouge_weight,
                )

            pair_scores.append(
                {
                    "pair": [left, right],
                    "score": score,
                    "rouge_l": rouge,
                    "bleu": bleu,
                }
            )

        consistent_pairs = [p for p in pair_scores if p["score"] >= self.hmrag_consistency_threshold]
        if consistent_pairs:
            best_pair = max(consistent_pairs, key=lambda item: item["score"])
            consensus = set(best_pair["pair"])
            for other in expert_answers.keys():
                if other in consensus:
                    continue
                for pair_info in consistent_pairs:
                    if other in pair_info["pair"] and any(name in consensus for name in pair_info["pair"]):
                        consensus.add(other)
                        break
            route = "consistency_vote"
        else:
            consensus = set()
            route = "expert_model_refinement"

        return {
            "route": route,
            "summaries": summaries,
            "pair_scores": pair_scores,
            "consensus_experts": sorted(consensus),
        }

    def _decision_prompt(
        self,
        qa_sample: dict,
        search_q: str,
        decomposed_query: str | List[str],
        expert_blocks: Dict[str, str],
        expert_answers: Dict[str, str],
        vote_info: dict,
    ) -> str:
        if isinstance(decomposed_query, list):
            sub_queries = "\n".join(f"- {q}" for q in decomposed_query)
        else:
            sub_queries = f"- {decomposed_query}"

        pair_lines = []
        for item in vote_info["pair_scores"]:
            pair_lines.append(
                f"{item['pair'][0]} vs {item['pair'][1]}: "
                f"score={item['score']:.3f}, rouge_l={item['rouge_l']:.3f}, bleu={item['bleu']:.3f}"
            )
        pair_text = "\n".join(pair_lines) if pair_lines else "(No pairwise scores.)"

        sections = [
            "HM-RAG decision agent context.",
            f"Original question:\n{qa_sample.get('question', '')}",
            f"Decomposed sub-queries:\n{sub_queries}",
            f"Search string for retrieval:\n{search_q}",
            f"Decision route: {vote_info['route']}",
            f"Consensus experts: {', '.join(vote_info['consensus_experts']) if vote_info['consensus_experts'] else '(none)'}",
            f"Pairwise consistency scores:\n{pair_text}",
        ]

        for name in expert_blocks.keys():
            sections.append(
                f"{name.title()} expert summary:\n{vote_info['summaries'].get(name, '')}\n\n"
                f"{name.title()} expert answer:\n{expert_answers.get(name, '')}\n\n"
                f"{name.title()} expert evidence:\n{expert_blocks.get(name, '')}"
            )

        if vote_info["route"] == "consistency_vote":
            sections.append(HMRAG_DECISION_REFINEMENT_CONSISTENT)
        else:
            sections.append(HMRAG_DECISION_REFINEMENT_CONFLICTING)

        sections.append(self._answer_format_instruction(qa_sample))
        return "\n\n".join(sections)

    def _truncate_if_needed(self, messages: list) -> list:
        if self.encoder is None:
            return messages
        input_tokens = self.count_tokens(messages)
        print("input_tokens: ", input_tokens)
        while input_tokens > self.token_limit:
            removed = False
            for i in range(1, len(messages) - 1):
                if messages[i]["role"] == "user":
                    message_tokens = self.count_tokens([messages[i]])
                    input_tokens -= message_tokens
                    del messages[i]
                    removed = True
                    break
            if not removed:
                break
        return messages

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60), retry=retry_if_exception_type(Exception))
    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
    ):
        question = qa_sample.get("question", "")
        decomposed_query = self.decomposer.decompose(question)
        if isinstance(decomposed_query, list):
            search_q = " ".join(decomposed_query)
        else:
            search_q = decomposed_query

        corpus_blob = self._context_messages_for_retrieval(
            qa_sample,
            conversations,
            conversation_streams,
        )
        vec_block, graph_block = dual_layer_retrieve(
            self._bge,
            search_q,
            corpus_blob,
            vector_chunk_tokens=self.hmrag_vector_chunk_tokens,
            graph_chunk_tokens=self.hmrag_graph_chunk_tokens,
            vector_top_k=self.hmrag_vector_top_k,
            graph_top_k=self.hmrag_graph_top_k,
            vector_retriever=self._vector_retriever,
            graph_retriever=self._graph_retriever,
        )
        web_block = self.web.find_top_k(search_q)

        expert_blocks = {
            "vector": vec_block,
            "graph": graph_block,
        }
        if self._is_usable_web_block(web_block):
            expert_blocks["web"] = "Web expert summary:\n" + web_block
        expert_answers = {
            name: self._expert_answer(name, search_q, block, qa_sample)
            for name, block in expert_blocks.items()
        }
        vote_info = self._consistency_vote(expert_answers, qa_sample)

        decision_prompt = self._decision_prompt(
            qa_sample,
            search_q,
            decomposed_query,
            expert_blocks,
            expert_answers,
            vote_info,
        )

        messages = _construct_hmrag_messages_without_conversation(qa_sample, self.eval_format)
        parsed_function_calls = None
        if qa_sample.get("category") == "Function_Call":
            function_call_context = "\n\n".join(
                block.strip()
                for block in expert_blocks.values()
                if isinstance(block, str) and block.strip()
            )
            messages = build_function_call_messages(
                qa_sample.get("question", ""),
                function_call_context,
                self._function_call_candidate_tools,
            )
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": decision_prompt}]})
            messages = self._truncate_if_needed(messages)

        try:
            raw_response = self._chat_text(
                self.model,
                messages,
                max_tokens=self.args.max_tokens,
                temperature=0.0,
            )
        except Exception as e:
            raw_response = f"Error: {e}"
            print(f"{self.__class__.__name__} error: {e}")
        print("raw_response: ", raw_response)
        if qa_sample.get("category") == "Function_Call":
            single_qa_result, parsed_function_calls = evaluate_function_call_response(
                raw_response,
                qa_sample.get("answer"),
            )
        else:
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

        hmrag_debug = {
            "decomposed_query": decomposed_query,
            "search_query": search_q,
            "expert_blocks": expert_blocks,
            "expert_answers": expert_answers,
            "vote_info": vote_info,
        }

        readable_answer = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": qa_sample.get("question"),
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": raw_response,
            "parsed_function_calls": parsed_function_calls,
            "single_qa_result": single_qa_result,
            "hmrag_debug": hmrag_debug,
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
            "hmrag_debug": hmrag_debug,
        }

        return readable_answer, read_message

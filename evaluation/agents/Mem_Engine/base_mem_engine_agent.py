import hashlib
import json
import os
import sys
from copy import deepcopy
from typing import Optional, List, Any, Dict, Type, Tuple, cast
from tqdm import tqdm

from ..base_mem_agent import BaseMemoryAgent
from ..prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    MEM_ENGINE_JSON_FALLBACK_SYSTEM_PROMPT,
    MEM_ENGINE_OVERALL_EVALUATION_SYSTEM_PROMPT,
    MEM_ENGINE_USER_PROMPT_TEMPLATE,
    MEM_ENGINE_USER_TAIL_INSTRUCTION,
    MULTIPLE_CHOICE_OUTPUT_FORMAT,
)
from utils import (
    get_response,
    build_function_call_messages,
    evaluate_function_call_response,
    extract_json_payload,
    format_function_call_candidate_tools,
    load_function_call_candidate_tools,
    normalize_function_call_answer,
)

MEM_ENGINE_CKPT_STATE = "mem_engine_ckpt_state.json"
MEM_ENGINE_CKPT_TEXTS = "mem_engine_turn_texts.jsonl"

def _ensure_mem_engine_path():

    workspace_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    mem_engine_path = os.path.join(workspace_root, "baselines", "Mem_Engine", "MemEngine")
    if mem_engine_path not in sys.path:
        sys.path.insert(0, mem_engine_path)

def _conversation_turn_to_memory_text(turn: dict, client, args) -> Optional[str]:
    content_type = turn.get("content_type", "text")
    sender_name = turn.get("sender_name", "Speaker")
    content = turn.get("content", "")
    conversation_name = turn.get("conversation_name", "")

    if content_type == "text":
        return f"Speaker {sender_name} says: {content} in conversation {conversation_name}" if content else None
    elif content_type == "image":
        caption = turn.get("caption", "")
        caption_content = f"Speaker {sender_name} sends an image in conversation {conversation_name}, here is the caption: {caption}"
        return caption_content
    elif content_type == "json_evidence":
        caption = turn.get("caption", "")
        caption_content = f"Speaker {sender_name} sends a document in conversation {conversation_name}, here is the caption: {caption}"
        return caption_content
    return None

def _mem_engine_ingest_fingerprint(ingest_list: List[dict], upto_index: int, eval_strategy: str) -> str:
    sig = {
        "eval_strategy": eval_strategy,
        "n": len(ingest_list),
        "upto_index": upto_index,
        "turns": [
            (
                i,
                t.get("content_type"),
                t.get("timestamp"),
                (t.get("content") or "")[:400] if isinstance(t.get("content"), str) else str(t.get("content"))[:400],
            )
            for i, t in enumerate(ingest_list)
        ],
    }
    blob = json.dumps(sig, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()

class BaseMemEngineAgent(BaseMemoryAgent):
    def __init__(self, client, args):
        super().__init__(args)
        self.memory_class = None
        self.memory_config = None
        self.client = client
        self._memory = None
        self._last_added_turn_index = -1
        self._eval_strategy = getattr(args, "eval_strategy", "only_qa_context")
        self._eval_format = getattr(args, "eval_format", "multiple_choice")
        self._overall_cluster_ingest_done = False
        self._function_call_candidate_tools = self._load_function_call_candidate_tools()
        self.counter = 0

    def _mem_engine_ckpt_dir(self) -> Optional[str]:
        p = getattr(self.args, "mem_engine_checkpoint_dir_path", None)
        assert p is not None, "mem_engine_checkpoint_dir_path is not set"
        return os.path.abspath(os.path.expanduser(str(p).strip()))

    def _mem_engine_ckpt_paths(self, ck_dir: str) -> Tuple[str, str]:
        return (
            os.path.join(ck_dir, MEM_ENGINE_CKPT_STATE),
            os.path.join(ck_dir, MEM_ENGINE_CKPT_TEXTS),
        )

    def _mem_engine_clear_checkpoint_if_mismatch(self, ingest_list: list, upto_index: int) -> None:
        ck = self._mem_engine_ckpt_dir()
        assert ck is not None, "mem_engine_checkpoint_dir_path is not set"
        state_p, jsonl_p = self._mem_engine_ckpt_paths(ck)
        if not os.path.isfile(state_p) or not os.path.isfile(jsonl_p):
            return
        with open(state_p, "r", encoding="utf-8") as f:
            state = json.load(f)
        fp = _mem_engine_ingest_fingerprint(ingest_list, upto_index, self._eval_strategy)
        if state.get("fingerprint") != fp or int(state.get("upto_index", -1)) != int(upto_index):
            for p in (jsonl_p, state_p):
                if os.path.isfile(p):
                    os.remove(p)
            print("[mem_engine checkpoint] fingerprint or target changed; cleared on-disk cache")

    def _mem_engine_resume(self, ingest_list: list, upto_index: int) -> None:
        ck = self._mem_engine_ckpt_dir()
        if not ck or self._memory is None:
            return
        state_p, jsonl_p = self._mem_engine_ckpt_paths(ck)
        if not os.path.isfile(state_p) or not os.path.isfile(jsonl_p):
            return
        with open(state_p, "r", encoding="utf-8") as f:
            state = json.load(f)
        fp = _mem_engine_ingest_fingerprint(ingest_list, upto_index, self._eval_strategy)
        if state.get("fingerprint") != fp or int(state.get("upto_index", -1)) != int(upto_index):
            return
        if state.get("eval_strategy") != self._eval_strategy:
            return
        last = int(state.get("last_added_turn_index", -1))
        if last < 0:
            return
        by_i: dict = {}
        with open(jsonl_p, "r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                by_i[int(o["i"])] = o
        for i in range(0, min(last, len(ingest_list) - 1, upto_index) + 1):
            rec = by_i.get(i)
            if not rec:
                continue
            text = rec.get("text") or ""
            if text:
                self._store_turn_to_memory(text, rec.get("time"))
        self._last_added_turn_index = min(last, upto_index, len(ingest_list) - 1)
        print(
            f"[mem_engine checkpoint] resumed dir={ck} last_turn={self._last_added_turn_index} "
            f"mem_replay_lines={len(by_i)}"
        )

    def _mem_engine_write_checkpoint(
        self,
        ingest_list: list,
        upto_index: int,
        turn_index: int,
        text: Optional[str],
        time: Optional[Any] = None,
    ) -> None:
        ck = self._mem_engine_ckpt_dir()
        if not ck:
            return
        os.makedirs(ck, exist_ok=True)
        _, jsonl_p = self._mem_engine_ckpt_paths(ck)
        row: dict = {"i": turn_index, "text": text or ""}
        if time is not None:
            row["time"] = time
        with open(jsonl_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fp = _mem_engine_ingest_fingerprint(ingest_list, upto_index, self._eval_strategy)
        state_p, _ = self._mem_engine_ckpt_paths(ck)
        with open(state_p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fp,
                    "upto_index": upto_index,
                    "last_added_turn_index": turn_index,
                    "eval_strategy": self._eval_strategy,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def build_memory(
        self,
        conversations: list,
        conversation_streams: Optional[list],
        qa_samples: List[dict],
    ) -> None:
        self._overall_cluster_ingest_done = False
        assert self._eval_strategy in ("only_qa_context", "time_sequential_memory", "file_sequential_memory"), "Invalid evaluation strategy"
        self._reset_memory_state()
        self._init_memory()
        upto = len(conversations) - 1
        self._add_turns_to_memory(conversations, upto)
        self._overall_cluster_ingest_done = True

    def _init_memory(self):

        if self._memory is not None:
            return
        assert self.memory_class is not None, "Subclass must set memory_class"
        assert self.memory_config is not None, "Subclass must set memory_config"
        _ensure_mem_engine_path()
        from memengine import MemoryConfig
        mem_cfg_dict = deepcopy(self.memory_config)
        mem_engine_gpu = getattr(cast(Any, self.args), "mem_engine_gpu", None)
        if mem_engine_gpu is not None:
            if "global_config" not in mem_cfg_dict:
                mem_cfg_dict["global_config"] = {}
            elif not isinstance(mem_cfg_dict["global_config"], dict):
                mem_cfg_dict["global_config"] = dict(mem_cfg_dict["global_config"])
            mem_cfg_dict["global_config"]["usable_gpu"] = str(mem_engine_gpu)
        mem_cfg = MemoryConfig(mem_cfg_dict)
        self._memory = self.memory_class(mem_cfg)

    def _store_turn_to_memory(self, text: Optional[str], timestamp: Optional[Any] = None) -> None:
        if not text:
            return
        assert self._memory is not None, "Memory not initialized; call _init_memory first"
        payload: dict[str, Any] = {"text": text}
        if timestamp is not None:
            payload["time"] = timestamp
        self._memory.store(payload)

    def _add_turns_to_memory(self, conversations: list, upto_index: int):
        assert self._memory is not None, "Memory not initialized; call _init_memory first"

        use_ckpt = self._mem_engine_ckpt_dir() is not None
        if use_ckpt:
            self._mem_engine_clear_checkpoint_if_mismatch(conversations, upto_index)
            self._mem_engine_resume(conversations, upto_index)

        for i in tqdm(
            range(self._last_added_turn_index + 1, min(upto_index + 1, len(conversations))),
            desc="Adding turns to memory",
        ):
            turn = conversations[i]
            text = _conversation_turn_to_memory_text(turn, self.client, self.args)
            ts = turn.get("timestamp")
            if text:
                self._store_turn_to_memory(text, ts)
            if use_ckpt:
                self._mem_engine_write_checkpoint(conversations, upto_index, i, text, ts)
        self._last_added_turn_index = upto_index

    def _reset_memory_state(self):
        self._last_added_turn_index = -1
        self._memory = None
        self._overall_cluster_ingest_done = False

    def _recall_for_question(self, question: str) -> str:
        assert self._memory is not None, "Memory not initialized; call _init_memory first"
        result = self._memory.recall(question)
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            items = cast(List[Any], result)
            return "\n".join(str(x) for x in items)
        return str(result)

    def _load_function_call_candidate_tools(self) -> List[dict]:
        return load_function_call_candidate_tools()

    def _format_function_call_candidate_tools(self) -> str:
        return format_function_call_candidate_tools(self._function_call_candidate_tools)

    def _response_to_text(self, response: Any) -> str:
        if isinstance(response, tuple):
            response = response[0]
        return response if isinstance(response, str) else str(response)

    def _answer_function_call_question_with_context(self, qa_sample: dict, context: str) -> Tuple[str, List[dict]]:
        question = qa_sample.get("question", "")
        messages = build_function_call_messages(
            question,
            context,
            self._function_call_candidate_tools,
        )
        response = self._response_to_text(get_response(self.client, messages, self.args))
        return response, messages

    def _extract_json_payload(self, raw_response: Any) -> Optional[Any]:
        return extract_json_payload(raw_response)

    def _normalize_function_call_answer(self, answer: Any) -> Optional[List[dict]]:
        return normalize_function_call_answer(answer)

    def _evaluate_function_call_response(self, raw_response: str, qa_sample: dict) -> Tuple[bool, Optional[List[dict]]]:
        return evaluate_function_call_response(raw_response, qa_sample.get("answer"))

    def _answer_question_with_context(self, qa_sample: dict, context: str, eval_format: str) -> Tuple[str, List[dict]]:
        question = qa_sample.get("question", "")
        multi_choice_QA = qa_sample.get("multi_choice_QA", {})
        multi_choice_QA_options = multi_choice_QA.get("multi_choice_QA_options", [])
        option_letters = ["(A)", "(B)", "(C)", "(D)"]
        option_letters_and_options = [f"{option_letters[index]}: {option}" for index, option in enumerate(multi_choice_QA_options)]
        option_letters_and_options_text = "\n".join(option_letters_and_options)

        user_prompt = MEM_ENGINE_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        system_prompt = MEM_ENGINE_OVERALL_EVALUATION_SYSTEM_PROMPT
        if eval_format == "multiple_choice":
            system_prompt += MULTIPLE_CHOICE_OUTPUT_FORMAT
            user_prompt += f"Candidate Options:\n {option_letters_and_options_text}"
        elif eval_format == "fill_in_the_blank":
            system_prompt += FILL_IN_THE_BLANK_OUTPUT_FORMAT
        else:
            raise ValueError(f"Invalid evaluation format: {eval_format}")

        user_prompt += MEM_ENGINE_USER_TAIL_INSTRUCTION

        try:
            messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            response = self._response_to_text(get_response(self.client, messages, self.args))
        except TypeError:

            messages = [
                    {"role": "system", "content": MEM_ENGINE_JSON_FALLBACK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            response = self._response_to_text(get_response(self.client, messages, self.args))
        try:
            parsed = json.loads(response)
            return parsed.get("answer", response), messages
        except (json.JSONDecodeError, TypeError):
            return response if isinstance(response, str) else str(response), messages

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: Optional[list] = None,
    ) -> Tuple[dict, dict]:
        if not self._overall_cluster_ingest_done or self._memory is None:
            raise RuntimeError(
                "BaseMemEngineAgent.evaluate_single_qa requires build_memory() first (via evaluate_cluster)"
            )
        self._init_memory()
        self.counter = 0

        question = qa_sample.get("question", "")
        messages: List[dict] = []
        parsed_function_calls: Optional[List[dict]] = None
        context = None
        try:
            context = self._recall_for_question(question)
            if qa_sample.get("category") == "Function_Call":
                raw_response, messages = self._answer_function_call_question_with_context(qa_sample, context)
                single_qa_result, parsed_function_calls = self._evaluate_function_call_response(raw_response, qa_sample)
            else:
                raw_response, messages = self._answer_question_with_context(qa_sample, context, self._eval_format)
                single_qa_result = self._evaluate_answer(raw_response, qa_sample)
        except Exception as e:
            raw_response = f"Error: {e}"
            single_qa_result = False
            print(f"{self.__class__.__name__} error: {e}")
        print(raw_response)
        ground_truth = qa_sample.get("answer")
        if self._eval_format == "multiple_choice" and "multi_choice_QA" in qa_sample and qa_sample.get("category") != "Function_Call":
            ground_truth = ["(A)", "(B)", "(C)", "(D)"][qa_sample["multi_choice_QA"]["multi_choice_QA_answer"]]

        readable_answer = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": question,
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": raw_response,
            "parsed_function_calls": parsed_function_calls,
            "single_qa_result": single_qa_result,
        }
        read_message = {
            "id": qa_sample.get("id"),
            "category": qa_sample.get("category"),
            "question": question,
            "answer": ground_truth,
            "multi_choice_QA": qa_sample.get("multi_choice_QA"),
            "raw_response": raw_response,
            "parsed_function_calls": parsed_function_calls,
            "single_qa_result": single_qa_result,
            "messages": messages,
            "recall_context": context,
        }

        return readable_answer, read_message

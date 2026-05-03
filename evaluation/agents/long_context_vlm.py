from typing import Any, List, Optional, Tuple

from .base import BaseAgent
from utils import check_answer_match_multiple_choice, image_path_to_data_url
from dataset_loader import build_memory_time_ordered_turns
from constant import API_VLMS, VLM_METAINFO_MAP
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import tiktoken
import json

from .prompt import (
    FILL_IN_THE_BLANK_OUTPUT_FORMAT,
    MULTIPLE_CHOICE_OUTPUT_FORMAT,
    ONLY_QAS_EVALUATION_SYSTEM_PROMPT,
    OVERALL_EVALUATION_SYSTEM_PROMPT,
)
def turn_dicts_to_messages(turns: list) -> list:

    overall_conversations = []
    for conversation in turns:
        preprocessed_conversation = []

        if conversation["content_type"] == "text":
            content = conversation["content"]
            preprocessed_conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": f"{conversation['sender_name']} at {conversation['timestamp']}: {content}"}]
            })

        elif conversation["content_type"] == "image":
            preprocessed_conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{conversation['sender_name']} sent an image at {conversation['timestamp']}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": conversation["image_url"]
                        }
                    }
                ]
            })
        else:
            raise ValueError(f"Invalid content type: {conversation['content_type']}")

        overall_conversations.extend(preprocessed_conversation)
    return overall_conversations

def preprocess_conversation_messages(
    conversations: list,
    eval_strategy: str,
    conversation_index: int = -1,
    conversation_streams: list | None = None,
    conversation_file_order_streams: list | None = None,
) -> list:
    if eval_strategy == "only_qas":
        return []

    if eval_strategy in ("memory_time", "time_sequential_RAG"):
        streams_anchor = (
            conversation_file_order_streams
            if conversation_file_order_streams is not None
            else conversation_streams
        )
        if not streams_anchor:
            raise ValueError(
                f"eval_strategy={eval_strategy!r} requires conversation_file_order_streams "
                "(time-sequential load) or multi-file conversation_streams for anchor resolution"
            )

        idx = conversation_index
        if idx < 0:
            idx = max(0, len(conversations) - 1)
        ordered_turns = build_memory_time_ordered_turns(streams_anchor, idx)
        return turn_dicts_to_messages(ordered_turns)

    overall_conversations = turn_dicts_to_messages(conversations)
    if eval_strategy in ("overall", "overall_context", "memory_sequential_overall_eval"):
        return overall_conversations

    assert conversation_index != -1, "conversation_index must be provided when eval_strategy is not overall"
    n_turns = len(conversations)
    last_turn = min(conversation_index, n_turns - 1) if n_turns else -1
    if last_turn < 0:
        return []
    return turn_dicts_to_messages(conversations[: last_turn + 1])

def construct_vlm_messages(
    qa_sample: dict,
    conversations: list,
    eval_strategy: str,
    eval_format: str,
    conversation_streams: list | None = None,
    conversation_file_order_streams: list | None = None,
) -> list:
    if eval_strategy in (
        "overall",
        "overall_context",
        "only_context",
        "only_qa_context",
        "memory_sequential_overall_eval",
        "memory_time",
        "memory_sequential",
        "file_sequential_RAG",
        "time_sequential_RAG",
    ):
        system_prompt = OVERALL_EVALUATION_SYSTEM_PROMPT
    elif eval_strategy == "only_qas":
        system_prompt = ONLY_QAS_EVALUATION_SYSTEM_PROMPT
    else:
        raise ValueError(f"Invalid evaluation strategy: {eval_strategy}")

    if eval_format == "multiple_choice":
        system_prompt += MULTIPLE_CHOICE_OUTPUT_FORMAT
    elif eval_format == "fill_in_the_blank":
        system_prompt += FILL_IN_THE_BLANK_OUTPUT_FORMAT
    else:
        raise ValueError(f"Invalid evaluation format: {eval_format}")

    conv_idx = qa_sample.get("conversation_turn_index", -1)
    if eval_strategy in (
        "only_context",
        "only_qa_context",
        "memory_sequential",
        "file_sequential_RAG",
        "memory_time",
        "time_sequential_RAG",
    ) and conv_idx < 0:
        conv_idx = max(0, len(conversations) - 1)
    overall_conversations = preprocess_conversation_messages(
        conversations,
        eval_strategy,
        conv_idx,
        conversation_streams=conversation_streams,
        conversation_file_order_streams=conversation_file_order_streams,
    )

    if eval_strategy in (
        "overall",
        "overall_context",
        "only_context",
        "only_qa_context",
        "memory_sequential_overall_eval",
        "memory_time",
        "memory_sequential",
        "file_sequential_RAG",
        "time_sequential_RAG",
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            *overall_conversations,
            {"role": "user", "content": qa_sample["question"]}
        ]
    elif eval_strategy == "only_qas":
        messages = []
        text_evidences = qa_sample["evidence"]["text_evidence"]
        image_evidences = qa_sample["evidence"]["image_evidence"]
        table_evidences = qa_sample["evidence"]["table_evidence"]

        messages = [{"role": "system", "content": system_prompt}]

        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here are the text evidence: \n"}]})
        if len(text_evidences) > 0:
            for text_evidence in text_evidences:
                messages.append({"role": "user", "content": [{"type": "text", "text": text_evidence}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": "No text evidence"}]})

        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here are the image evidence: \n"}]})
        if len(image_evidences) > 0:
            for image_evidence in image_evidences:
                messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_path_to_data_url(image_evidence)}}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": "No image evidence"}]})

        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here are the table evidence: \n"}]})
        if len(table_evidences) > 0:
            for table_evidence in table_evidences:
                messages.append({"role": "user", "content": [{"type": "text", "text": json.dumps(table_evidence, indent=4)}]})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": "No table evidence"}]})

        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here is the question: \n"}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Question:\n{qa_sample['question']}"}]})

    else:
        raise ValueError(f"Invalid evaluation strategy: {eval_strategy}")

    if eval_format == "multiple_choice":
        options = ["(A)", "(B)", "(C)", "(D)"]
        multi_choice_QA_options = qa_sample["multi_choice_QA"]["multi_choice_QA_options"]
        candidate_options = [f"{options[index]}: {option}" for index, option in enumerate(multi_choice_QA_options)]
        options_text = "\n".join(candidate_options)
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here are the candidate options: {options_text}"}]})
    return messages

class LongContextVLM(BaseAgent):
    def __init__(self, client,  args: dict):
        super().__init__(args)
        self.client = client
        self.args = args
        self.max_tokens = args.max_tokens
        self.eval_strategy = args.eval_strategy
        self.eval_format = args.eval_format
        self.model = args.model
        self.encoder = tiktoken.encoding_for_model(args.model) if args.model in API_VLMS else None
        self.token_limit = VLM_METAINFO_MAP[args.model]["token_limit"]

    def _turn_dicts_to_messages(self, turns: list) -> list:
        return turn_dicts_to_messages(turns)

    def preprocess_conversation(
        self,
        conversations: list,
        eval_strategy: str,
        conversation_index: int = -1,
        conversation_streams: list | None = None,
        conversation_file_order_streams: list | None = None,
    ) -> list:
        return preprocess_conversation_messages(
            conversations,
            eval_strategy,
            conversation_index,
            conversation_streams=conversation_streams,
            conversation_file_order_streams=conversation_file_order_streams,
        )

    def count_tokens(self, messages):
        total = 0
        for m in messages:

            if isinstance(m["content"], list):
                for c in m["content"]:
                    if c["type"] == "text":
                        total += len(self.encoder.encode(c["text"]))
            elif isinstance(m["content"], str):
                total += len(self.encoder.encode(m["content"]))
            else:
                raise ValueError(f"Invalid content type: {type(m['content'])}")

        return total

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60), retry=retry_if_exception_type(Exception))
    def get_response(self, client, messages, args):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
            )
            raw_response = response.choices[0].message.content
            print("prompt_tokens:", response.usage.prompt_tokens)
            return raw_response
        except Exception as e:
            raise e

    def construct_messages(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
        conversation_file_order_streams: list | None = None,
    ) -> list:
        return construct_vlm_messages(
            qa_sample,
            conversations,
            self.eval_strategy,
            self.eval_format,
            conversation_streams=conversation_streams,
            conversation_file_order_streams=conversation_file_order_streams,
        )

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: list | None = None,
        conversation_file_order_streams: list | None = None,
    ):
        messages = self.construct_messages(
            qa_sample,
            conversations,
            conversation_streams=conversation_streams,
            conversation_file_order_streams=conversation_file_order_streams,
        )

        if self.encoder is not None:
            input_tokens = self.count_tokens(messages)
            print("input_tokens: ", input_tokens)

            while input_tokens > self.token_limit:
                for i in range(1, len(messages)-1):
                    if messages[i]['role'] == 'user':
                        message_tokens = self.count_tokens([messages[i]])
                        input_tokens -= message_tokens
                        del messages[i]
                        break

        raw_response = self.get_response(self.client, messages, self.args)
        single_qa_result = self.evaluate_single_qa_answer(raw_response, qa_sample)
        return_result = {
            "question": qa_sample["question"],
            "answer": qa_sample["answer"] if self.eval_format == "fill_in_the_blank" else ["(A)", "(B)", "(C)", "(D)"][qa_sample["multi_choice_QA"]["multi_choice_QA_answer"]],
            "raw_response": raw_response,
            "single_qa_result": single_qa_result
        }
        return return_result, None

    def evaluate_cluster(
        self,
        qa_samples: List[dict],
        conversations: list,
        conversation_streams: Optional[list] = None,
        conversation_file_order_streams: Optional[list] = None,
    ) -> Tuple[List[Any], List[Any]]:
        from tqdm import tqdm

        evaluation_result: List[Any] = []
        readable_message_result: List[Any] = []
        for qa_sample in tqdm(qa_samples):
            qa_result, readable_message = self.evaluate_single_qa(
                qa_sample,
                conversations,
                conversation_streams=conversation_streams,
                conversation_file_order_streams=conversation_file_order_streams,
            )
            evaluation_result.append(qa_result)
            if readable_message is not None:
                readable_message_result.append(readable_message)
        return evaluation_result, readable_message_result

    def evaluate_single_qa_answer(self, raw_response: str, qa_sample: dict):

        if self.eval_format == "multiple_choice":
            return check_answer_match_multiple_choice(["(A)", "(B)", "(C)", "(D)"][qa_sample["multi_choice_QA"]["multi_choice_QA_answer"]], raw_response)
        elif self.eval_format == "fill_in_the_blank":
            return qa_sample["answer"].lower().strip() == raw_response.lower().strip()
        else:
            raise ValueError(f"Invalid evaluation format: {self.eval_format}")

from __future__ import annotations

import traceback
import sys
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from .base import BaseAgent

class BaseRAGAgent(BaseAgent):

    def _failed_qa_pair(self, qa_sample: dict, error_log: str) -> Tuple[dict, dict]:
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

    def evaluate_cluster(
        self,
        qa_samples: List[dict],
        conversations: list,
        conversation_streams: Optional[list] = None,
    ) -> Tuple[List[Any], List[Any]]:

        evaluation_result: List[Any] = []
        readable_message_result: List[Any] = []
        for qa_sample in tqdm(qa_samples):
            try:
                qa_result, readable_message = self.evaluate_single_qa(
                    qa_sample,
                    conversations,
                    conversation_streams=conversation_streams,
                )
            except Exception:
                error_log = traceback.format_exc()
                print(
                    f"[evaluate_cluster] evaluate_single_qa failed "
                    f"qa id={qa_sample.get('id')!r}\n{error_log}",
                    file=sys.stderr,
                    flush=True,
                )
                qa_result, readable_message = self._failed_qa_pair(qa_sample, error_log)
            evaluation_result.append(qa_result)
            if readable_message is not None:
                readable_message_result.append(readable_message)
        return evaluation_result, readable_message_result

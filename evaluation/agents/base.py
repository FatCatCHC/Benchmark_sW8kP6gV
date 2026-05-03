from typing import Any, List, Optional, Tuple
from utils import check_answer_match_multiple_choice

class BaseAgent:
    def __init__(self, args: dict):
        self.args = args

    def evaluate_cluster(
        self,
        qa_samples: List[dict],
        conversations: list,
        conversation_streams: Optional[list] = None,
    ) -> Tuple[List[Any], List[Any]]:
        raise NotImplementedError(
            "Subclasses must implement evaluate_cluster (e.g. LongContextVLM), "
            "inherit BaseRAGAgent for the RAG QA loop, or BaseMemoryAgent for build_memory + QA loop."
        )

    def evaluate_single_qa(
        self,
        qa_sample: dict,
        conversations: list,
        conversation_streams: Optional[list] = None,
    ):
        raise NotImplementedError("Subclasses must implement this method")

    def _evaluate_answer(self, raw_response: str, qa_sample: dict) -> bool:
        if self.args.eval_format == "multiple_choice":
            answer = ["(A)", "(B)", "(C)", "(D)"][qa_sample["multi_choice_QA"]["multi_choice_QA_answer"]]
            return check_answer_match_multiple_choice(answer, raw_response)
        gt = qa_sample.get("answer", "")
        return gt.lower().strip() == raw_response.lower().strip()

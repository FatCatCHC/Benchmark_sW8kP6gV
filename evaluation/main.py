import os
import sys
import argparse
import importlib
import importlib.util

_EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)

_PREPROCESS_DIR = os.path.join(_EVAL_ROOT, "preprocess")
if os.path.isdir(_PREPROCESS_DIR) and _PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, _PREPROCESS_DIR)

from dotenv import load_dotenv

from constant import ALL_VLMS, VLM_METAINFO_MAP
from utils import get_client, save_json_as_list
from dataset_loader import load_qa_dataset_by_strategy

_EVAL_AGENT_MODULE_CLASS = {
    "long_context_vlm": ("agents.long_context_vlm", "LongContextVLM"),
    "reflexion_mem": ("agents.Mem_Engine.reflexion_mem_agent", "ReflexionMemAgent"),
    "gen_agent_mem": ("agents.Mem_Engine.gen_agent_mem_agent", "GenAgentMemAgent"),
    "mb_mem": ("agents.Mem_Engine.mb_mem_agent", "MBMemAgent"),
    "st_mem": ("agents.Mem_Engine.st_mem_agent", "STMemAgent"),
    "lt_mem": ("agents.Mem_Engine.lt_mem_agent", "LTMemAgent"),
    "mg_mem": ("agents.Mem_Engine.mg_mem_agent", "MGMemAgent"),
    "sc_mem": ("agents.Mem_Engine.sc_mem_agent", "SCMemAgent"),
    "mt_mem": ("agents.Mem_Engine.mt_mem_agent", "MTMemAgent"),
    "universal_rag": ("agents.universal_rag_agent", "UniversalRAGAgent"),
    "universal_rag_caption": ("agents.universal_rag_agent_caption", "UniversalRAGCaptionAgent"),
    "hmrag": ("agents.HMRAG.hmrag_agent", "HMRAGAgent"),
    "vrag": ("agents.VRAG.vrag_agent", "VRAGAgent"),
    "omnisimplemem": ("agents.OmniSimpleMem.omni_simplemem_agent", "OmniSimpleMemAgent"),
    "mirix": ("agents.MIRIX.mirix_agent", "MIRIXAgent"),
    "mem0": ("agents.Mem0.mem0_agent", "Mem0Agent"),
    "memverse": ("agents.MemVerse.memverse_agent", "MemVerseAgent"),
    "memgpt": ("agents.MemGPT.memgpt_agent", "MemGPTAgent"),
}

_EVAL_AGENT_CHOICES = sorted(_EVAL_AGENT_MODULE_CLASS.keys())

def _prepare_env_for_agent_import() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tik_dir = os.path.join(_EVAL_ROOT, ".tiktoken_cache")
    os.makedirs(tik_dir, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", tik_dir)

def _path_is_under(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(root), os.path.abspath(path)]) == os.path.abspath(root)
    except ValueError:
        return False

def _ensure_local_agents_package() -> None:
    root_abs = os.path.abspath(_EVAL_ROOT)
    if "agents" in sys.modules:
        pkg = sys.modules["agents"]
        paths = list(getattr(pkg, "__path__", None) or [])
        ok = any(_path_is_under(root_abs, p) for p in paths)
        if not ok and paths:
            raise ImportError(
                f"Python already imported a different top-level package named 'agents' from {paths!r}. "
                f"This shadows {os.path.join(_EVAL_ROOT, 'agents')!r}. "
                "Start a fresh interpreter or remove/rename the conflicting package."
            )
        return
    spec = importlib.util.find_spec("agents.base")
    if spec is None:
        raise ImportError(
            f"Cannot find local agents.base under {_EVAL_ROOT!r}; check sys.path[0] is the evaluation directory."
        )
    origin = getattr(spec, "origin", None)
    if origin and not _path_is_under(root_abs, origin):
        raise ImportError(
            f"agents.base resolves outside this repo: {origin!r} (expected under {_EVAL_ROOT!r})."
        )
    importlib.import_module("agents.base")

def _import_agent_module(mod_name: str):
    _prepare_env_for_agent_import()
    _ensure_local_agents_package()
    if importlib.util.find_spec(mod_name) is None:
        raise ImportError(f"No module named {mod_name!r} (sys.path[0]={sys.path[0]!r})")
    print(f"[eval] importing {mod_name!r} ...", flush=True)
    return importlib.import_module(mod_name)

def _create_eval_agent(client, args: argparse.Namespace):
    spec = _EVAL_AGENT_MODULE_CLASS.get(args.eval_agent)
    if not spec:
        raise ValueError(f"Invalid evaluation agent: {args.eval_agent}")
    mod_name, cls_name = spec
    mod = _import_agent_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(client, args)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation entrypoint (cluster directory layout)")
    parser.add_argument("--dataset_dir_path", type=str, default=None)
    parser.add_argument("--dataset_parent_dir", type=str, default=None)
    parser.add_argument("--save_root_path", type=str, required=True)
    parser.add_argument("--save_dir_name", type=str, default=None)
    parser.add_argument("--start_cluster_index", type=int, default=-1)
    parser.add_argument("--end_cluster_index", type=int, default=-1)
    parser.add_argument("--model", type=str, default="gpt-4o", choices=ALL_VLMS)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--vlm_type", type=str, default="OpenAI", choices=["POST", "OpenAI"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="overall_context",
        choices=[
            "overall_context",
            "only_qa_context",
            "file_sequential_RAG",
            "file_sequential_memory",
            "time_sequential_RAG",
            "time_sequential_memory",
        ],
    )
    parser.add_argument(
        "--eval_format",
        type=str,
        default="fill_in_the_blank",
        choices=["multiple_choice", "fill_in_the_blank"],
    )
    parser.add_argument("--eval_agent", type=str, default="long_context_vlm", choices=_EVAL_AGENT_CHOICES)
    parser.add_argument("--eval_rounds", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vllm_port", type=str, default=None)

    parser.add_argument("--mem_engine_gpu", type=str, default=None)
    parser.add_argument("--mem_engine_retrieval_num", type=int, default=10)
    parser.add_argument("--sc_mem_flash_capacity", type=int, default=3)
    parser.add_argument("--sc_mem_activation_topk", type=int, default=7)

    parser.add_argument("--universal_rag_router_model", type=str)
    parser.add_argument("--universal_rag_text_pkl", type=str, default="")
    parser.add_argument("--universal_rag_json_pkl", type=str, default="")
    parser.add_argument("--universal_rag_paragraph_pkl", type=str, default="")
    parser.add_argument("--universal_rag_document_pkl", type=str, default="")
    parser.add_argument("--universal_rag_corpus_meta", type=str, default="")
    parser.add_argument("--universal_rag_bge_model_path", type=str, default="")
    parser.add_argument("--universal_rag_top_k", type=int, default=10)
    parser.add_argument("--universal_rag_image_pkl", type=str, default="")

    parser.add_argument("--hmrag_decompose_temperature", type=float, default=0.0)
    parser.add_argument("--hmrag_serper_api_key", type=str, default="")
    parser.add_argument("--hmrag_serper_num", type=int, default=5)
    parser.add_argument("--hmrag_vector_top_k", type=int, default=4)
    parser.add_argument("--hmrag_graph_top_k", type=int, default=4)
    parser.add_argument("--hmrag_vector_chunk_tokens", type=int, default=256)
    parser.add_argument("--hmrag_graph_chunk_tokens", type=int, default=512)
    parser.add_argument("--hmrag_embedding_root", type=str, default="")
    parser.add_argument("--hmrag_vector_pkl", type=str, default="")
    parser.add_argument("--hmrag_graph_pkl", type=str, default="")
    parser.add_argument("--hmrag_corpus_meta", type=str, default="")
    parser.add_argument("--hmrag_auxiliary_model", type=str, default="")
    parser.add_argument("--hmrag_decompose_model", type=str, default="")

    parser.add_argument("--vrag_embedding_root", type=str, default="")
    parser.add_argument("--vrag_image_pkl", type=str, default="")
    parser.add_argument("--vrag_corpus_meta", type=str, default="")
    parser.add_argument("--vrag_visual_bge_weight", type=str, default="")
    parser.add_argument("--vrag_bge_model_path", type=str, default="")
    parser.add_argument("--vrag_gpu", type=str, default=None)
    parser.add_argument("--vrag_top_k", type=int, default=3)
    parser.add_argument("--vrag_max_steps", type=int, default=6)

    parser.add_argument("--omnisimplemem_baseline_root", type=str, default="")
    parser.add_argument("--omnisimplemem_runtime_root", type=str, default="")
    parser.add_argument("--omnisimplemem_top_k", type=int, default=5)
    parser.add_argument("--omnisimplemem_embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--omnisimplemem_embedding_dim", type=int, default=384)
    parser.add_argument("--omnisimplemem_visual_embedding_model", type=str, default="UCSC-VLAA/openvision-vit-large-patch14-224")
    parser.add_argument("--omnisimplemem_visual_embedding_dim", type=int, default=768)
    parser.add_argument("--omnisimplemem_chunk_turns", type=int, default=20)
    parser.add_argument("--omnisimplemem_chunk_max_chars", type=int, default=6000)
    parser.add_argument("--omnisimplemem_chunk_max_images", type=int, default=8)

    parser.add_argument("--mirix_api_url", type=str, default="")
    parser.add_argument("--mirix_org_id", type=str, default="")
    parser.add_argument("--mirix_user_id", type=str, default="")
    parser.add_argument("--mirix_llm_model", type=str, default="")
    parser.add_argument("--mirix_embedding_model", type=str, default="")
    parser.add_argument("--mirix_embedding_dim", type=int, default=1536)
    parser.add_argument("--mirix_chunk_turns", type=int, default=6)
    parser.add_argument("--mirix_chunk_max_chars", type=int, default=12000)
    parser.add_argument("--mirix_chunk_max_images", type=int, default=8)
    parser.add_argument("--mirix_retrieve_limit", type=int, default=8)
    parser.add_argument("--mirix_debug", action="store_true")
    parser.add_argument(
        "--mirix_memory_modal_mode",
        type=str,
        default="",
        choices=["", "original", "caption"],
    )
    parser.add_argument(
        "--mirix_ingest_media_mode",
        type=str,
        default="original",
        choices=["original", "caption"],
    )
    parser.add_argument("--mirix_no_embeddings", action="store_true")

    parser.add_argument("--mem0_user_id", type=str, default="")
    parser.add_argument("--mem0_collection_name", type=str, default="")
    parser.add_argument("--mem0_qdrant_path", type=str, default="")
    parser.add_argument("--mem0_embedding_model", type=str, default="text-embedding-v4")
    parser.add_argument("--mem0_embedding_dim", type=int, default=1024)
    parser.add_argument("--mem0_chunk_turns", type=int, default=6)
    parser.add_argument("--mem0_chunk_max_chars", type=int, default=12000)
    parser.add_argument("--mem0_chunk_max_images", type=int, default=8)
    parser.add_argument("--mem0_retrieve_limit", type=int, default=8)
    parser.add_argument(
        "--mem0_memory_modal_mode",
        type=str,
        default="",
        choices=["", "original", "caption"],
    )
    parser.add_argument(
        "--mem0_ingest_media_mode",
        type=str,
        default="original",
        choices=["original", "caption"],
    )

    parser.add_argument("--memverse_api_url", type=str, default="")
    parser.add_argument("--memverse_insert_timeout_s", type=float, default=1800.0)
    parser.add_argument("--memverse_query_timeout_s", type=float, default=180.0)
    parser.add_argument("--memverse_chunk_turns", type=int, default=6)
    parser.add_argument("--memverse_chunk_max_chars", type=int, default=12000)
    parser.add_argument("--memverse_query_mode", type=str, default="hybrid")
    parser.add_argument("--memverse_use_pm", action="store_true")

    parser.add_argument("--memgpt_server_url", type=str, default="")
    parser.add_argument("--memgpt_server_password", type=str, default="")
    parser.add_argument("--memgpt_chunk_turns", type=int, default=6)
    parser.add_argument("--memgpt_chunk_max_chars", type=int, default=12000)
    parser.add_argument("--memgpt_chunk_max_images", type=int, default=8)
    parser.add_argument("--memgpt_context_window", type=int, default=16000)
    parser.add_argument("--memgpt_embedding_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--memgpt_embedding_dim", type=int, default=1536)
    parser.add_argument(
        "--memgpt_ingest_mode",
        type=str,
        default="archival",
        choices=["archival", "messages"],
    )
    parser.add_argument("--memgpt_archival_passage_max_chars", type=int, default=12000)
    parser.add_argument("--memgpt_archival_top_k", type=int, default=12)
    parser.add_argument("--memgpt_archival_tag", type=str, default="memgpt_eval")
    parser.add_argument("--memgpt_archival_qa_max_images", type=int, default=8)
    parser.add_argument(
        "--memgpt_memory_modal_mode",
        type=str,
        default="",
        choices=["", "original", "caption"],
    )
    parser.add_argument(
        "--memgpt_ingest_media_mode",
        type=str,
        default="original",
        choices=["original", "caption"],
    )

    return parser

def _apply_paths_and_strategy(args: argparse.Namespace) -> None:
    if not args.dataset_dir_path or not os.path.isdir(args.dataset_dir_path):
        raise FileNotFoundError(f"--dataset_dir_path is not a directory: {args.dataset_dir_path!r}")

    if not args.save_dir_name:
        raise ValueError("save_dir_name must be set before _apply_paths_and_strategy")

    os.makedirs(args.save_root_path, exist_ok=True)
    args.output_dir = os.path.join(args.save_root_path, args.save_dir_name)
    os.makedirs(args.output_dir, exist_ok=True)
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.eval_format != "multiple_choice":
        args.eval_format = "multiple_choice"
        print("[INFO] Cluster QA uses multiple_choice; eval_format set to multiple_choice")

    if args.eval_agent in (
        "reflexion_mem",
        "gen_agent_mem",
        "mb_mem",
        "st_mem",
        "lt_mem",
        "mg_mem",
        "sc_mem",
        "mt_mem",
    ):
        args.mem_engine_checkpoint_dir_path = os.path.join(args.checkpoint_dir, "mem_engine")
        os.makedirs(args.mem_engine_checkpoint_dir_path, exist_ok=True)
    else:
        args.mem_engine_checkpoint_dir_path = None

def get_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset_parent_dir:
        if not os.path.isdir(args.dataset_parent_dir):
            parser.error(f"--dataset_parent_dir is not a directory: {args.dataset_parent_dir}")
        if (args.start_cluster_index == -1) != (args.end_cluster_index == -1):
            parser.error(
                "--start_cluster_index and --end_cluster_index must both be -1, "
                "or both be valid non-negative indices."
            )
        if args.start_cluster_index != -1 and args.end_cluster_index != -1:
            if args.start_cluster_index < 0 or args.end_cluster_index < 0:
                parser.error("--start_cluster_index and --end_cluster_index must be >= 0 or both be -1.")
            if args.start_cluster_index > args.end_cluster_index:
                parser.error("--start_cluster_index cannot be greater than --end_cluster_index.")
        os.makedirs(args.save_root_path, exist_ok=True)
        args._batch_from_parent = True
        return args
    args._batch_from_parent = False
    if not args.dataset_dir_path or not args.save_dir_name:
        parser.error("Provide --dataset_dir_path and --save_dir_name, or use --dataset_parent_dir.")
    _apply_paths_and_strategy(args)
    return args

def evaluate_single_subset(subset_dataset, args):
    client = get_client(args)
    agent = _create_eval_agent(client, args)

    conversations = subset_dataset["conversation"]
    qa_samples = subset_dataset["qas"]
    conversation_streams = subset_dataset.get("conversation_streams")
    evaluation_result, readable_message_result = agent.evaluate_cluster(
        qa_samples,
        conversations,
        conversation_streams=conversation_streams,
    )
    for qa_result in evaluation_result:
        print(qa_result)
        print("-" * 100)
    return evaluation_result, readable_message_result

def _truncate_subset_for_debug(subset_dataset: dict) -> dict:
    debug_turn_limit = 10
    truncated = dict(subset_dataset)
    conversations = subset_dataset.get("conversation") or []
    truncated["conversation"] = conversations[:debug_turn_limit]

    conversation_streams = subset_dataset.get("conversation_streams")
    if isinstance(conversation_streams, list):
        truncated_streams = []
        remaining = debug_turn_limit
        for stream in conversation_streams:
            if remaining <= 0:
                break
            if not isinstance(stream, list):
                continue
            stream_slice = stream[:remaining]
            if stream_slice:
                truncated_streams.append(stream_slice)
                remaining -= len(stream_slice)
        truncated["conversation_streams"] = truncated_streams

    return truncated

def evaluate(args):
    subset_dataset = load_qa_dataset_by_strategy(args.dataset_dir_path, args.eval_strategy)
    if args.debug:
        subset_dataset = _truncate_subset_for_debug(subset_dataset)
    evaluation_results, readable_message_results = evaluate_single_subset(subset_dataset, args)
    return {"cluster": evaluation_results}, {"cluster": readable_message_results}

def _result_model_name(args) -> str:
    return args.model

def get_output_path(args, round_index: int) -> str:
    model_name = _result_model_name(args)
    name = f"{args.eval_strategy}__{args.eval_agent}__{model_name}__round{round_index}.json"
    return os.path.join(args.output_dir, name)

def get_readable_message_output_path(args, round_index: int) -> str:
    model_name = _result_model_name(args)
    name = f"{args.eval_strategy}__{args.eval_agent}__{model_name}__round{round_index}_readable.json"
    return os.path.join(args.output_dir, name)

def run_evaluation_pipeline(args):
    args.api_key = os.environ.get(VLM_METAINFO_MAP[args.model]["api_key_name"], None)
    args.base_url = VLM_METAINFO_MAP[args.model]["base_url"]
    args.model = VLM_METAINFO_MAP[args.model]["model"]
    if args.vllm_port is not None:
        args.base_url = f"http://localhost:{args.vllm_port}/v1"
    for i in range(args.eval_rounds):
        evaluation_result, readable_message_results = evaluate(args)
        save_json_as_list(evaluation_result, get_output_path(args, i))
        save_json_as_list(readable_message_results, get_readable_message_output_path(args, i))

def _parse_cluster_index(dirname: str):
    prefix = "cluster_"
    if not isinstance(dirname, str) or not dirname.startswith(prefix):
        return None
    suffix = dirname[len(prefix):]
    return int(suffix) if suffix.isdigit() else None

def _collect_cluster_subdirs(parent: str, start_cluster_index: int, end_cluster_index: int):
    subdirs = []
    for name in sorted(os.listdir(parent)):
        path = os.path.join(parent, name)
        if not os.path.isdir(path):
            continue
        cluster_index = _parse_cluster_index(name)
        if cluster_index is None:
            continue
        if start_cluster_index != -1 and end_cluster_index != -1:
            if not (start_cluster_index <= cluster_index <= end_cluster_index):
                continue
        subdirs.append((cluster_index, name))
    subdirs.sort(key=lambda item: item[0])
    return [name for _, name in subdirs]

def main():
    args = get_args()
    load_dotenv()
    if getattr(args, "_batch_from_parent", False):
        parent = args.dataset_parent_dir
        subdirs = _collect_cluster_subdirs(
            parent,
            args.start_cluster_index,
            args.end_cluster_index,
        )
        if not subdirs:
            raise ValueError(
                f"No cluster subdirectories found under {parent} "
                f"for range [{args.start_cluster_index}, {args.end_cluster_index}]"
            )
        for name in subdirs:
            run_args = argparse.Namespace(**vars(args))
            run_args.dataset_dir_path = os.path.join(parent, name)
            run_args.save_dir_name = name
            run_args._batch_from_parent = False
            _apply_paths_and_strategy(run_args)
            print(f"\n{'=' * 80}\n[batch] cluster={name!r} path={run_args.dataset_dir_path!r}\n{'=' * 80}")
            run_evaluation_pipeline(run_args)
    else:
        run_evaluation_pipeline(args)

if __name__ == "__main__":
    main()

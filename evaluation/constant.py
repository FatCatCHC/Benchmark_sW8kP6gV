import os

raw_dataset_root_dir_path = os.environ.get("RAW_DATASET_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "raw_datasets"))
dataset_root_dir_path = os.environ.get("DATASET_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets"))
evaluation_output_root_dir_path = os.environ.get("EVALUATION_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation_output"))

Image_Root_Dir_Path = os.environ.get("IMAGE_ROOT_DIR", os.path.join(dataset_root_dir_path, "PreFinal_Dataset", "MidFinal_Images"))

VLLM_VLMS = [
    "Qwen3-VL-8B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "GLM-4.6V-Flash",
]

API_VLMS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "kimi-k2.5",
    "qwen3-vl-plus",
    "qwen3-vl-flash",
    "qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-32b-instruct",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-235b-a22b-instruct",
]

ALL_VLMS = VLLM_VLMS + API_VLMS

_MODEL_API_BASE_URL = os.environ.get("MODEL_API_BASE_URL", "")
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

_DEFAULT_TOKEN_LIMIT = 108000

VLM_METAINFO_MAP = {
    name: {
        "base_url": _VLLM_BASE_URL,
        "api_key_name": "VLLM_API_KEY",
        "token_limit": _DEFAULT_TOKEN_LIMIT,
        "model": name,
    }
    for name in VLLM_VLMS
}

VLM_METAINFO_MAP.update(
    {
        name: {
            "base_url": _MODEL_API_BASE_URL,
            "api_key_name": "MODEL_API_KEY",
            "token_limit": _DEFAULT_TOKEN_LIMIT,
            "model": name,
        }
        for name in API_VLMS
    }
)

import os
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace

import requests
from openai import OpenAI
import mimetypes
import base64
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    RetryError,
    stop_after_attempt,
    wait_fixed,
)
import time

EVALUATION_CANDIDATE_TOOLS = [
  {
    "function_name": "AMAZON_search_products",
    "default_arguments": {},
    "function_comment": "Search Amazon products via Rainforest API and return a concise plain-text list.\nArgs:\n  keywords (str): Product search keywords.\n  n (int): Number of results to include (1–10).\n  page (int): Result page index starting from 1.\n  amazon_domain (str): Amazon domain such as \"amazon.com\".\nReturns:\n  result (str): Plain-text list of up to n products or an error message."
  },
  {
    "function_name": "CAR_PRICE_get_car_brands",
    "default_arguments": {},
    "function_comment": "Get all available car brands from FIPE via the fipe.online API.\nArgs:\n  (none)\nReturns:\n  result (str): JSON string containing the list of brand identifiers and names."
  },
  {
    "function_name": "FOOD_NUTRITION_MCP_get_nutrition",
    "default_arguments": {},
    "function_comment": "Get nutritional information for a described food item using Edamam APIs.\nArgs:\n  query (Optional[str]): Food name with quantity description.\n  food (Optional[str]): Alias for query when provided instead.\nReturns:\n  result (str): Summary of key nutrients and units or an error message."
  },
  {
    "function_name": "GOOGLE_MAPS_geocode",
    "default_arguments": {},
    "function_comment": "Forward geocode a human-readable address using Google Geocoding API.\nArgs:\n  address (str): Free-text address to geocode.\n  region (Optional[str]): Region bias such as \"us\".\n  language (Optional[str]): Response language code such as \"en\".\nReturns:\n  data (Dict[str, Any]): JSON with endpoint, request parameters, and API response."
  },
  {
    "function_name": "GOOGLE_MAPS_places_text_search",
    "default_arguments": {},
    "function_comment": "Search for places using a free-text query via Google Places Text Search.\nArgs:\n  query (str): Text query describing the place.\n  languageCode (Optional[str]): Response language code.\n  regionCode (Optional[str]): Region bias code.\n  locationBias (Optional[Dict[str, Any]]): Optional circle bias for location preference.\n  maxResultCount (int): Maximum number of results to keep client-side.\n  fields (Optional[str]): Unused field mask parameter for signature parity.\nReturns:\n  data (Dict[str, Any]): JSON with endpoint, request parameters, and trimmed Places response."
  },
  {
    "function_name": "HEALTHCARE_MCP_fda_drug_lookup",
    "default_arguments": {
      "search_type": "general"
    },
    "function_comment": "Look up FDA drug information by name.\nArgs:\n  drug_name (str): Drug name to search.\n  search_type (str): Information type to retrieve; one of \"general\" (default), \"label\", or \"adverse_events\".\nReturns:\n  result (object): JSON object with status, drug_name, total_results, and matching FDA drug records."
  },
  {
    "function_name": "HEALTHCARE_MCP_health_topics",
    "default_arguments": {
      "language": "en"
    },
    "function_comment": "Get evidence-based health topic summaries from Health.gov.\nArgs:\n  topic (str): Health topic search term.\n  language (str): Result language code; \"en\" (default) or \"es\".\nReturns:\n  result (object): JSON object with status, search_term, language, total_results, and matching topic entries."
  },
  {
    "function_name": "HEALTHCARE_MCP_lookup_icd_code",
    "default_arguments": {
      "max_results": 10
    },
    "function_comment": "Look up ICD-10 diagnosis codes and descriptions.\nArgs:\n  code (Optional[str]): ICD-10 code to look up; optional if description is provided.\n  description (Optional[str]): Medical condition description to search; optional if code is provided.\n  max_results (int): Maximum number of code records to return; default 10, up to 50.\nReturns:\n  result (object): JSON object with status, search_term, total_results, and matching ICD-10 code records."
  },
  {
    "function_name": "HEALTHCARE_MCP_pubmed_search",
    "default_arguments": {
      "max_results": 5,
      "open_access": False
    },
    "function_comment": "Search medical literature in the PubMed database.\nArgs:\n  query (str): Search query for medical literature.\n  max_results (int): Maximum number of results to return; default 5, range 1-100.\n  date_range (str): Optional year-window filter such as \"5\" for the last 5 years.\n  open_access (bool): Whether to restrict results to open-access articles; default false.\nReturns:\n  result (object): JSON object with status, query, total_results, and matching article records."
  },
  {
    "function_name": "METMUSEUM_MCP_list_departments",
    "default_arguments": {},
    "function_comment": "Lists all the valid departments at The Met"
  },
  {
    "function_name": "NASA_MCP_get_notifications",
    "default_arguments": {
      "notification_type": "all"
    },
    "function_comment": "Retrieve real-time space weather alerts from NASA DONKI Notifications API.\nEndpoint:\n    GET https://api.nasa.gov/DONKI/notifications\n\nPurpose:\n    Provides notification messages for various space weather disturbances,\n    useful for monitoring solar events that may impact satellites, aviation,\n    power grids, or auroras on Earth.\n\nArgs:\n    start_date (str):\n        Filter notifications occurring on/after this date.\n        Format: \"YYYY-MM-DD\".\n        Default: 7 days before current date.\n    end_date (str):\n        Filter notifications occurring on/before this date.\n        Format: \"YYYY-MM-DD\".\n        Default: current date.\n    notification_type (str):\n        Category filter:\n            - \"all\" (default): return every type available\n            - \"FLR\": Solar Flare\n            - \"SEP\": Solar Energetic Particle Event\n            - \"CME\": Coronal Mass Ejection\n            - \"IPS\": Interplanetary Shock\n            - \"MPC\": Magnetopause Crossing\n            - \"GST\": Geomagnetic Storm\n            - \"RBE\": Radiation Belt Enhancement\n            - \"report\": Composite space weather summary reports\n\nReturns:\n    str: Human-readable formatted output including:\n        - Total notification count\n        - For up to 10 notifications:\n          * messageID (unique identifier)\n          * messageType\n          * messageIssueTime\n          * messageHeader\n          * messageBody (first 200 characters)\n\nCommon Errors:\n    - 400 Bad Request: invalid date format / type filter\n    - 429 Too Many Requests: API key rate limit exceeded\n    - 500 Server Errors: NASA endpoint temporarily unavailable\n\nUsage example (MCP prompt):\n    Step 1:\n    Call nasa-mcp/get_notifications with:\n    {\n      \"start_date\": \"2025-10-22\",\n      \"end_date\": \"2025-10-29\",\n      \"notification_type\": \"FLR\"\n    }"
  },
  {
    "function_name": "NATIONALPARKS_getAlerts",
    "default_arguments": {},
    "function_comment": "Get current alerts for national parks.\n  Args:\n    (see schema): Use GetAlertsSchema fields such as park codes or state.\n  Returns:\n    result (text): JSON string listing active alerts."
  },
  {
    "function_name": "NATIONALPARKS_getCampgrounds",
    "default_arguments": {},
    "function_comment": "Get information about available campgrounds and amenities.\n  Args:\n    (see schema): Use GetCampgroundsSchema fields such as park code or state.\n  Returns:\n    result (text): JSON string listing campgrounds."
  },
  {
    "function_name": "NIXOS_nixos_search",
    "default_arguments": {},
    "function_comment": "Search NixOS packages, options, programs, or flakes.\nArgs:\n  query (str): Search term to match.\n  search_type (str): One of packages, options, programs, or flakes.\n  limit (int): Maximum number of results to return (1–100).\n  channel (str): NixOS channel name such as unstable or stable.\nReturns:\n  result (str): Plain-text summary of matching entries or an error message."
  },
  {
    "function_name": "OKX_get_price",
    "default_arguments": {},
    "function_comment": "Get the latest price for an OKX trading instrument.\n  Args:\n    instrument (string): Instrument ID such as BTC-USDT.\n  Returns:\n    result (text): JSON string with current price and basic ticker fields."
  },
  {
    "function_name": "REDDIT_MCP_SERVER_search_hot_posts",
    "default_arguments": {},
    "function_comment": "Retrieve hot posts from a subreddit using the Reddit API.\nArgs:\n  subreddit (str): Subreddit name without the r/ prefix.\n  limit (int): Number of posts to retrieve.\nReturns:\n  result (str): Plain-text list of posts with title, score, comments, author, and link, or an error message."
  },
  {
    "function_name": "TMDB_search_movies",
    "default_arguments": {
      "page": 1
    },
    "function_comment": "Search for movies using The Movie Database API.\n  Args:\n    query (string): Search query text.\n    year (number): Optional filter by release year.\n    page (number): Optional page number, defaults to 1.\n  Returns:\n    result (text): JSON string of the TMDB /search/movie response."
  },
  {
    "function_name": "WEATHER_get_weather",
    "default_arguments": {},
    "function_comment": "Get current weather for a U.S. location using Open‑Meteo with ZIP/city geocoding.\nArgs:\n  location (str): City/state name or a 5‑digit U.S. ZIP code.\n  units (Literal[\"us\",\"metric\"]): \"us\" for imperial units or \"metric\" for SI units.\nReturns:\n  result (str): Plain-text sentence summarizing temperature, wind, code, and place name."
  },
  {
    "function_name": "WIKI_search",
    "default_arguments": {},
    "function_comment": "Search Wikipedia for pages matching a query and return top titles.\nArgs:\n  query (str): Free-text search query.\n  n (int): Number of results to include (1–10).\nReturns:\n  result (str): Plain-text list of up to n titles or a no-results message."
  },
  {
    "function_name": "WIKI_summary",
    "default_arguments": {},
    "function_comment": "Fetch a short Wikipedia summary for the given page title.\nArgs:\n  title (str): Exact or redirectable page title.\nReturns:\n  result (str): Plain-text block with title, short description, and extract."
  }
]

def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value

class _PostCompletionsAPI:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def create(self, **kwargs):
        payload = {}
        for key in (
            "model",
            "messages",
            "max_tokens",
            "temperature",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
        ):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=600,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"POST chat completion failed with status {response.status_code}: {response.text}"
            ) from exc

        try:
            response_json = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"POST chat completion returned invalid JSON: {response.text}"
            ) from exc

        response_json.setdefault(
            "usage",
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        )
        return _to_namespace(response_json)

class _PostChatAPI:
    def __init__(self, api_key: str, base_url: str):
        self.completions = _PostCompletionsAPI(api_key, base_url)

class PostCompatibleClient:
    def __init__(self, api_key: str, base_url: str):
        self.chat = _PostChatAPI(api_key, base_url)

def load_json_as_list(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise e

def save_json_as_list(data, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise e

def get_client(args):
    if args.vlm_type == "OpenAI":
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
        )
        return client
    elif args.vlm_type == "POST":
        return PostCompatibleClient(
            api_key=args.api_key,
            base_url=args.base_url,
        )
    else:
        raise ValueError(f"Unsupported VLM type: {args.vlm_type}")

def _usage_from_chat_response(response) -> Optional[Dict[str, Any]]:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
    }

def _last_attempt_exception(retry_err: RetryError) -> Optional[BaseException]:
    la = retry_err.last_attempt
    if la is None:
        return None
    ex_fn = getattr(la, "exception", None)
    if callable(ex_fn):
        inner = ex_fn()
        if isinstance(inner, BaseException):
            return inner
    outcome = getattr(la, "outcome", None)
    if outcome is not None:
        ex_fn = getattr(outcome, "exception", None)
        if callable(ex_fn):
            inner = ex_fn()
            if isinstance(inner, BaseException):
                return inner
    return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(60), retry=retry_if_exception_type(Exception))
def _get_response_with_retry(
    client, messages, args, return_usage: bool = False
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
        )
        raw_response = response.choices[0].message.content
        if return_usage:
            return raw_response, _usage_from_chat_response(response)
        return raw_response
    except Exception as e:
        print(f"Error: {e}")
        raise

def get_response(
    client, messages, args, return_usage: bool = False
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
    try:
        time.sleep(2)
        return _get_response_with_retry(client, messages, args, return_usage=return_usage)
    except RetryError as e:
        inner = _last_attempt_exception(e)
        if inner is not None:
            raise inner from e
        raise

def get_response_universal_rag(
    client, messages, args, return_usage: bool = False
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
    time.sleep(2)
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
        )
        raw_response = response.choices[0].message.content
        if return_usage:
            return raw_response, _usage_from_chat_response(response)
        return raw_response
    except Exception as e:
        print(f"Error: {e}")
        raise

from agents.prompt import (
    FUNCTION_CALL_EVALUATION_SYSTEM_PROMPT,
    FUNCTION_CALL_USER_PROMPT_TEMPLATE,
)

def load_function_call_candidate_tools() -> List[dict]:
    if not isinstance(EVALUATION_CANDIDATE_TOOLS, list):
        raise ValueError("Invalid candidate tools format in evaluation/candidate_tools.py: expected a list")
    return [item for item in EVALUATION_CANDIDATE_TOOLS if isinstance(item, dict)]

def format_function_call_candidate_tools(candidate_tools: Optional[List[dict]] = None) -> str:
    tools = candidate_tools if candidate_tools is not None else load_function_call_candidate_tools()
    if not tools:
        return "No candidate tools available."

    formatted_tools: List[str] = []
    for tool in tools:
        function_name = str(tool.get("function_name", "")).strip()
        function_comment = str(tool.get("function_comment", "")).strip()
        if not function_name:
            continue
        if function_comment:
            formatted_tools.append(f"- {function_name}\n{function_comment}")
        else:
            formatted_tools.append(f"- {function_name}")
    return "\n\n".join(formatted_tools) if formatted_tools else "No candidate tools available."

def build_function_call_messages_multimodal(
    question: str,
    context_parts: List[dict],
    candidate_tools: Optional[List[dict]] = None,
) -> List[dict]:
    system_prompt = FUNCTION_CALL_EVALUATION_SYSTEM_PROMPT.format(
        candidate_tools=format_function_call_candidate_tools(candidate_tools)
    )
    parts: List[dict] = [{"type": "text", "text": "Retrieved context:\n"}]
    parts.extend(context_parts)
    parts.append(
        {
            "type": "text",
            "text": (
                f"\n\nQuestion:\n{question}\n\n"
                "Return only the JSON function-call plan.\n"
            ),
        }
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": parts},
    ]

def flatten_user_messages_to_content_parts(messages: List[dict]) -> List[dict]:
    parts: List[dict] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(item)
        elif isinstance(content, dict):
            text = _content_item_to_text(content)
            if text.strip():
                parts.append({"type": "text", "text": text})
    return parts

def build_function_call_messages(
    question: str,
    context: str,
    candidate_tools: Optional[List[dict]] = None,
) -> List[dict]:
    system_prompt = FUNCTION_CALL_EVALUATION_SYSTEM_PROMPT.format(
        candidate_tools=format_function_call_candidate_tools(candidate_tools)
    )
    user_prompt = FUNCTION_CALL_USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def build_function_call_prompt_text(
    question: str,
    context: str,
    candidate_tools: Optional[List[dict]] = None,
) -> str:
    messages = build_function_call_messages(question, context, candidate_tools)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    return f"{system_prompt}\n\n{user_prompt}"

def _extract_balanced_json_substring(text: str, start_index: int) -> Optional[str]:
    opening_char = text[start_index]
    if opening_char not in "[{":
        return None
    closing_char = "]" if opening_char == "[" else "}"
    stack = [opening_char]
    in_string = False
    escaped = False

    for index in range(start_index + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                return None
            expected = closing_char if len(stack) == 1 else ("]" if stack[-1] == "[" else "}")
            if char != expected:
                return None
            stack.pop()
            if not stack:
                return text[start_index:index + 1]
    return None

def extract_json_payload(raw_response: Any) -> Optional[Any]:
    if isinstance(raw_response, (list, dict)):
        return raw_response
    if not isinstance(raw_response, str):
        return None

    response_text = raw_response.strip()
    if not response_text:
        return None

    candidates = [response_text]
    if "```" in response_text:
        segments = response_text.split("```")
        for i in range(1, len(segments), 2):
            block = segments[i].strip()
            if "\n" in block:
                first_line, remainder = block.split("\n", 1)
                if first_line.strip().lower() == "json":
                    block = remainder.strip()
            candidates.append(block)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    for opening in ("[", "{"):
        start = response_text.find(opening)
        while start != -1:
            candidate = _extract_balanced_json_substring(response_text, start)
            if candidate is not None:
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    pass
            start = response_text.find(opening, start + 1)
    return None

def _normalize_function_call_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key).lower(): _normalize_function_call_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]).lower())
        }
    if isinstance(value, list):
        return [_normalize_function_call_value(item) for item in value]
    if isinstance(value, str):
        return value.lower()
    return value

def _merge_missing_default_arguments_from_candidate_tools(
    function_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    merged = dict(arguments)
    fn_lower = function_name.strip().lower()
    for item in EVALUATION_CANDIDATE_TOOLS:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("function_name")
        if not isinstance(tool_name, str) or tool_name.strip().lower() != fn_lower:
            continue
        defaults = item.get("default_arguments")
        if not isinstance(defaults, dict):
            break
        merged_keys_lower = {str(k).lower() for k in merged}
        for key, val in defaults.items():
            key_lower = str(key).lower()
            if key_lower not in merged_keys_lower:
                merged[key] = val
                merged_keys_lower.add(key_lower)
        break
    return merged

def normalize_function_call_answer(answer: Any) -> Optional[List[dict]]:
    payload = answer
    if isinstance(payload, dict):
        if isinstance(payload.get("answer"), list):
            payload = payload["answer"]
        elif isinstance(payload.get("steps"), list):
            payload = payload["steps"]
        elif isinstance(payload.get("calls"), list):
            payload = [{"step": payload.get("step", 1), "calls": payload["calls"]}]
        else:
            return None

    if not isinstance(payload, list):
        return None

    normalized_steps: List[dict] = []
    for index, step in enumerate(payload, start=1):
        if not isinstance(step, dict):
            return None

        raw_calls = step.get("calls")
        if not isinstance(raw_calls, list):
            return None

        step_id = step.get("step", index)
        if isinstance(step_id, str) and step_id.strip().isdigit():
            step_id = int(step_id.strip())

        normalized_calls: List[dict] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                return None

            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                return None
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return None

            name_stripped = name.strip()
            arguments = _merge_missing_default_arguments_from_candidate_tools(
                name_stripped, arguments
            )

            normalized_calls.append(
                {
                    "name": name_stripped.lower(),
                    "arguments": _normalize_function_call_value(arguments),
                }
            )

        normalized_calls.sort(
            key=lambda call: json.dumps(call, ensure_ascii=False, sort_keys=True)
        )

        normalized_steps.append(
            {
                "step": step_id,
                "calls": normalized_calls,
            }
        )
    return normalized_steps

def evaluate_function_call_response_exact_match(
    raw_response: Any,
    expected_answer: Any,
) -> Tuple[bool, Optional[List[dict]]]:
    expected = normalize_function_call_answer(expected_answer)
    predicted_payload = extract_json_payload(raw_response)
    predicted = normalize_function_call_answer(predicted_payload)
    return expected is not None and predicted == expected, predicted

def _predicted_step_covers_expected_calls(
    predicted_calls: List[dict], expected_calls: List[dict]
) -> bool:
    pool = list(predicted_calls)
    for ec in expected_calls:
        matched_idx: Optional[int] = None
        for j, pc in enumerate(pool):
            if pc == ec:
                matched_idx = j
                break
        if matched_idx is None:
            return False
        pool.pop(matched_idx)
    return True

def evaluate_function_call_response(
    raw_response: Any,
    expected_answer: Any,
) -> Tuple[bool, Optional[List[dict]]]:
    expected = normalize_function_call_answer(expected_answer)
    predicted_payload = extract_json_payload(raw_response)
    predicted = normalize_function_call_answer(predicted_payload)
    if expected is None or predicted is None:
        return False, predicted

    if len(expected) != len(predicted):
        return False, predicted

    for exp_step, pred_step in zip(expected, predicted):
        exp_calls = exp_step.get("calls")
        pred_calls = pred_step.get("calls")
        if not isinstance(exp_calls, list) or not isinstance(pred_calls, list):
            return False, predicted
        if not _predicted_step_covers_expected_calls(pred_calls, exp_calls):
            return False, predicted

    return True, predicted

def _content_item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""

    item_type = item.get("type")
    if item_type == "text":
        return str(item.get("text", "") or "")
    if item_type == "json":
        return json.dumps(item.get("json"), ensure_ascii=False)
    if "text" in item:
        return str(item.get("text", "") or "")
    return ""

def messages_to_text_context(messages: List[dict]) -> str:
    lines: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "") or "").strip()
        content = message.get("content")
        parts: List[str] = []
        if isinstance(content, str):
            parts = [content]
        elif isinstance(content, dict):
            text = _content_item_to_text(content)
            if text:
                parts = [text]
        elif isinstance(content, list):
            parts = [text for text in (_content_item_to_text(item) for item in content) if text.strip()]

        text_blob = "\n".join(part.strip() for part in parts if part.strip()).strip()
        if not text_blob:
            continue
        lines.append(f"{role}: {text_blob}" if role else text_blob)
    return "\n\n".join(lines)

def merge_args_with_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:

    args_dict = vars(args)

    merged = {**config, **args_dict}

    return argparse.Namespace(**merged)

def guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"

def image_path_to_data_url(path: str) -> str:
    mime = guess_mime_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def data_url_to_image_path(data_url: str, save_path: str) -> str:
    if not data_url.startswith("data:"):
        raise ValueError("Input is not a valid data URL")

    try:
        header, b64_data = data_url.split(",", 1)
    except ValueError:
        raise ValueError("Malformed data URL: missing comma between header and payload")

    if ";base64" not in header:
        raise ValueError("Data URL is missing the base64 marker")

    try:
        binary_data = base64.b64decode(b64_data)
    except Exception as e:
        raise ValueError(f"base64 decode failed: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(binary_data)

    return save_path

def equal_answer_with_or_without_parentheses(answer: str, raw_response: str) -> bool:

    predicted_answer = raw_response.replace("(", "").replace(")", "").lower().strip()
    ground_truth = answer.replace("(", "").replace(")", "").lower().strip()
    return predicted_answer == ground_truth

def check_answer_match_multiple_choice(answer: str, raw_response: str) -> bool:
    try:
        if answer is None or raw_response is None:
            return False
        predicted_answer = raw_response.lower().strip()
        ground_truth = answer.lower().strip()

        if equal_answer_with_or_without_parentheses(ground_truth, predicted_answer):
            return True
        elif ground_truth in predicted_answer:
            return True
        elif "<|begin_of_box|>" in predicted_answer:

            predicted_answer = predicted_answer.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
            if equal_answer_with_or_without_parentheses(ground_truth, predicted_answer):
                return True
            else:
                return False
        else:
            return False
    except Exception:
        return False

if __name__ == "__main__":

    raw_response = "{\n  \"calls\": [\n    {\n      \"name\": \"HEALTHCARE_MCP_fda_drug_lookup\",\n      \"arguments\": {\n        \"drug_name\": \"aspirin\",\n        \"search_type\": \"general\"\n      }\n    },\n    {\n      \"name\": \"HEALTHCARE_MCP_fda_drug_lookup\",\n      \"arguments\": {\n        \"drug_name\": \"ibuprofen\",\n        \"search_type\": \"general\"\n      }\n    },\n    {\n      \"name\": \"HEALTHCARE_MCP_fda_drug_lookup\",\n      \"arguments\": {\n        \"drug_name\": \"acetaminophen\",\n        \"search_type\": \"general\"\n      }\n    },\n    {\n      \"name\": \"HEALTHCARE_MCP_lookup_icd_code\",\n      \"arguments\": {\n        \"description\": \"angina pectoris\"\n      }\n    }\n  ]\n}"
    expected_answer = [
            {
                "step": 1,
                "calls": [
                    {
                        "name": "HEALTHCARE_MCP_fda_drug_lookup",
                        "arguments": {
                            "drug_name": "aspirin"
                        }
                    }
                ]
            }
        ]
    print(evaluate_function_call_response(raw_response, expected_answer))

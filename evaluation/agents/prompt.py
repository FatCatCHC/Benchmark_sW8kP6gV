OVERALL_EVALUATION_SYSTEM_PROMPT = """\
You are a long-context multimodal assistant. You will be given a conversation history that may include a large amount of text and images. Your task is to carefully read the entire provided conversation, understand the user's question, and answer it as accurately and completely as possible.

## Specific Instructions
1. Use only the information available in the conversation and images.
2. Pay attention to earlier parts of the conversation if they contain definitions, assumptions, or details needed to answer the question.
3. If the question cannot be answered from the given context and images alone, say that you cannot answer based on the available information instead of guessing.
4. Multimodal inputs and captions: Some runs may withhold part or all multimodal evidence from you—such as images, figures, tables, document/JSON (or PDF) segments—or only provide short textual **captions/descriptions** instead of the original content. If the question clearly depends on that kind of evidence and you have **not** received either the original material or an adequate caption/description in the messages you see, answer cautiously: do not invent facts, and you may state explicitly that you have not received enough information (including missing or insufficient captions) to answer reliably, consistent with the output-format rules below.
"""

ONLY_QAS_EVALUATION_SYSTEM_PROMPT = """\
You are a long-context multimodal assistant. You will be given a piece of text, an image, and a question. Your task is to carefully read the provided text, image, and question, understand the user's question, and answer it as accurately and completely as possible.

## Specific Instructions
1. Use only the information available in the text and image.
2. Pay attention to the text and image if they contain definitions, assumptions, or details needed to answer the question.
3. If the question cannot be answered from the given text and image alone, say that you cannot answer based on the available information instead of guessing.
4. Multimodal inputs and captions: Sometimes the **image** or other non-text evidence may be missing from what you receive, or you may only get a **caption/description** instead of the original pixels—if that text is not enough to answer, do not guess; state that you lack sufficient information (including when captions are missing or too vague), consistent with the output-format rules below.
"""

MEM_ENGINE_OVERALL_EVALUATION_SYSTEM_PROMPT = """\
You are a helpful assistant. You will be given the textual context of a conversation history that may include a large amount of text and images. Your task is to carefully read the entire provided context, understand the question, and answer it as accurately and completely as possible.

## Specific Instructions
1. Only answer based on the information explicitly provided in the context text. Do not make assumptions or speculate about the meaning of the question beyond what is stated; avoid guessing any implied intent.
2. Pay attention to earlier parts of the context if they contain definitions, assumptions, or details needed to answer the question.
3. If the question cannot be answered from the given context alone, say that you cannot answer based on the available information instead of guessing.
4. Multimodal content and captions: The text you receive may omit original images, tables, documents, or PDF/JSON segments, or replace them with short captions only. If the question depends on such material and the context does not give you enough detail (including when captions are missing or insufficient), answer cautiously—do not invent facts, and say you lack enough information to answer reliably, consistent with the output-format rules below.
"""

MULTIPLE_CHOICE_OUTPUT_FORMAT = """\
## Output Format
Output exactly one choice from the options: (A), (B), (C), or (D). Do not output anything else — no explanations, no additional text, no other symbols. If the question cannot be answered from the given context and images alone, you must output "No, I can not answer this question based on the available information".
"""

FILL_IN_THE_BLANK_OUTPUT_FORMAT = """\
## Output Format
Output the correct answer in the form of a single word or phrase. Do not output any additional text else. If the question cannot be answered from the given context and images alone, you must output "No, I can not answer this question based on the available information".
"""

FUNCTION_CALL_EVALUATION_SYSTEM_PROMPT = """\
You are a function-calling planner. You will be given recalled context from a conversation, a user question, and a candidate tool set.

Your task is to determine the correct function calls needed to answer the question.

## Specific Instructions
1. Use only the candidate tools listed below.
2. Do not answer the question in natural language.
3. Output only the function-call plan as valid JSON.
4. Use the exact tool names from the candidate tools list.
5. Include every required argument and do not invent unsupported arguments.
6. Plan only the function calls required by the question itself. You do not need to reproduce a full workflow from the context if some steps are irrelevant to answering the question.
7. Return only one step of calls. Do not generate multiple sequential steps.
8. A single `calls` list may contain multiple parallel calls if the question requires them.

## Candidate Tools
{candidate_tools}

## Output Format
Return a JSON object with this exact structure:
{{
  "calls": [
    {{
      "name": "FUNCTION_NAME",
      "arguments": {{
        "arg_name": "arg_value"
      }}
    }}
  ]
}}

If multiple tool invocations are needed, place them together inside the same `calls` array.

Do not wrap the JSON in markdown code fences. Do not output any extra commentary.
"""

FUNCTION_CALL_USER_PROMPT_TEMPLATE = """\
Context:
{context}

Question:
{question}

Return only the JSON function-call plan.
"""

MEM_ENGINE_USER_PROMPT_TEMPLATE = """\
Context:
{context}

Question:
{question}
"""

MEM_ENGINE_USER_TAIL_INSTRUCTION = (
    "Answer the question based on the information provided in the context above. "
    "You must only output the option letter, not the option content or other extra text."
)

MEM_ENGINE_JSON_FALLBACK_SYSTEM_PROMPT = (
    "You must respond with a JSON object with an 'answer' field."
)

UNIVERSAL_RAG_ROUTER_PROMPT = """\
Classify the following query into one of four categories: [No, Text, Json, Image], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:
- No: The query can be answered directly with common knowledge, reasoning, or computation without external data.
- Text: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- Json: The query requires multi-hop reasoning, combining information from multiple sources or structured evidence to form a complete answer.
- Image: The query focuses on visual aspects like appearances, structures, or spatial relationships.

Examples:
1. "What is the capital of France?" → No
2. "What is the birth date of Alan Turing?" → Text
3. "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" → Json
4. "Describe the appearance of a blue whale." → Image
5. "Solve 12 * 8." → No
6. "Who played a key role in the development of the iPhone?" → Text
7. "Which Harvard University graduate played a key role in the development of the iPhone?" → Json
8. "Describe the structure of the Eiffel Tower." → Image

Classify the following query: {query}
Provide only the category.
"""

VRAG_NO_ANSWER_TEXT = "No, I can not answer this question based on the available information"

VRAG_SYSTEM_PROMPT = """\
You are a multimodal retrieval-augmented reasoning agent modeled after VRAG-RL.

At every step, you must first reason inside <think>...</think>. Then output exactly one of these actions:
1. <search>query</search> to retrieve new evidence relevant to the question. Search may return text passages, images, or both.
2. <bbox>[x1, y1, x2, y2]</bbox> to crop the most recently observed image and zoom into a region.
3. <answer>final answer</answer> when you can answer the question.

Rules:
- Always produce <think> before the action.
- Output exactly one action tag after each thought.
- Use <bbox> only for the most recently observed image.
- Bounding-box coordinates must be a JSON list of four non-negative numbers: [x1, y1, x2, y2].
- If the available information is insufficient, answer with <answer>""" + VRAG_NO_ANSWER_TEXT + """</answer>.
- Do not include any explanation outside the required tags.
"""

VRAG_INVALID_ACTION_MESSAGE = """\
Your previous action is invalid. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and the user will return searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view; the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. If you find no further external knowledge needed, directly provide the answer inside <answer> and </answer>. Please try again.
"""

VRAG_SEARCH_FAILED_MESSAGE = """\
No new relevant evidence was retrieved for your latest search. Please think again. If the current information is already sufficient, answer with <answer>...</answer>; otherwise, reformulate the search query.
"""

VRAG_FORCE_ANSWER_MESSAGE = """\
Please stop searching and answer the question now. Output the final answer only inside <answer>...</answer>.
"""

VRAG_FUNCTION_CALL_USER_SUFFIX = """\
This is a function-call planning task: when you emit <answer>...</answer>, put the JSON tool-call plan inside the tags so it matches the labeled reference (extractable JSON).
"""

VRAG_MULTIPLE_CHOICE_ANSWER_FORMAT_INSTRUCTION = (
    "Final answer format: inside <answer> output exactly one option token from "
    "(A), (B), (C), or (D)."
)

VRAG_FILL_IN_THE_BLANK_ANSWER_FORMAT_INSTRUCTION = (
    "Final answer format: inside <answer> output a short word or phrase only."
)

HMRAG_EXPERT_SYSTEM_TEMPLATE = (
    "You are the {expert_name} retrieval expert in an HM-RAG system. "
    "Answer using only the provided evidence. Do not use outside knowledge. "
    "If the evidence is weak or contradictory, still give the best grounded answer."
)

HMRAG_EXPERT_USER_TEMPLATE = (
    "Search query:\n{search_q}\n\n"
    "Original question:\n{question}\n\n"
    "{format_instruction}\n\n"
    "Retrieved evidence from the {expert_name} expert:\n{evidence_block}"
)

HMRAG_SUMMARIZE_PROMPT_TEMPLATE = (
    "You are summarizing the {expert_name} expert answer for HM-RAG consistency voting.\n"
    "Compress it into one short sentence capturing only the final claim.\n"
    "Do not add new information.\n\n"
    "Question:\n{question}\n\nAnswer:\n{answer}"
)

HMRAG_FUNCTION_CALL_FORMAT_INSTRUCTION = (
    'Answer with a JSON-formatted function call only. If the evidence is insufficient, reply exactly: '
    '"No, I can not answer this question based on the available information".'
)

HMRAG_SHORT_PHRASE_FORMAT_INSTRUCTION = (
    'Answer with a short phrase only. If the evidence is insufficient, reply exactly: '
    '"No, I can not answer this question based on the available information".'
)

HMRAG_MULTIPLE_CHOICE_FORMAT_INSTRUCTION_TEMPLATE = (
    "Choose exactly one option.\n"
    "Return only the final option in the form (A), (B), (C), or (D).\n"
    "Candidate options:\n{candidate_options}"
)

HMRAG_DECISION_REFINEMENT_CONSISTENT = (
    "The experts above are sufficiently consistent. Perform lightweight decision refinement: "
    "synthesize a final answer grounded primarily in the consensus experts, while checking the "
    "full evidence for support."
)

HMRAG_DECISION_REFINEMENT_CONFLICTING = (
    "The experts conflict or do not support one another strongly enough. Perform expert model "
    "refinement: reconcile the conflicting evidence carefully, reason over the disagreement, and "
    "produce the single best grounded answer."
)

HMRAG_WEB_SUMMARY_PROMPT_TEMPLATE = (
    "You are a search assistant. Given web search snippets and the user query, "
    "write a concise factual summary (3–6 sentences) that helps answer the query. "
    "If snippets are irrelevant, say so briefly.\n\n"
    "Query: {query}\n\nSnippets:\n{snippets}"
)

HMRAG_DECOMPOSE_COUNT_INTENTS_PROMPT_TEMPLATE = (
    "How many independent intents (separate information needs or tasks) are in the following query? "
    "Reply with a single integer only, no other text.\n\nQuery:\n"
    "{query}\n\nNumber of intents:"
)

HMRAG_DECOMPOSE_SPLIT_QUERY_PROMPT_TEMPLATE = (
    "Split the following query into multiple independent sub-queries. "
    "Separate them with '||' only (no numbering, no explanation).\n\nQuery:\n"
    "{query}\n\nSub-queries:"
)

QUESTION_WITH_OPTIONS_TEMPLATE = (
    "{question}\n\nCandidate options:\n{candidate_options}"
)

OMNISIMPLEMEM_MULTIPLE_CHOICE_QUESTION_TEMPLATE = (
    "{question}\n\n"
    "Answer with exactly one option token from (A), (B), (C), or (D).\n"
    "Candidate options:\n{candidate_options}"
)

MEMORY_INGEST_CHUNK_PREFIX_TEMPLATE = (
    "This is part of the conversation ({idx}/{total} chunk).\n"
    "Please remember this data; subsequent questions depend on all chunks.\n\n"
    "[{agent_label} ingest chunk]\n"
)

MEMORY_INGEST_FINAL_MESSAGE = (
    "All conversation chunks for this evaluation cluster have been ingested. "
    "Retain everything for later questions."
)

MEMORY_CAPTION_ONLY_QUERY_IMAGES_PREFIX = (
    "\n\n[Caption-only mode: query images are listed by reference name only (no pixels).]\n"
)

MEM0_USER_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer using only the retrieved Mem0 memory context."
)

MIRIX_USER_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer using only the retrieved MIRIX memory context."
)

MEMVERSE_FUNCTION_CALL_CONTEXT_NOTE = (
    "The MemVerse server appends retrieved long-term memory as 'Relevant memory' "
    "when generating the answer. Use that memory as the recalled context for your JSON plan."
)

MEMGPT_IMAGE_REF_MARKER_TAG = "memgpt_image_ref"

MEMGPT_ARCHIVAL_QA_INTRO = (
    "Use ONLY the following retrieved excerpts from the stored conversation (and any images appended below "
    "that correspond to [memgpt_image_ref:...] lines in the excerpts). "
    "If the answer is not supported by them, state that the context is insufficient.\n\n"
)

MEMGPT_ARCHIVAL_QA_BRIDGE_TEMPLATE = (
    "\n\nThe images below follow the order of first-seen [memgpt_image_ref:...] paths in the excerpts "
    "({attached_count} attached; remaining refs may be omitted if files were missing or over the cap)."
)

MEMGPT_ARCHIVAL_PERSONA_BLOCK = (
    "You answer benchmark questions using only the conversation excerpts the user message provides. "
    "Do not invent facts. If excerpts are insufficient, say the context is insufficient."
)

MEMGPT_ARCHIVAL_HUMAN_BLOCK = (
    "Evaluation setting: questions refer to a replayed multi-party conversation; "
    "relevant text (and optionally images matching [memgpt_image_ref:...] lines) is provided in the user message."
)

MEMGPT_MESSAGES_PERSONA_BLOCK = (
    "You are a faithful conversational memory agent for evaluation. "
    "Remember conversation facts; answer later questions only from what was said."
)

MEMGPT_MESSAGES_HUMAN_BLOCK = (
    "The user is replaying a multi-party conversation chunk by chunk."
)

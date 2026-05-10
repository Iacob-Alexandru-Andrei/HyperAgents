"""F-class meta-agent tool for one-shot LLM calls to alternate models.

Tokens in, tokens out. No tools, no chat history, no recursion. Used by
the meta-agent to delegate self-contained work (summaries, drafts, code
generation) to a model picked from the catalog -- without polluting its
own conversation context.

Bounds enforcement: the catalog is injected at load time by
``hyperagents/agent/tools/__init__.py:load_tools`` via ``functools.partial``,
so this tool's ``tool_function`` always receives the validated catalog
list. ``id`` and ``effort`` are validated against the catalog before
the underlying ``get_response_from_llm`` call; on violation, raise
``ValueError`` so ``process_tool_call`` (in
``hyperagents/agent/llm_withtools.py``) folds the error into the
meta-agent's chat as a tool-output message it can recover from.
"""


def _resolve(catalog, chosen_id, chosen_effort):
    entry = next((e for e in catalog if e.get("id") == chosen_id), None)
    if entry is None:
        known = sorted(e.get("id", "") for e in catalog)
        raise ValueError(f"unknown model id {chosen_id!r}; allowed: {known}")
    allowed = tuple(entry.get("allowed_reasoning_efforts") or ())
    if not allowed and chosen_effort is not None:
        raise ValueError(
            f"model {chosen_id!r} does not accept reasoning_effort; got {chosen_effort!r}"
        )
    if allowed and chosen_effort is not None and chosen_effort not in allowed:
        raise ValueError(
            f"reasoning_effort {chosen_effort!r} not allowed for {chosen_id!r}; "
            f"allowed: {sorted(allowed)}"
        )
    return entry["model"], chosen_effort


def tool_info(*, catalog=None):
    catalog = list(catalog or [])
    if not catalog:
        return {
            "name": "query_model",
            "description": (
                "Disabled: no model_catalog configured for this run. "
                "Configure ScientistConfig.model_catalog to enable one-shot "
                "delegation to alternate models."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ids = [e.get("id", "") for e in catalog]
    return {
        "name": "query_model",
        "description": (
            "Fire a one-shot LLM call to a model from the catalog. Tokens in, "
            "tokens out -- no chat history, no tools, no recursion. Use to "
            "delegate self-contained work (drafts, summaries, code) to a "
            f"different model. Allowed ids: {ids}. See MODEL_CATALOG.md for "
            "per-model context window, cost, and allowed reasoning_effort values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The user message to send."},
                "id": {"type": "string", "enum": ids, "description": "Catalog id."},
                "effort": {
                    "type": ["string", "null"],
                    "description": (
                        "OpenAI reasoning_effort (low|medium|high). Pass null "
                        "for models that do not accept it."
                    ),
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max response tokens (default 4096).",
                },
            },
            "required": ["prompt", "id"],
        },
    }


def tool_function(prompt, id, effort=None, max_tokens=4096, *, catalog=None):
    catalog = list(catalog or [])
    if not catalog:
        raise ValueError("query_model is disabled: no model_catalog configured.")
    model_str, eff = _resolve(catalog, id, effort)
    from agent.llm import get_response_from_llm

    response, _, _ = get_response_from_llm(
        msg=prompt,
        model=model_str,
        msg_history=[],
        reasoning_effort=eff,
        max_tokens=max_tokens,
    )
    return response

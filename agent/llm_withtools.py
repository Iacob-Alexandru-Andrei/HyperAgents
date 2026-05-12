import json
import os
import re

from agent.llm import get_response_from_llm
from agent.tools import load_tools

# F2o (recursive-scientist deviation): the cost proxy writes a JSON
# status file alongside the markdown one (``status.latest.json``). The
# JSON carries ``cumulative_prompt_tokens`` and ``cumulative_chat_chars``
# which the compaction trigger reads to derive a live bytes-per-token
# compression ratio. The two files are siblings of ``status.md``; the
# host mounts the entire ``budget_dir`` at ``BUDGET_STATUS_CONTAINER_DIR``
# so this side-band lookup is just a sibling filename read.
_PROXY_STATUS_JSON_BASENAME = "status.latest.json"
# Conservative bytes-per-token fallback used before the proxy has logged
# any usage. 3.0 over-estimates (more bytes per token than reality for
# Nemotron's reasoning-heavy turns, where the live ratio settles around
# 3.0-3.5) so compaction errs on the side of firing early rather than
# late. ``chars / 3.0`` overshoots; ``chars / 4.0`` (the legacy default)
# undershoots and triggered the May 11-12 late-compaction incident.
_COMPRESSION_RATIO_FALLBACK = 3.0
# Hard floor below which we never trust the live ratio. A transient
# under-report of ``cumulative_chat_chars`` (e.g. a first response that
# got streamed before its tokens were logged) could otherwise push the
# ratio toward 1.0 and make the compactor think the model is suddenly
# 1-byte-per-token, which would never fire.
_COMPRESSION_RATIO_FLOOR = 2.0


_COMPACTION_SUMMARY_PROMPT = (
    "You are compressing your own reasoning history to free up context.\n"
    "Below are the turns of a meta-agent (your prior self) working on a "
    "code-improvement task. Produce a single dense summary preserving "
    "ONLY the facts a continuation would need:\n\n"
    "1. Files you read and their relevant contents (filename + 1-3 line summary each).\n"
    "2. Hypotheses you formed and tested, with outcomes.\n"
    "3. Code changes you attempted: diff intent, what they did, what worked.\n"
    "4. Things you ruled out, with reasons.\n"
    "5. The current plan / next-step intent.\n\n"
    "Format: terse bulleted markdown, <=2000 tokens. No preamble, no "
    "\"In summary...\" chatter. Start directly with the facts.\n\n"
    "<HISTORY>\n{history}\n</HISTORY>"
)


def _read_proxy_status_tokens(budget_status_path):
    """Read ``(cumulative_prompt_tokens, cumulative_chat_chars)`` from the
    proxy's JSON status file (sibling of ``status.md``).

    Returns ``(None, None)`` when:
      - ``budget_status_path`` is unset,
      - the sibling JSON file does not exist (first call before the proxy
        has logged any usage),
      - the JSON is empty or malformed,
      - the required fields are absent (older proxy versions before the
        F2o schema bump).
    A return of ``(0, 0)`` is distinguishable from ``(None, None)``:
    ``(0, 0)`` means the proxy is up but no chat completion has been
    intercepted yet (e.g. between leases); ``(None, None)`` means we have
    no observation at all and the caller should use the conservative
    fallback ratio.
    """
    if not budget_status_path:
        return None, None
    json_path = os.path.join(
        os.path.dirname(budget_status_path) or ".",
        _PROXY_STATUS_JSON_BASENAME,
    )
    try:
        with open(json_path) as fp:
            payload = json.load(fp)
    except (OSError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    prompt_tokens = payload.get("cumulative_prompt_tokens")
    chat_chars = payload.get("cumulative_chat_chars")
    if not isinstance(prompt_tokens, int) or not isinstance(chat_chars, int):
        return None, None
    return prompt_tokens, chat_chars


def _compression_ratio(budget_status_path):
    """Return a live ``chars / token`` ratio from the proxy's status file,
    or the conservative fallback when no observation is available.

    The ratio is floored at ``_COMPRESSION_RATIO_FLOOR`` so a transient
    under-report cannot make the compactor under-estimate token cost.
    """
    cum_prompt_tokens, cum_chat_chars = _read_proxy_status_tokens(budget_status_path)
    if (
        cum_prompt_tokens is None
        or cum_chat_chars is None
        or cum_prompt_tokens <= 0
        or cum_chat_chars <= 0
    ):
        return _COMPRESSION_RATIO_FALLBACK
    return max(cum_chat_chars / cum_prompt_tokens, _COMPRESSION_RATIO_FLOOR)


def _estimate_tokens(msg_history, input_msg, *, budget_status_path=None):
    """Estimate the token count for ``msg_history + input_msg``.

    When the cost proxy has logged any usage, the in-flight estimate is
    ``cumulative_prompt_tokens + (delta_chars / live_ratio)`` where
    ``live_ratio = cumulative_chat_chars / cumulative_prompt_tokens``
    floored at ``_COMPRESSION_RATIO_FLOOR``. The delta covers any
    not-yet-logged fresh content (a freshly-appended tool result, the
    current user message, etc.) by dividing its chars by the ratio.

    When the proxy has no observation yet (first call of the lease), we
    fall back to ``chars / _COMPRESSION_RATIO_FALLBACK`` so the trigger
    over-estimates and fires early rather than late.
    """
    cum_prompt_tokens, cum_chat_chars = _read_proxy_status_tokens(budget_status_path)
    history_chars = sum(len(m.get("content", "")) for m in msg_history)
    input_chars = len(input_msg or "")
    total_chars = history_chars + input_chars
    if cum_prompt_tokens and cum_chat_chars:
        ratio = max(cum_chat_chars / cum_prompt_tokens, _COMPRESSION_RATIO_FLOOR)
        delta_chars = max(0, total_chars - cum_chat_chars)
        return int(cum_prompt_tokens + delta_chars / ratio)
    # Pre-proxy fallback: conservative ratio so estimate overshoots.
    return int(total_chars / _COMPRESSION_RATIO_FALLBACK)


def _resolve_compaction_config(catalog, model):
    """Return (enabled, soft_cap, summary_max_tok) for the active model.

    Soft cap is derived from the catalog entry (context_window_tokens,
    max_output_tokens) when available; falls back to 128000/4096 to
    match the Nemotron defaults documented in the project YAMLs. Env
    overrides apply on top.
    """
    enabled = os.environ.get("HYPERAGENTS_COMPACTION_ENABLED", "1") != "0"
    soft_fraction = float(
        os.environ.get("HYPERAGENTS_COMPACTION_SOFT_FRACTION", "0.85")
    )
    summary_max_tok = int(
        os.environ.get("HYPERAGENTS_COMPACTION_SUMMARY_MAX_TOK", "2500")
    )
    safety_margin_tok = int(
        os.environ.get("HYPERAGENTS_COMPACTION_SAFETY_MARGIN_TOK", "4000")
    )

    entry = None
    for e in catalog or []:
        if not isinstance(e, dict):
            continue
        if e.get("model") == model or e.get("id") == model:
            entry = e
            break
    context_window = int((entry or {}).get("context_window_tokens", 128000))
    max_output = int((entry or {}).get("max_output_tokens", 4096))
    soft_cap = int((context_window - max_output - safety_margin_tok) * soft_fraction)
    return enabled, soft_cap, summary_max_tok


def _is_tool_result_message(message):
    """Last-msg-is-tool-result probe: starts with `<json>` and contains `tool_output`."""
    if not isinstance(message, dict):
        return False
    content = message.get("content", "")
    return isinstance(content, str) and content.lstrip().startswith("<json>") and (
        "tool_output" in content
    )


def _truncate_to_tokens(text, max_tokens):
    """Trim ``text`` to ~max_tokens via the char/4 heuristic, suffixing an ellipsis."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def _maybe_compact_history(
    msg_history,
    input_msg,
    *,
    model,
    catalog,
    reasoning_effort,
    logging,
    summarize_fn=None,
    budget_status_path=None,
):
    """Compact ``msg_history`` in place when the estimated token count exceeds
    the model's soft cap. Returns the (possibly new) msg_history list.

    Algorithm (per the LLM-summarization context-compaction contract):

    - Estimate ``est_tokens`` = ``_estimate_tokens(msg_history, input_msg,
      budget_status_path=budget_status_path)`` -- proxy-derived live
      ratio when available, conservative ``chars/3`` fallback otherwise.
    - If under ``soft_cap`` or compaction disabled, return as-is.
    - Keep ``msg_history[0]`` (initial user message) verbatim.
    - Keep ``msg_history[-1]`` IF it looks like a tool result.
    - Fire ONE summarization call to the same model with an empty
      ``msg_history`` to compress the span between them.
    - Replace the span with a single ``role=assistant`` message wrapping
      ``<COMPACTED HISTORY>``.
    - If post-compaction the estimate still exceeds the soft cap (e.g. a
      single tool result is itself huge), raise ``RuntimeError``.
    - If the summarization call raises, log and fall through with the
      original history.

    The compaction message role MUST be ``"assistant"``. Compaction events
    are surfaced through the chat history (which run-observability layer
    15 watches) by virtue of the ``<COMPACTED HISTORY>`` marker landing in
    the persisted transcript on the very next ``get_response_from_llm``
    call.
    """
    enabled, soft_cap, summary_max_tok = _resolve_compaction_config(catalog, model)
    if not enabled:
        return msg_history
    if not msg_history:
        return msg_history

    est_tokens = _estimate_tokens(
        msg_history, input_msg, budget_status_path=budget_status_path
    )
    if est_tokens <= soft_cap:
        return msg_history

    if len(msg_history) < 2:
        # Only the initial user message present; nothing to summarize.
        return msg_history

    keep_tail = _is_tool_result_message(msg_history[-1]) and len(msg_history) >= 3
    span = msg_history[1:-1] if keep_tail else msg_history[1:]
    if not span:
        return msg_history

    history_blob = "\n\n".join(
        m.get("content", "") for m in span if isinstance(m, dict)
    )
    summary_prompt = _COMPACTION_SUMMARY_PROMPT.format(history=history_blob)

    call_fn = summarize_fn or get_response_from_llm
    try:
        logging(
            f"COMPACTION: est_tokens={est_tokens} > soft_cap={soft_cap}; "
            f"summarizing {len(span)} message(s) of span"
        )
        summary, _summary_history, _info = call_fn(
            msg=summary_prompt,
            model=model,
            msg_history=[],
            reasoning_effort=reasoning_effort,
            max_tokens=summary_max_tok,
        )
    except Exception as exc:  # rate-limit, 5xx, summary-call overflow, etc.
        logging(f"compaction failed: {exc}")
        return msg_history

    if not isinstance(summary, str):
        summary = "" if summary is None else str(summary)
    summary = _truncate_to_tokens(summary, summary_max_tok)
    compacted = {
        "role": "assistant",
        "content": f"<COMPACTED HISTORY>\n{summary}\n</COMPACTED HISTORY>",
    }

    new_history = [msg_history[0], compacted]
    if keep_tail:
        new_history.append(msg_history[-1])

    post_tokens = _estimate_tokens(
        new_history, input_msg, budget_status_path=budget_status_path
    )
    if post_tokens > soft_cap:
        raise RuntimeError(
            "compaction insufficient: post-compaction estimate "
            f"{post_tokens} tokens still exceeds soft_cap {soft_cap} "
            "(likely a single retained message is itself larger than the cap)."
        )
    logging(
        f"COMPACTION: compacted {len(span)} message(s) -> 1 summary "
        f"(post_tokens={post_tokens}, soft_cap={soft_cap}); marker "
        "<COMPACTED HISTORY> emitted."
    )
    return new_history


def _budget_status_prefix(budget_status_path):
    """Return the budget-status prefix to inject into the meta-agent prompt.

    Returns ``""`` only when the path is unset or the file is empty. If
    the host passes an explicit path, it must be readable from this
    process's view; missing or unreadable paths are configuration errors.
    """
    if not budget_status_path:
        return ""
    with open(budget_status_path) as f:
        text = f.read().strip()
    if not text:
        return ""
    return text + "\n\n"

def get_tooluse_prompt(tool_infos=[]):
    """
    Get the prompt for using the available tools.
    """
    # If no tools are available, return an empty string
    if not tool_infos or len(tool_infos) == 0:
        return ""
    # Create the prompt
    tools_available = [str(tool_info) for tool_info in tool_infos]
    tools_available = '\n\n'.join(tools_available) if tools_available else 'None'
    tooluse_prompt = """Here are the available tools:
```
{tools_available}
```

Use only one tool (if needed) in this format:
<json>
{{
    "tool_name": ...,
    "tool_input": ...
}}
</json>

ONLY USE ONE TOOL PER RESPONSE, AND STRICTLY FOLLOW THE FORMAT OF TOOL_NAME AND TOOL_INPUT ABOVE.
DO NOT HALLUCINATE OR MAKE UP ANYTHING.
""".format(tools_available=tools_available)
    return tooluse_prompt.strip()

def should_retry_tool_use(response, tool_uses=None):
    """
    Decide whether to send a corrective retry when the chat turn
    emitted no parseable tool call.

    Returns True for two malformed-output patterns; the caller surfaces
    a corrective error message that the model can incorporate on the
    next turn:

    1. **Output was truncated mid-tool.** Response is long (>=2000
       chars) AND has the json / tool_name / tool_input markers in
       the right order. The model started a tool call but ran out of
       output context before closing the JSON.
    2. **JSON body is malformed.** Same marker pattern as (1) but
       length is irrelevant. Nemotron 3 Super in particular tends to
       emit unterminated strings, missing closing braces, extra
       ``</json>``-style closers, or wrong-quote styles. Without a
       retry, the agent exits early after one bad call -- so an
       expansion that should produce 30 tool calls produces 1, and
       ``parent_agent_success`` ends up False with no model_patch.diff.
    """
    # If there are tool uses, we don't need to check for retry
    if tool_uses is not None and len(tool_uses) > 0:
        return False

    # Find positions of the markers
    json_pos = response.find("<json>")
    tool_name_pos = response.find("tool_name")
    tool_input_pos = response.find("tool_input")

    # If the response shows tool-use intent (markers present in the
    # right order) but no tool was parsed, retry. Length is informative
    # for the corrective message but is not a gate.
    if (
        json_pos != -1
        and tool_name_pos != -1
        and tool_input_pos != -1
        and json_pos < tool_name_pos < tool_input_pos
    ):
        return True

    # No retry
    return False

def check_for_tool_uses(response):
    """
    Checks if the response contains one or more tool calls in json code blocks.
    Returns a list of tool use dictionaries.

    F-class: balanced-brace scan instead of strict ``<json>...</json>`` regex.
    Some models (e.g. Nemotron) emit ``</script>`` (or other tag mismatches)
    as the closing token, which breaks the original regex and silently
    returns no tool uses -- making every meta-agent expansion a no-op. The
    scanner here only requires the ``<json>`` opening hint and then a
    syntactically balanced JSON object; the closing tag is ignored.
    """
    tool_uses = []
    pos = 0
    decoder = json.JSONDecoder()
    while True:
        start = response.find('<json>', pos)
        if start == -1:
            break
        brace = response.find('{', start + len('<json>'))
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(response, idx=brace)
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        if isinstance(obj, dict) and 'tool_name' in obj and 'tool_input' in obj:
            tool_uses.append(obj)
        pos = end

    return tool_uses if tool_uses else None

def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

def chat_with_agent(
    msg,
    model,
    msg_history=None,
    logging=print,
    tools_available=[],  # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,  # Whether to allow multiple tool calls in a single response
    max_tool_calls=40,  # Maximum number of tool calls allowed in a single response, -1 for unlimited
    reasoning_effort=None,
    catalog=None,  # F-class: model_catalog injected into tools that declare it (query_model).
    budget_status_path=None,
    workspace_root=None,
    current_gen=None,
):
    get_response_fn = get_response_from_llm
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

    try:
        # Load all tools
        all_tools = load_tools(
            logging=logging,
            names=tools_available,
            catalog=catalog,
            workspace_root=workspace_root,
            current_gen=current_gen,
        )
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        system_msg = f"{get_tooluse_prompt([tool['info'] for tool in all_tools])}\n\n"
        num_tool_calls = 0

        # Call API
        input_msg = _budget_status_prefix(budget_status_path) + system_msg + msg
        new_msg_history = _maybe_compact_history(
            new_msg_history,
            input_msg,
            model=model,
            catalog=catalog,
            reasoning_effort=reasoning_effort,
            logging=logging,
            budget_status_path=budget_status_path,
        )
        logging(f"Input: {repr(input_msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=input_msg,
            model=model,
            msg_history=new_msg_history,
            reasoning_effort=reasoning_effort,
        )
        logging(f"Output: {repr(response)}")
        # logging(f"Info: {repr(info)}")

        # Tool use
        tool_uses = check_for_tool_uses(response)
        retry_tool_use = should_retry_tool_use(response, tool_uses)
        while tool_uses or retry_tool_use:
            # Check for max tool calls
            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                logging("Error: Maximum number of tool calls reached.")
                break

            tool_msgs = []

            # Process tool uses
            if tool_uses:
                tool_uses = tool_uses if multiple_tool_calls else tool_uses[:1]
                for tool_use in tool_uses:
                    tool_name = tool_use['tool_name']
                    tool_input = tool_use['tool_input']
                    tool_output = process_tool_call(tools_dict, tool_name, tool_input)
                    num_tool_calls += 1
                    tool_msg = f'''<json>
    {{
        "tool_name": "{tool_name}",
        "tool_input": {tool_input},
        "tool_output": "{tool_output}"
    }}
    </json>'''.strip()
                    logging(f"Tool output: {repr(tool_msg)}")
                    tool_msgs.append(tool_msg)

            # Check for retry
            if retry_tool_use:
                # Distinguish "ran out of output context" from "JSON
                # body was malformed" so the model can correct
                # appropriately on the next turn. Both produce the same
                # observable shape (markers present, no tool parsed),
                # so we hint at both possibilities and quote the
                # common Nemotron 3 Super failure modes.
                err_msg = (
                    "Error: your previous response contained a tool-call "
                    "intent (``<json>`` markers + ``tool_name`` + ``tool_input``) "
                    "but no parseable tool was extracted. Common causes: "
                    "(1) the JSON body had an unterminated string (missing "
                    "closing quote), (2) a missing closing brace ``}``, "
                    "(3) extra ``</json>`` closing tags, (4) the response "
                    "was truncated mid-output. Re-emit ONE complete tool call "
                    "in the exact format ``<json>{\"tool_name\": ..., "
                    "\"tool_input\": ...}</json>`` with valid JSON and no "
                    "trailing tags."
                )
                logging(err_msg)
                tool_msgs.append(err_msg)

            # Get tool response
            input_msg = (
                _budget_status_prefix(budget_status_path)
                + system_msg
                + '\n\n'.join(tool_msgs)
            )
            new_msg_history = _maybe_compact_history(
                new_msg_history,
                input_msg,
                model=model,
                catalog=catalog,
                reasoning_effort=reasoning_effort,
                logging=logging,
            )
            logging(f"Input: {repr(input_msg)}")
            response, new_msg_history, info = get_response_fn(
                msg=input_msg,
                model=model,
                msg_history=new_msg_history,
                reasoning_effort=reasoning_effort,
            )
            logging(f"Output: {repr(response)}")
            # logging(f"Info: {repr(info)}")

            # Check for next tool use
            tool_uses = check_for_tool_uses(response)
            retry_tool_use = should_retry_tool_use(response, tool_uses)

    except Exception as e:
        logging(f"Error: {str(e)}")
        raise e

    return new_msg_history

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to use")
    args = parser.parse_args()

    msg = """hello"""
    new_msg_history = chat_with_agent(msg, model=args.model)

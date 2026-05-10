import json
import os
import re
import time
import urllib.error
import urllib.request

from agent.llm import get_response_from_llm
from agent.tools import load_tools

# F2h (recursive-scientist Tier 3 per-turn cost visibility): when the
# host has spawned a per-expansion cost-proxy and a wall-clock cap, two
# env vars surface the live source and the deadline so the meta-agent
# can self-regulate before the proxy's 402 hard-cap fires:
#
#   RS_COST_PROXY_HEALTH_URL  HTTP GET that returns
#                             {"calls", "cumulative_cost_usd",
#                              "budget_usd", ...}.
#   RS_EXPANSION_DEADLINE_TS  Unix timestamp at which the in-container
#                             wall-clock kill fires.
#
# Either env var absent => the corresponding fragment is omitted; both
# absent => the prefix is empty and behavior matches the pre-Tier-3
# default. The HTTP probe is wrapped in a tight timeout so a stalled
# proxy does not deadlock the meta-agent.
_HEALTH_URL_ENV = "RS_COST_PROXY_HEALTH_URL"
_DEADLINE_ENV = "RS_EXPANSION_DEADLINE_TS"
_HEALTH_PROBE_TIMEOUT_SEC = 1.0


def _budget_line():
    """Return the per-turn budget prefix or an empty string.

    Format::

        Remaining budget: $X (of $Y). Remaining wall-clock: T seconds (of B).
        Catalog cost-per-1M-tokens for each model is in MODEL_CATALOG.md.

    Either fragment is dropped when its env var is unset; if both are
    unset the line itself is omitted (empty string), keeping pre-Tier-3
    behavior identical.
    """
    health_url = os.environ.get(_HEALTH_URL_ENV)
    deadline_ts = os.environ.get(_DEADLINE_ENV)
    parts = []
    if health_url:
        try:
            with urllib.request.urlopen(health_url, timeout=_HEALTH_PROBE_TIMEOUT_SEC) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            cap = payload.get("budget_usd")
            cum = payload.get("cumulative_cost_usd")
            if cap is not None and cum is not None:
                remaining = max(0.0, float(cap) - float(cum))
                parts.append(
                    f"Remaining budget: ${remaining:.2f} (of ${float(cap):.2f})."
                )
    if deadline_ts:
        try:
            deadline = float(deadline_ts)
            now = time.time()
            remaining_sec = int(max(0.0, deadline - now))
            parts.append(f"Remaining wall-clock: {remaining_sec} seconds.")
        except ValueError:
            pass
    if not parts:
        return ""
    return (
        " ".join(parts)
        + " Catalog cost-per-1M-tokens for each model is in MODEL_CATALOG.md.\n\n"
    )

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
):
    get_response_fn = get_response_from_llm
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

    try:
        # Load all tools
        all_tools = load_tools(logging=logging, names=tools_available, catalog=catalog)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        system_msg = f"{get_tooluse_prompt([tool['info'] for tool in all_tools])}\n\n"
        num_tool_calls = 0

        # Call API
        # F2h: prepend the per-turn budget line freshly each turn so the
        # meta-agent always sees the LIVE remaining budget (the proxy's
        # in-memory accumulator) and the LIVE remaining wall-clock.
        logging(f"Input: {repr(msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=_budget_line() + system_msg + msg,
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
            # F2h: prepend the per-turn budget line freshly each turn.
            response, new_msg_history, info = get_response_fn(
                msg=_budget_line() + system_msg + '\n\n'.join(tool_msgs),
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

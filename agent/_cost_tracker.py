"""F-class minimal in-container LLM cost logger.

Mirrors the host-side ``recursive_scientist.llm.cost_tracker`` JSONL
schema so the host's ``rebuild_cost_md`` (called after each generation)
can ingest rows produced inside the container. The container does NOT
render ``cost.md`` -- it only appends. The host re-renders ``cost.md``
from the merged JSONL between generations.

Each row carries: ts, model, model_family, reasoning_effort,
input_tokens, output_tokens, reasoning_tokens, input_cost_usd,
output_cost_usd, cost_usd, running_total_usd. ``running_total_usd`` is
NOT computed here -- it's left as 0.0 and the host's renderer
reconstructs running totals on read. (Avoids needing in-memory state
inside the container.)
"""
import contextlib
import json
import os
import time

import litellm

_LOG_PATH = None


def _model_family(full):
    head = str(full).split("@", 1)[0]
    return head.rsplit("/", 1)[-1]


def _extract_effort(kwargs):
    if not isinstance(kwargs, dict):
        return None
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict):
        v = extra.get("reasoning_effort")
        if v is not None:
            return str(v)
    v = kwargs.get("reasoning_effort")
    return None if v is None else str(v)


def _on_completion(kwargs, response_obj, start_time, end_time):
    if _LOG_PATH is None:
        return
    with contextlib.suppress(Exception):
        usage = getattr(response_obj, "usage", None)
        if usage is None:
            return
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        reasoning_tokens = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            rt = getattr(details, "reasoning_tokens", None)
            if rt is None and isinstance(details, dict):
                rt = details.get("reasoning_tokens")
            if rt is None and hasattr(details, "model_dump"):
                rt = details.model_dump().get("reasoning_tokens")
            reasoning_tokens = int(rt or 0)
        model = getattr(response_obj, "model", None) or kwargs.get("model", "<unknown>")
        family = _model_family(model)
        input_cost_usd = 0.0
        output_cost_usd = 0.0
        with contextlib.suppress(Exception):
            ic, oc = litellm.cost_per_token(
                model=str(model),
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            input_cost_usd = float(ic or 0.0)
            output_cost_usd = float(oc or 0.0)
        os.makedirs(os.path.dirname(_LOG_PATH) or ".", exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "model": str(model),
                "model_family": family,
                "reasoning_effort": _extract_effort(kwargs),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "input_cost_usd": input_cost_usd,
                "output_cost_usd": output_cost_usd,
                "cost_usd": input_cost_usd + output_cost_usd,
                "running_total_usd": 0.0,
            }) + "\n")


def init_cost_tracker(log_path):
    global _LOG_PATH
    _LOG_PATH = log_path
    cb_list = list(litellm.success_callback or [])
    if _on_completion not in cb_list:
        cb_list.append(_on_completion)
        litellm.success_callback = cb_list

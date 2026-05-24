import backoff
import os
import socket
from typing import Tuple
import requests
import litellm
import litellm.exceptions
from dotenv import load_dotenv
import json

load_dotenv()

socket.setdefaulttimeout(int(os.environ.get("HYPERAGENTS_LLM_TIMEOUT_S", "300")))

MAX_TOKENS = 32768

litellm.drop_params=True
# F2e: disable LiteLLM's built-in retries so the ``backoff`` decorator
# below is the sole retry authority. Default num_retries=2 causes
# thundering-herd amplification when many sibling threads share a
# rate-limited upstream (NVIDIA Inference's AWS-WAF CPE_RateLimit_IP
# returns 429 with no Retry-After).
litellm.num_retries = 0

# Models that accept the OpenAI-style ``reasoning_effort`` parameter
# (low | medium | high). For other models, the param is dropped silently
# rather than blindly injected via ``extra_body`` (which would have no
# effect server-side and could confuse downstream telemetry).
# Match against the litellm-routed model name (after stripping the
# ``openai/`` provider prefix) — substrings, not full equality, so
# ``gpt-5-mini``, ``gpt-5.4-mini``, ``gpt-5.5-codex`` etc. all qualify.
_REASONING_EFFORT_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)


def _supports_reasoning_effort(litellm_model: str) -> bool:
    """Whether the model honors the OpenAI ``reasoning_effort`` param."""
    return any(p in litellm_model for p in _REASONING_EFFORT_PREFIXES)


def parse_openai_endpoint_model(model: str) -> tuple[str, dict[str, str]]:
    if "@" not in model:
        return model, {}

    routed_model, api_base = (part.strip() for part in model.rsplit("@", 1))
    if not routed_model:
        raise ValueError("Endpoint-routed model must include a model before '@'")
    if not api_base:
        raise ValueError("Endpoint-routed model must include an API base after '@'")
    return routed_model, {
        "api_base": api_base,
        "api_key": os.getenv("OPENAI_API_KEY", "local"),
    }


# F2e: WAF- and 504-storm-survivable retry policy.
# - max_time default 21600s (6h), env HYPERAGENTS_LLM_RETRY_BUDGET_S.
#   NVIDIA Inference 504 outages can outlast the prior 1h budget;
#   6h covers observed worst cases without giving up on transient infra.
# - max_value default 600s, env HYPERAGENTS_LLM_RETRY_MAX_WAIT_S.
#   10-min sleep ceiling lets the bucket drain across the 6h window.
# - jitter=backoff.full_jitter: uniform in [0, current_delay].
#   Eliminates thundering-herd at 1/2/4/8s with 16 sibling threads.
# - BadRequestError and AuthenticationError DO NOT retry: 400s/401s
#   are deterministic failures; burning the retry budget is wasted lease.
@backoff.on_exception(
    backoff.expo,
    (
        requests.exceptions.RequestException,
        json.JSONDecodeError,
        KeyError,
        litellm.exceptions.RateLimitError,
        litellm.exceptions.APIError,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.Timeout,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.ServiceUnavailableError,
    ),
    max_time=int(os.environ.get("HYPERAGENTS_LLM_RETRY_BUDGET_S", "21600")),
    max_value=int(os.environ.get("HYPERAGENTS_LLM_RETRY_MAX_WAIT_S", "600")),
    jitter=backoff.full_jitter,
    giveup=lambda exc: isinstance(
        exc,
        (
            litellm.exceptions.BadRequestError,
            litellm.exceptions.AuthenticationError,
        ),
    ),
)
def get_response_from_llm(
    msg: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    max_continuation_rounds: int = 0,
    msg_history=None,
    reasoning_effort: str | None = None,
    extra_body: dict | None = None,
    allow_truncated_response: bool = False,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []

    # Convert text to content, compatible with LITELLM API
    msg_history = [
        {**msg, "content": msg.pop("text")} if "text" in msg else msg
        for msg in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # Build kwargs - handle model-specific requirements
    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
    }
    litellm_model, endpoint_kwargs = parse_openai_endpoint_model(model)
    completion_kwargs["model"] = litellm_model
    completion_kwargs.update(endpoint_kwargs)

    # GPT-5 and GPT-5-mini only support default temperature (1), skip it
    # GPT-5.2 supports temperature
    if litellm_model in ["openai/gpt-5", "openai/gpt-5-mini"]:
        pass  # Don't set temperature
    else:
        completion_kwargs["temperature"] = temperature

    # GPT-5 models require max_completion_tokens instead of max_tokens
    if "gpt-5" in litellm_model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    else:
        # Claude Haiku has a 4096 token limit.
        if "claude-3-haiku" in litellm_model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    # ``litellm.drop_params=True`` strips unknown top-level kwargs before they
    # hit the wire. ``extra_body`` is the litellm-blessed escape hatch: its
    # contents go into the request JSON body verbatim. We only inject
    # ``reasoning_effort`` for models that document support for it; for
    # everything else the kwarg is silently a no-op so callers can pass it
    # uniformly across a heterogeneous fleet.
    request_extra_body = dict(extra_body or {})
    if reasoning_effort and _supports_reasoning_effort(litellm_model):
        request_extra_body.setdefault("reasoning_effort", reasoning_effort)
    if request_extra_body:
        completion_kwargs["extra_body"] = request_extra_body

    completion_kwargs.setdefault(
        "timeout", int(os.environ.get("HYPERAGENTS_LLM_TIMEOUT_S", "300"))
    )
    response = litellm.completion(**completion_kwargs)
    choice = response['choices'][0]  # pyright: ignore
    msg_obj = choice['message']
    response_text = msg_obj.get('content')
    # Reasoning models (Nemotron 3 Super, o1-style) populate 'reasoning_content'
    # separately from 'content'. If 'content' is empty we surface the reasoning
    # so the parser downstream sees something rather than ``None``.
    if not isinstance(response_text, str) or not response_text.strip():
        reasoning = msg_obj.get('reasoning_content')
        if isinstance(reasoning, str) and reasoning.strip():
            response_text = reasoning
    # F2d: continuation pass on output-cap truncation. Continuations are
    # explicit opt-in via ``max_continuation_rounds``; the default is zero
    # so a model catalog output ceiling never turns into extra calls.
    continuation_rounds = 0
    while (
        choice.get('finish_reason') == 'length'
        and continuation_rounds < max_continuation_rounds
    ):
        continuation_rounds += 1
        # Stitch: append assistant's partial, ask to continue verbatim.
        cont_msgs = new_msg_history + [
            {"role": "assistant", "content": response_text or ""},
            {
                "role": "user",
                "content": (
                    "Your previous response was cut off mid-output (you hit "
                    "the max-tokens cap). Continue EXACTLY from where you "
                    "stopped, in the same format, without repeating any "
                    "tokens you already produced. Do not restart, do not "
                    "summarize. Just continue."
                ),
            },
        ]
        cont_kwargs = dict(completion_kwargs)
        cont_kwargs["messages"] = cont_msgs
        response = litellm.completion(**cont_kwargs)
        choice = response['choices'][0]  # pyright: ignore
        msg_obj = choice['message']
        chunk = msg_obj.get('content')
        if not isinstance(chunk, str) or not chunk.strip():
            chunk = msg_obj.get('reasoning_content') or ""
        # Concatenate the chunk; the model is asked NOT to repeat the
        # prefix, so naive concatenation is correct in the common case.
        response_text = (response_text or "") + chunk
    if choice.get('finish_reason') == 'length':
        if allow_truncated_response:
            new_msg_history.append({"role": "assistant", "content": response_text})
            new_msg_history = [
                {**msg, "text": msg.pop("content")} if "content" in msg else msg
                for msg in new_msg_history
            ]
            return response_text, new_msg_history, {"finish_reason": "length"}
        raise RuntimeError(
            f"truncated response from {litellm_model}: finish_reason=length "
            f"after {continuation_rounds} continuation rounds "
            f"(max_tokens={completion_kwargs.get('max_tokens') or completion_kwargs.get('max_completion_tokens')}). "
            f"Output exceeds {max_continuation_rounds + 1}x the cap; "
            f"shorten the prompt, split the task, or raise "
            f"max_continuation_rounds for this call."
        )
    new_msg_history.append({"role": "assistant", "content": response_text})

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**msg, "text": msg.pop("content")} if "content" in msg else msg
        for msg in new_msg_history
    ]

    return response_text, new_msg_history, {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to test")
    parser.add_argument("--msg", default="Hello there!", help="Message to send")
    args = parser.parse_args()

    output_msg, msg_history, info = get_response_from_llm(args.msg, model=args.model)
    print(output_msg)

import backoff
import os
from typing import Tuple
import requests
import litellm
from dotenv import load_dotenv
import json

load_dotenv()

MAX_TOKENS = 16384

litellm.drop_params=True


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


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
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
        # Claude Haiku has a 4096 token limit
        if "claude-3-haiku" in litellm_model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    response = litellm.completion(**completion_kwargs)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

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

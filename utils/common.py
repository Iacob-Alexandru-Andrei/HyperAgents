import re
import json
import os


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _try_loads(s):
    """``json.loads`` without raising; returns ``None`` for any failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _try_loads_with_relaxation(s):
    """Parse ``s`` as JSON; on failure, retry once with trailing commas stripped.

    Trailing commas (``{"a": 1,}``) are valid in JSON5 / Python literals but
    rejected by ``json.loads``. Many LLMs emit them. Relaxation is one-shot
    and only fires when the raw parse failed AND the substitution made a
    real change — so a clean response pays zero extra cost.
    """
    v = _try_loads(s)
    if v is not None:
        return v
    relaxed = _TRAILING_COMMA.sub(r"\1", s)
    if relaxed != s:
        return _try_loads(relaxed)
    return None


def read_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

def extract_jsons(response):
    """Extract all JSON objects from a model response string, or ``None``.

    Tolerates: BOM-prefixed responses; ``<json>...</json>`` and
    ```` ```json ... ``` ```` and plain ```` ``` ... ``` ```` fence styles;
    JSON embedded in surrounding prose; trailing-comma violations
    (``{"a": 1,}``).
    """
    if not isinstance(response, str):
        return None
    patterns = [
        r'<json>(.*?)</json>',
        r'```json(.*?)```',
        # Plain fence that wraps a JSON object specifically. Constrained to
        # a leading ``{`` so it doesn't claim non-JSON code blocks.
        r'```\s*(\{.*?\})\s*```',
    ]
    extracted_jsons = []

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            obj = _try_loads_with_relaxation(match.strip())
            if obj is not None:
                extracted_jsons.append(obj)

    # F2g: reasoning models (e.g. Nemotron 3 Super) routinely produce raw
    # JSON without the ``<json>``/```json``` wrapper their system prompt
    # asked for. Fall back to (a) parsing the whole response as a single
    # JSON object, then (b) walk the response and let JSONDecoder.raw_decode
    # find each balanced ``{...}`` independently. The earlier greedy regex
    # ``\{.*\}`` would join an ``{example}`` and an ``{answer}`` into one
    # malformed concatenation; raw_decode handles each in isolation.
    if not extracted_jsons:
        # BOM-tolerant strip so a leading ``﻿`` from some endpoints
        # does not poison the first ``json.loads``.
        stripped = response.strip().lstrip("﻿")
        obj = _try_loads_with_relaxation(stripped)
        if obj is not None:
            extracted_jsons.append(obj)
        else:
            decoder = json.JSONDecoder()
            pos = 0
            n = len(stripped)
            while pos < n:
                start = stripped.find("{", pos)
                if start == -1:
                    break
                try:
                    obj, end = decoder.raw_decode(stripped, idx=start)
                except json.JSONDecodeError:
                    pos = start + 1
                    continue
                if isinstance(obj, dict):
                    extracted_jsons.append(obj)
                pos = end

    return extracted_jsons if extracted_jsons else None

def file_exist_and_not_empty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def load_json_file(file_path):
    """
    Load a JSON file and return its contents as a dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

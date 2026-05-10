import re
import json
import os


def read_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

def extract_jsons(response):
    """
    Extracts all JSON objects from the given response string.
    """
    patterns = [
        r'<json>(.*?)</json>',
        r'```json(.*?)```',
    ]
    extracted_jsons = []

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                extracted_json = json.loads(match.strip())
                extracted_jsons.append(extracted_json)
            except json.JSONDecodeError:
                continue  # Skip malformed JSON blocks

    # F2g: reasoning models (e.g. Nemotron 3 Super) routinely produce raw
    # JSON without the ``<json>``/```json``` wrapper their system prompt
    # asked for. Fall back to (a) parsing the whole response as a single
    # JSON object, then (b) walk the response and let JSONDecoder.raw_decode
    # find each balanced ``{...}`` independently. The earlier greedy regex
    # ``\{.*\}`` would join an ``{example}`` and an ``{answer}`` into one
    # malformed concatenation; raw_decode handles each in isolation.
    if not extracted_jsons and isinstance(response, str):
        stripped = response.strip()
        try:
            extracted_jsons.append(json.loads(stripped))
        except json.JSONDecodeError:
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

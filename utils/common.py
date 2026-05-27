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

    if not extracted_jsons:
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(response):
            start = response.find("{", idx)
            if start == -1:
                break
            try:
                extracted_json, end = decoder.raw_decode(response[start:])
            except json.JSONDecodeError:
                idx = start + 1
                continue
            if isinstance(extracted_json, dict):
                extracted_jsons.append(extracted_json)
            idx = start + end

    if not extracted_jsons:
        decision_match = re.search(
            r'"Decision"\s*:\s*"?(Accept|Reject)"?',
            response,
            re.IGNORECASE,
        )
        if decision_match:
            extracted_jsons.append({"Decision": decision_match.group(1).title()})

    return extracted_jsons if extracted_jsons else None

def file_exist_and_not_empty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def load_json_file(file_path):
    """
    Load a JSON file and return its contents as a dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

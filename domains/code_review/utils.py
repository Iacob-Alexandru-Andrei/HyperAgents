import json


QUESTION_ID = "question_id"
GROUND_TRUTH_KEY = "outcome"


def encode_patch_text(patch):
    return json.dumps(str(patch))


def decode_patch_text(patch):
    text = str(patch)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    return decoded if isinstance(decoded, str) else text


def format_input_dict(row):
    return {
        "domain": "code_review",
        "patch": decode_patch_text(row["patch"]),
    }

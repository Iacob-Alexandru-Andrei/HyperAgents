QUESTION_ID = "question_id"
GROUND_TRUTH_KEY = "outcome"
MODEL = "openai/nvidia/nvidia/nemotron-3-super-v3@https://inference-api.nvidia.com/v1"

def format_input_dict(row):
    # Extract the inputs for the task from the row
    return {
        "domain": "paper_review",
        "paper_text": row['paper_text'],
    }

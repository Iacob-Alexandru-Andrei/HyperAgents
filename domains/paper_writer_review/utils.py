QUESTION_ID = "question_id"
GROUND_TRUTH_KEY = "outcome"


def format_input_dict(row):
    return {
        "domain": "paper_writer_review",
        "paper_text": row["paper_text"],
    }

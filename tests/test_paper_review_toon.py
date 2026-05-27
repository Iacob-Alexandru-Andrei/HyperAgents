import json
import importlib
import sys
import types

import pandas as pd
from pandas.testing import assert_frame_equal


def _paper_review_df():
    return pd.DataFrame(
        [
            {
                "question_id": "q1",
                "paper_text": 'Title: CSV, TOON\nBody with comma, quote "x", and newline.',
                "outcome": "accept",
            },
            {
                "question_id": "q2",
                "paper_text": "Second paper\n\nContains multiple paragraphs and colon: value.",
                "outcome": "reject",
            },
        ]
    )


def test_toon_roundtrip_preserves_paper_review_strings(tmp_path):
    from domains._toon_io import read_toon, write_toon

    source = _paper_review_df()
    toon_path = tmp_path / "dataset.toon"

    write_toon(source, toon_path)
    decoded = read_toon(toon_path)

    assert_frame_equal(source, decoded, check_dtype=False)


def test_csv_to_toon_matches_csv_loader(tmp_path):
    from domains._toon_io import read_toon
    from domains.paper_review.csv_to_toon import convert_csv_to_toon

    source = _paper_review_df()
    csv_path = tmp_path / "dataset.csv"
    toon_path = tmp_path / "dataset.toon"
    source.to_csv(csv_path, index=False)

    convert_csv_to_toon(csv_path, toon_path)

    expected = pd.read_csv(csv_path, dtype=str)
    decoded = read_toon(toon_path)
    assert_frame_equal(expected, decoded, check_dtype=False)


def test_harness_csv_and_toon_predictions_match_with_stubbed_llm(tmp_path, monkeypatch):
    from domains._toon_io import write_toon

    source = _paper_review_df()
    csv_path = tmp_path / "dataset.csv"
    toon_path = tmp_path / "dataset.toon"
    source.to_csv(csv_path, index=False)
    write_toon(source, toon_path)

    hydra_stub = types.ModuleType("hydra")
    hydra_stub.compose = lambda *args, **kwargs: None
    hydra_stub.initialize_config_dir = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "hydra", hydra_stub)

    response_text = (
        "THOUGHT:\nStubbed review.\n\n"
        'REVIEW JSON:\n```json\n{"Decision": "Accept"}\n```'
    )

    fake_litellm = types.SimpleNamespace(drop_params=False)

    def fake_completion(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        return {
            "choices": [{"message": {"content": response_text}}],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        }

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("agent.llm", None)
    sys.modules.pop("agent.base_agent", None)
    sys.modules.pop("agent.llm_withtools", None)
    sys.modules.pop("domains.harness", None)

    from domains.harness import harness

    common_kwargs = {
        "agent_path": "baselines/ai_reviewer/agent.py",
        "domain": "paper_review",
        "num_samples": 2,
        "num_workers": 1,
        "output_dir": str(tmp_path),
    }
    csv_run = harness(
        **common_kwargs,
        dataset_path=str(csv_path),
        run_id="csv",
        token_log=str(tmp_path / "csv" / "token_usage.jsonl"),
    )
    toon_run = harness(
        **common_kwargs,
        dataset_path=str(toon_path),
        run_id="toon",
        token_log=str(tmp_path / "toon" / "token_usage.jsonl"),
    )

    csv_predictions = pd.read_csv(f"{csv_run}/predictions.csv", dtype=str)
    toon_predictions = pd.read_csv(f"{toon_run}/predictions.csv", dtype=str)
    assert_frame_equal(csv_predictions, toon_predictions, check_dtype=False)

    for run_name in ["csv", "toon"]:
        token_log = tmp_path / run_name / "token_usage.jsonl"
        records = [json.loads(line) for line in token_log.read_text().splitlines()]
        assert [record["question_id"] for record in records] == ["q1", "q2"]
        assert all(record["total_tokens"] > 0 for record in records)


def test_harness_import_for_paper_review_does_not_require_hydra():
    sys.modules.pop("domains.harness", None)
    sys.modules.pop("hydra", None)

    harness_module = importlib.import_module("domains.harness")

    assert hasattr(harness_module, "load_dataset_path")


def test_llm_splits_model_api_base_suffix(monkeypatch):
    captured = {}
    fake_litellm = types.SimpleNamespace(drop_params=False)

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("agent.llm", None)

    from agent.llm import get_response_from_llm

    get_response_from_llm(
        "hello",
        model="openai/nvidia/nvidia/nemotron-3-super-v3@https://inference-api.nvidia.com/v1",
        max_tokens=8,
    )

    assert captured["model"] == "openai/nvidia/nvidia/nemotron-3-super-v3"
    assert captured["api_base"] == "https://inference-api.nvidia.com/v1"


def test_extract_jsons_handles_unfenced_review_json():
    from utils.common import extract_jsons

    response = """THOUGHT:
The paper is weak.

REVIEW JSON:
{
  "Summary": "Short summary.",
  "Decision": "Reject"
}
"""

    assert extract_jsons(response)[-1]["Decision"] == "Reject"


def test_extract_jsons_falls_back_to_review_decision_field():
    from utils.common import extract_jsons

    response = """REVIEW JSON:
{
  "Questions": ["Malformed string],
  "Decision": Reject
}
"""

    assert extract_jsons(response)[-1]["Decision"] == "Reject"


def test_review_packet_keeps_core_sections_and_drops_back_matter():
    from domains.paper_review.prompt_packet import build_review_packet

    paper_text = """'Title: Efficient Widgets

Abstract: We introduce a widget method with strong results.

Section: INTRODUCTION
The introduction motivates the widget problem.

Section: METHOD
The method uses sparse widget routing.

Section: EXPERIMENTS
Experiments compare against strong baselines.

Section: REFERENCES
REFERENCES SHOULD DROP.

Figures:
FIGURE NOISE SHOULD DROP.

Formulas:
FORMULA NOISE SHOULD DROP.
"""

    packet = build_review_packet(paper_text, section_char_limit=120, max_sections=4)
    packet_text = json.dumps(packet)

    assert packet["title"] == "Efficient Widgets"
    assert packet["abstract"] == "We introduce a widget method with strong results."
    assert [section["name"] for section in packet["sections"]] == [
        "INTRODUCTION",
        "METHOD",
        "EXPERIMENTS",
    ]
    assert "REFERENCES SHOULD DROP" not in packet_text
    assert "FIGURE NOISE SHOULD DROP" not in packet_text
    assert "FORMULA NOISE SHOULD DROP" not in packet_text


def test_review_packet_handles_escaped_newlines_from_dataset():
    from domains.paper_review.prompt_packet import build_review_packet

    paper_text = (
        "'Title: Escaped Paper\\n\\n"
        "Abstract: Escaped abstract.\\n\\n"
        "Section: INTRODUCTION\\nIntro body.\\n\\n"
        "Section: METHOD\\nMethod body.\\n\\n"
        "Section: REFERENCES\\nReferences should drop."
    )

    packet = build_review_packet(paper_text, section_char_limit=120, max_sections=4)

    assert packet["title"] == "Escaped Paper"
    assert packet["abstract"] == "Escaped abstract."
    assert [section["name"] for section in packet["sections"]] == [
        "INTRODUCTION",
        "METHOD",
    ]


def test_review_packet_preserves_key_result_tables():
    from domains.paper_review.prompt_packet import build_review_packet

    paper_text = """Title: Table Paper

Abstract: We evaluate a useful method.

Section: INTRODUCTION
The paper states three contributions.

Section: RESULTS
The method performs well.

Figures:
Figure tab_0: 1
Type: table
Caption: Performance comparison on benchmark tasks.
Data: Method Accuracy F1 Baseline 70 68 Ours 84 82

Figure tab_1: 2
Type: table
Caption: Hyperparameter sweep.
Data: SHOULD DROP

Formulas:
FORMULA DUMP
"""

    packet = build_review_packet(
        paper_text,
        section_char_limit=120,
        table_char_limit=120,
        max_key_tables=2,
    )
    packet_text = json.dumps(packet)

    assert packet["key_tables"] == [
        {
            "caption": "Performance comparison on benchmark tasks.",
            "data": "Method Accuracy F1 Baseline 70 68 Ours 84 82",
        }
    ]
    assert "SHOULD DROP" not in packet_text
    assert "FORMULA DUMP" not in packet_text


def test_core_text_packet_keeps_raw_core_text_and_toon_sidecar():
    from domains.paper_review.prompt_packet import build_core_text_packet_parts

    paper_text = """Title: Core Paper

Abstract: The method improves benchmark accuracy.

Section: INTRODUCTION
The introduction explains the task.

Section: METHOD
The raw method prose should be preserved.

Section: RESULTS
Ours beats the baseline by 10 points.

Section: REFERENCES
References should not appear.

Figures:
Figure tab_0: 1
Type: table
Caption: Performance comparison on benchmark tasks.
Data: Method Accuracy Baseline 70 Ours 80

Figure fig_0: 2
Type: figure
Caption: Figure dump should not appear.

Formulas:
Formula dump should not appear.
"""

    parts = build_core_text_packet_parts(
        paper_text,
        section_char_limit=200,
        max_sections=4,
        table_char_limit=120,
    )

    assert "Section: METHOD" in parts["core_text"]
    assert "The raw method prose should be preserved." in parts["core_text"]
    assert "Ours beats the baseline by 10 points." in parts["core_text"]
    assert "References should not appear" not in parts["core_text"]
    assert "Figure dump should not appear" not in parts["core_text"]
    assert "Formula dump should not appear" not in parts["core_text"]
    assert "metadata_toon" in parts
    assert "Performance comparison on benchmark tasks." in parts["metadata_toon"]


def test_retrieval_packet_prefers_review_evidence_chunks():
    from domains.paper_review.prompt_packet import build_retrieval_packet

    paper_text = """Title: Retrieval Paper

Abstract: We propose a useful training method.

Section: INTRODUCTION
Background sentence without much review evidence.

Section: METHOD
The method introduces contrastive routing and a new objective.

Section: RESULTS
Experiments show higher accuracy than strong baselines.

Section: REFERENCES
References should not appear.

Figures:
Figure tab_0: 1
Type: table
Caption: Accuracy comparison against baselines.
Data: Method Accuracy Baseline 70 Ours 82
"""

    packet = build_retrieval_packet(
        paper_text,
        chunk_char_limit=160,
        max_chunks=2,
        table_char_limit=120,
    )
    packet_text = json.dumps(packet)

    assert [chunk["section"] for chunk in packet["evidence_chunks"]] == [
        "RESULTS",
        "METHOD",
    ]
    assert "strong baselines" in packet_text
    assert "contrastive routing" in packet_text
    assert "References should not appear" not in packet_text
    assert packet["key_tables"][0]["caption"] == "Accuracy comparison against baselines."


def test_neutral_toon_agent_omits_reject_if_unsure_instruction(monkeypatch, tmp_path):
    captured = {}
    fake_litellm = types.SimpleNamespace(drop_params=False)

    def fake_completion(**kwargs):
        captured["prompt"] = kwargs["messages"][-1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": 'REVIEW JSON:\n{"Decision": "Accept"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("agent.llm", None)
    sys.modules.pop("agent.base_agent", None)

    from baselines.ai_reviewer_toon_neutral.agent import TaskAgent

    agent = TaskAgent(
        model="fake-model",
        chat_history_file=str(tmp_path / "chat_history.md"),
    )
    prediction, _ = agent.forward(
        {
            "paper_text": "Title: Neutral\n\nAbstract: Useful.\n\nSection: RESULTS\nStrong result."
        }
    )

    assert prediction == "Accept"
    assert "paper_packet_toon" in captured["prompt"]
    assert "If a paper is bad or you are unsure" not in captured["prompt"]


def test_raw_neutral_agent_uses_raw_paper_without_reject_if_unsure(monkeypatch, tmp_path):
    captured = {}
    fake_litellm = types.SimpleNamespace(drop_params=False)

    def fake_completion(**kwargs):
        captured["prompt"] = kwargs["messages"][-1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": 'REVIEW JSON:\n{"Decision": "Accept"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("agent.llm", None)
    sys.modules.pop("agent.base_agent", None)

    from baselines.ai_reviewer_raw_neutral.agent import TaskAgent

    agent = TaskAgent(
        model="fake-model",
        chat_history_file=str(tmp_path / "chat_history.md"),
    )
    prediction, _ = agent.forward(
        {
            "paper_text": "Title: Raw Neutral\n\nAbstract: Useful.\n\nSection: RESULTS\nStrong result."
        }
    )

    assert prediction == "Accept"
    assert "Here is the paper you are asked to review:" in captured["prompt"]
    assert "Title: Raw Neutral" in captured["prompt"]
    assert "paper_packet_toon" not in captured["prompt"]
    assert "If a paper is bad or you are unsure" not in captured["prompt"]


def test_two_stage_agent_extracts_evidence_then_reviews(monkeypatch, tmp_path):
    captured = []
    fake_litellm = types.SimpleNamespace(drop_params=False)

    responses = [
        'EVIDENCE JSON:\n{"Summary": "Strong results.", "Decision-Relevant Evidence": ["Accuracy improves."]}',
        'REVIEW JSON:\n{"Decision": "Accept"}',
    ]

    def fake_completion(**kwargs):
        captured.append(kwargs["messages"][-1]["content"])
        content = responses[len(captured) - 1]
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": len(kwargs["messages"][-1]["content"].split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(kwargs["messages"][-1]["content"].split())
                + len(content.split()),
            },
        }

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("agent.llm", None)
    sys.modules.pop("agent.base_agent", None)

    from baselines.ai_reviewer_two_stage_toon.agent import TaskAgent

    paper_text = """Title: Two Stage

Abstract: Useful.

Section: RESULTS
Accuracy improves over baselines.

Section: REFERENCES
References should not appear.
"""
    agent = TaskAgent(
        model="fake-model",
        chat_history_file=str(tmp_path / "chat_history.md"),
    )
    prediction, _ = agent.forward({"paper_text": paper_text})

    assert prediction == "Accept"
    assert len(captured) == 2
    assert "evidence_packet_toon" in captured[0]
    assert "Decision-Relevant Evidence" in captured[1]
    assert "References should not appear" not in "\n".join(captured)


def test_toon_prompt_agent_sends_compact_packet(monkeypatch, tmp_path):
    captured = {}
    fake_litellm = types.SimpleNamespace(drop_params=False)

    def fake_completion(**kwargs):
        captured["prompt"] = kwargs["messages"][-1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": 'REVIEW JSON:\n{"Decision": "Accept"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("agent.llm", None)
    sys.modules.pop("agent.base_agent", None)

    from baselines.ai_reviewer_toon.agent import TaskAgent

    paper_text = """Title: Compact Test

Abstract: Short abstract.

Section: INTRODUCTION
Useful context.

Section: REFERENCES
RAW REFERENCES SHOULD NOT APPEAR.
"""
    agent = TaskAgent(
        model="fake-model",
        chat_history_file=str(tmp_path / "chat_history.md"),
    )

    prediction, _ = agent.forward({"paper_text": paper_text})

    assert prediction == "Accept"
    assert "paper_packet_toon" in captured["prompt"]
    assert "Compact Test" in captured["prompt"]
    assert "Penalize missing or truncated evidence" not in captured["prompt"]
    assert "RAW REFERENCES SHOULD NOT APPEAR" not in captured["prompt"]


def test_comparison_report_accepts_custom_labels(tmp_path):
    from scripts.compare_toon_runs import write_comparison

    raw_dir = tmp_path / "raw"
    toon_dir = tmp_path / "toon"
    raw_dir.mkdir()
    toon_dir.mkdir()
    report = {
        "overall_accuracy": 1.0,
        "accuracy_by_ground_truth": {
            "accept": {"precision": 1.0, "recall": 1.0},
        },
    }
    (raw_dir / "report.json").write_text(json.dumps(report))
    (toon_dir / "report.json").write_text(json.dumps(report))
    (raw_dir / "token_usage.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            }
        )
        + "\n"
    )
    (toon_dir / "token_usage.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "prompt_tokens": 40,
                "completion_tokens": 8,
                "total_tokens": 48,
            }
        )
        + "\n"
    )

    out = write_comparison(
        raw_dir,
        toon_dir,
        tmp_path / "comparison.md",
        title="Raw vs Prompt TOON",
        csv_label="raw",
        toon_label="prompt_toon",
    )

    text = out.read_text()
    assert "# Raw vs Prompt TOON" in text
    assert "| token_type | raw | prompt_toon | delta | delta_pct |" in text
    assert "raw_prompt_tokens" in text
    assert "prompt_toon_prompt_tokens" in text

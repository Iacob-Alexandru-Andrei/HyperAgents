from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.gl_utils import binary_outcomes_from_report, get_binary_outcomes


def _write_report(root: Path, domain: str, genid: object, split: str, report: dict) -> None:
    dirname = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
    path = root / f"gen_{genid}" / dirname / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_binary_outcomes_from_pass_fail_question_ids():
    assert binary_outcomes_from_report(
        {
            "question_ids_passed": ["a", "b"],
            "question_ids_failed": ["c"],
        }
    ) == [1, 1, 0]


def test_binary_outcomes_from_total_correct_shape():
    assert binary_outcomes_from_report({"total_correct": 2, "total": 5}) == [
        1,
        1,
        0,
        0,
        0,
    ]


def test_binary_outcomes_from_polyglot_submitted_shape():
    assert binary_outcomes_from_report(
        {
            "total_resolved_instances": 1,
            "total_submitted_instances": 3,
        }
    ) == [1, 0, 0]


def test_binary_outcomes_from_polyglot_per_run_shape():
    assert binary_outcomes_from_report(
        {
            "resolved_instances": 2,
            "submitted_instances": 4,
        }
    ) == [1, 1, 0, 0]


def test_binary_outcomes_returns_none_for_unknown_shape():
    assert binary_outcomes_from_report({"accuracy_score": 0.5}) is None


def test_binary_outcomes_reject_invalid_counts():
    with pytest.raises(ValueError, match="Invalid binary outcome counts"):
        binary_outcomes_from_report({"total_correct": 4, "total": 3})


def test_get_binary_outcomes_reads_existing_eval_report(tmp_path: Path):
    _write_report(
        tmp_path,
        "paper_review",
        7,
        "val",
        {"question_ids_passed": ["ok"], "question_ids_failed": ["bad"]},
    )

    assert get_binary_outcomes("paper_review", str(tmp_path), 7, split="val") == [1, 0]
    assert get_binary_outcomes("paper_review", str(tmp_path), 8, split="val") is None

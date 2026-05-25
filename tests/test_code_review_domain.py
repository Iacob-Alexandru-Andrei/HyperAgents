import pytest


def test_normalize_crave_row_maps_label_and_strips_metadata():
    from domains.code_review.prepare_crave_dataset import normalize_crave_row
    from domains.code_review.utils import decode_patch_text

    patch = "diff --git a/foo.py b/foo.py\n+print('ok')\n"
    row = {
        "patch": patch,
        "label": "APPROVE",
        "language": "Python",
        "reviewer": "hidden-from-agent",
    }

    record = normalize_crave_row(row, split="train", index=7)

    assert record == {
        "question_id": "crave_train_000007",
        "patch": '"diff --git a/foo.py b/foo.py\\n+print(\'ok\')\\n"',
        "outcome": "pass",
    }
    assert decode_patch_text(record["patch"]) == patch


def test_normalize_crave_row_rejects_unknown_labels():
    from domains.code_review.prepare_crave_dataset import normalize_crave_row

    with pytest.raises(ValueError, match="Unsupported CRAVE label"):
        normalize_crave_row({"patch": "diff", "label": "MAYBE"}, split="train", index=1)


def test_split_output_names_preserve_train_val_test_mapping():
    from domains.code_review.prepare_crave_dataset import (
        HF_TO_LOCAL_SPLITS,
        filtered_dataset_name,
        full_dataset_name,
    )

    assert HF_TO_LOCAL_SPLITS == {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    assert full_dataset_name("train") == "dataset_train.csv"
    assert full_dataset_name("val") == "dataset_val.csv"
    assert full_dataset_name("test") == "dataset_test.csv"
    assert filtered_dataset_name("train") == "dataset_filtered_100_train.csv"
    assert filtered_dataset_name("val") == "dataset_filtered_100_val.csv"
    assert filtered_dataset_name("test") == "dataset_filtered_100_test.csv"


def test_filtered_rows_balances_labels_and_caps_per_split():
    from domains.code_review.prepare_crave_dataset import filtered_rows

    rows = [
        {"question_id": f"pass_{i}", "patch": f"p{i}", "outcome": "pass"}
        for i in range(20)
    ] + [
        {"question_id": f"fail_{i}", "patch": f"f{i}", "outcome": "fail"}
        for i in range(20)
    ]

    subset = filtered_rows(rows, limit=10, seed=123)
    outcomes = [row["outcome"] for row in subset]

    assert len(subset) == 10
    assert outcomes.count("pass") == 5
    assert outcomes.count("fail") == 5
    assert subset == filtered_rows(rows, limit=10, seed=123)


def test_format_input_dict_exposes_only_domain_and_patch():
    from domains.code_review import utils

    patch = "diff --git a/a.py b/a.py\n+pass\n"
    row = {
        "question_id": "crave_train_000001",
        "patch": utils.encode_patch_text(patch),
        "outcome": "fail",
        "language": "Python",
    }

    assert utils.QUESTION_ID == "question_id"
    assert utils.GROUND_TRUTH_KEY == "outcome"
    assert utils.format_input_dict(row) == {
        "domain": "code_review",
        "patch": patch,
    }


def test_domain_utils_registers_code_review_as_accuracy_dataset():
    from utils import domain_utils

    assert "code_review" in domain_utils.HUMAN_PREFERENCE_DOMAINS
    assert "code_review" in domain_utils.ACCURACY_REPORT_DOMAINS
    assert domain_utils.get_domain_score_key("code_review") == "overall_accuracy"
    assert domain_utils.get_domain_splits("code_review") == ["train", "val"]
    assert domain_utils.get_domain_splits("code_review", eval_test=True) == [
        "train",
        "val",
        "test",
    ]
    assert domain_utils.can_domain_ensembled("code_review") is True
    assert domain_utils.get_domain_eval_subset("code_review") == "_filtered_100_train"
    assert domain_utils.get_domain_test_subset("code_review") == "_filtered_100_test"
    assert domain_utils.get_domain_stagedeval_samples("code_review") == 10
    assert domain_utils.get_domain_stagedeval_frac("code_review") == 10 / 100
    assert domain_utils.has_domain_val_subset("code_review") is True


def test_task_agent_has_code_review_pass_fail_contract():
    from task_agent import TaskAgent

    output_format, extract_field = TaskAgent.OUTPUT_FORMATS["code_review"]

    assert extract_field == "Verdict"
    assert '"Verdict": "pass" | "fail"' in output_format
    assert "use only \"pass\" or \"fail\"" in output_format.lower()


def test_harness_loads_committed_filtered_code_review_split():
    from domains.harness import get_dataset

    df = get_dataset("code_review", subset="_filtered_100_train")

    assert list(df.columns) == ["question_id", "patch", "outcome"]
    assert len(df) == 100
    assert set(df["outcome"]) == {"pass", "fail"}

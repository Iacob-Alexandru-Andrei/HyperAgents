from __future__ import annotations

from pathlib import Path
import csv


def test_domain_utils_registers_polyglot_as_train_val_domain():
    from utils import domain_utils

    assert domain_utils.get_domain_score_key("polyglot") == "accuracy_score"
    assert domain_utils.get_domain_splits("polyglot") == ["train", "val"]
    assert domain_utils.get_domain_splits("polyglot", eval_test=True) == [
        "train",
        "val",
        "test",
    ]
    assert domain_utils.can_domain_ensembled("polyglot") is False
    assert domain_utils.get_domain_eval_subset("polyglot") == ""
    assert domain_utils.get_domain_stagedeval_samples("polyglot") == 10
    assert domain_utils.get_domain_stagedeval_frac("polyglot") == 10 / 50
    assert domain_utils.has_domain_val_subset("polyglot") is True


def test_polyglot_subset_mapping_uses_small_train_medium_val():
    import generation_step

    assert generation_step._polyglot_subset_for_split("train", "") == "small"
    assert generation_step._polyglot_subset_for_split("val", "") == "medium"
    assert generation_step._polyglot_subset_for_split("test", "") == "medium"
    assert generation_step._polyglot_subset_for_split("train", "medium") == "medium"


def test_polyglot_task_list_prefers_staged_split_csv(tmp_path: Path):
    import generation_step

    staged_csv = tmp_path / "root" / "domains" / "polyglot" / "dataset_val.csv"
    staged_csv.parent.mkdir(parents=True)
    with staged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("instance_id", "problem_statement", "language")
        )
        writer.writeheader()
        writer.writerow(
            {
                "instance_id": "python__staged",
                "problem_statement": "fix staged",
                "language": "python",
            }
        )

    assert generation_step._polyglot_task_list_for_eval(
        str(tmp_path / "root"),
        "val",
        "",
    ) == ["python__staged"]


def test_polyglot_worker_routes_to_host_harness(monkeypatch, tmp_path: Path):
    import generation_step

    calls = []

    def fake_run_harness_polyglot(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        generation_step, "run_harness_polyglot", fake_run_harness_polyglot
    )

    generation_step._eval_polyglot_produced_agent(
        root_dir=str(tmp_path / "root"),
        output_dir=str(tmp_path / "outputs"),
        genid=7,
        model="openai/fake",
        split="val",
        eval_subset="",
        eval_samples=3,
        eval_workers=2,
        skip_staged_eval=True,
    )

    assert calls == [
        {
            "root_dir": str(tmp_path / "root"),
            "output_dir": str(tmp_path / "outputs"),
            "genid": 7,
            "model": "openai/fake",
            "split": "val",
            "eval_subset": "",
            "num_samples": 3,
            "max_workers": 2,
            "skip_staged_eval": True,
        }
    ]

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


def test_polyglot_task_list_fallback_uses_staged_root_subsets(tmp_path: Path):
    import json

    import generation_step

    subset = tmp_path / "root" / "domains" / "polyglot" / "subsets" / "medium.json"
    subset.parent.mkdir(parents=True)
    subset.write_text(json.dumps(["python__from_root"]), encoding="utf-8")

    assert generation_step._polyglot_task_list_for_eval(str(tmp_path / "root"), "val", "") == [
        "python__from_root"
    ]


def test_polyglot_metadata_path_uses_staged_root(tmp_path: Path):
    import generation_step

    assert generation_step._polyglot_metadata_path(str(tmp_path / "root")) == str(
        tmp_path / "root" / "domains" / "polyglot" / "polyglot_benchmark_metadata.json"
    )


def test_polyglot_build_image_copies_repo_from_staged_root(tmp_path: Path):
    from domains.polyglot.docker_build import build_image

    staged_repo = tmp_path / "root" / "domains" / "polyglot"
    staged_repo.mkdir(parents=True)
    (staged_repo / "sentinel.txt").write_text("from staged root", encoding="utf-8")
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    class FakeAPI:
        def build(self, **_kwargs):
            yield {"stream": "done"}

    class FakeClient:
        api = FakeAPI()

    build_image(
        image_name="fake-polyglot",
        setup_scripts={},
        dockerfile="FROM scratch\n",
        platform="linux/amd64",
        client=FakeClient(),
        build_dir=build_dir,
        repo="domains/polyglot",
        repo_root=tmp_path / "root",
    )

    assert (build_dir / "domains" / "polyglot" / "sentinel.txt").read_text(
        encoding="utf-8"
    ) == "from staged root"


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

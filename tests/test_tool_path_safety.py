"""Cross-tool safety matrix.

Proves the Phase 7 hard requirement: read-only meta-agent tools cannot
read held-out / test / hard-pool future data, budget internals, sibling
gen outputs, or evaluator telemetry. The matrix below is the explicit
enumeration of (forbidden path) x (tool) pairs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent.tools import load_tools
from agent.tools.file_tree import file_tree
from agent.tools.py_check import py_check
from agent.tools.read_file import read_file
from agent.tools.search import search


# (test_id, relative_path). Each path is one violation class from the plan.
FORBIDDEN_PATHS: list[tuple[str, str]] = [
    ("budget", "budget/lease.json"),
    ("eval_val", "gen_3/_eval_val/predictions.csv"),
    ("eval_test", "gen_3/_eval_test/scores.json"),
    ("paper_review_hard", "paper_review_hard/dataset.csv"),
    ("epoch_data", "epoch_data/critic/epochs/0/agent.py"),
    ("sibling_gen", "gen_5/output.txt"),  # not the agent's current gen
    ("pending", "_pending/gen_4/scratch.json"),
    ("rs_scratch", "_rs_scratch/worker_2/temp.txt"),
    ("git", ".git/HEAD"),
    ("path_escape", "../etc/passwd"),
]


def _materialize(tmp_path: Path, relative_path: str) -> None:
    """Create the file (and parent directories) inside ``tmp_path`` so the
    tool's path-safety check fires before any "missing" error."""
    if relative_path.startswith(".."):
        # Path-escape targets live outside the workspace; do not create them.
        return
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("forbidden content", encoding="utf-8")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_refuses_budget_path(tmp_path: Path) -> None:
    _materialize(tmp_path, "budget/lease.json")
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=tmp_path, relative_path="budget/lease.json")


def test_read_file_refuses_eval_val(tmp_path: Path) -> None:
    _materialize(tmp_path, "gen_3/_eval_val/predictions.csv")
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=tmp_path, relative_path="gen_3/_eval_val/predictions.csv")


def test_read_file_refuses_paper_review_hard(tmp_path: Path) -> None:
    _materialize(tmp_path, "paper_review_hard/dataset.csv")
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=tmp_path, relative_path="paper_review_hard/dataset.csv")


def test_read_file_refuses_epoch_data(tmp_path: Path) -> None:
    _materialize(tmp_path, "epoch_data/critic/epochs/0/agent.py")
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=tmp_path, relative_path="epoch_data/critic/epochs/0/agent.py")


def test_read_file_refuses_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=tmp_path, relative_path="../etc/passwd")


# ---------------------------------------------------------------------------
# Forbidden matrix: every tool, every forbidden path. Parameterized so the
# safety surface is enumerated explicitly.
# ---------------------------------------------------------------------------


def _invoke_for_tool(name: str, *, root: Path, relative_path: str) -> None:
    if name == "read_file":
        read_file(root_dir=root, relative_path=relative_path)
    elif name == "file_tree":
        file_tree(root_dir=root, relative_path=relative_path)
    elif name == "search":
        search(root_dir=root, pattern="anything", relative_path=relative_path)
    elif name == "py_check":
        py_check(root_dir=root, relative_path=relative_path)
    else:
        raise AssertionError(f"unknown tool: {name}")


@pytest.mark.parametrize(
    ("path_id", "relative_path"),
    FORBIDDEN_PATHS,
    ids=[p[0] for p in FORBIDDEN_PATHS],
)
def test_file_tree_refuses_forbidden(tmp_path: Path, path_id: str, relative_path: str) -> None:
    _materialize(tmp_path, relative_path)
    with pytest.raises(ValueError, match="forbidden"):
        _invoke_for_tool("file_tree", root=tmp_path, relative_path=relative_path)


@pytest.mark.parametrize(
    ("path_id", "relative_path"),
    FORBIDDEN_PATHS,
    ids=[p[0] for p in FORBIDDEN_PATHS],
)
def test_search_refuses_forbidden(tmp_path: Path, path_id: str, relative_path: str) -> None:
    _materialize(tmp_path, relative_path)
    with pytest.raises(ValueError, match="forbidden"):
        _invoke_for_tool("search", root=tmp_path, relative_path=relative_path)


@pytest.mark.parametrize(
    ("path_id", "relative_path"),
    FORBIDDEN_PATHS,
    ids=[p[0] for p in FORBIDDEN_PATHS],
)
def test_py_check_refuses_forbidden(tmp_path: Path, path_id: str, relative_path: str) -> None:
    _materialize(tmp_path, relative_path)
    with pytest.raises(ValueError, match="forbidden"):
        _invoke_for_tool("py_check", root=tmp_path, relative_path=relative_path)


# ---------------------------------------------------------------------------
# Discovery shape: prove load_tools surfaces every read-only tool with a
# bound workspace root, the meta-agent-facing contract.
# ---------------------------------------------------------------------------


def test_load_tools_discovers_all_four_read_only_tools(tmp_path: Path) -> None:
    discovered = {
        t["name"]
        for t in load_tools(
            logging=lambda _msg: None,
            names="all",
            workspace_root=tmp_path,
        )
    }
    assert {"read_file", "file_tree", "search", "py_check"}.issubset(discovered)


def test_load_tools_binds_workspace_root_into_callable(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("x = 1\n", encoding="utf-8")
    tools = load_tools(
        logging=lambda _msg: None,
        names=["read_file"],
        workspace_root=tmp_path,
    )
    assert len(tools) == 1
    entry = tools[0]
    assert entry["name"] == "read_file"
    fn = entry["function"]
    assert isinstance(fn, Callable)
    result = fn(relative_path="hello.py")
    assert result["content"] == "x = 1\n"


def test_tool_schema_does_not_expose_current_gen_knob(tmp_path: Path) -> None:
    tools = load_tools(
        logging=lambda _msg: None,
        names=["read_file", "file_tree", "search", "py_check"],
        workspace_root=tmp_path,
        current_gen="gen_2",
    )

    for entry in tools:
        schema = entry["info"]["input_schema"]
        assert "current_gen" not in schema.get("properties", {})


def test_load_tools_injects_current_gen_and_ignores_model_override(tmp_path: Path) -> None:
    (tmp_path / "gen_2").mkdir()
    (tmp_path / "gen_2" / "own.txt").write_text("mine", encoding="utf-8")
    (tmp_path / "gen_3").mkdir()
    (tmp_path / "gen_3" / "sibling.txt").write_text("theirs", encoding="utf-8")
    tools = load_tools(
        logging=lambda _msg: None,
        names=["read_file"],
        workspace_root=tmp_path,
        current_gen="gen_2",
    )
    fn = tools[0]["function"]

    result = fn(relative_path="gen_2/own.txt", current_gen="gen_3")
    assert result["content"] == "mine"
    with pytest.raises(ValueError, match="forbidden"):
        fn(relative_path="gen_3/sibling.txt", current_gen="gen_3")


# ---------------------------------------------------------------------------
# Symlink escape: prove containment uses .resolve() and is_relative_to.
# ---------------------------------------------------------------------------


def test_read_file_refuses_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        link = workspace / "link.txt"
        link.symlink_to(outside)

        with pytest.raises(ValueError, match="forbidden"):
            read_file(root_dir=workspace, relative_path="link.txt")
    finally:
        if outside.exists():
            outside.unlink()


# ---------------------------------------------------------------------------
# Current-gen allowlist: when current_gen is provided, that one gen_* is
# readable but siblings remain forbidden.
# ---------------------------------------------------------------------------


def test_read_file_allows_current_gen_but_blocks_siblings(tmp_path: Path) -> None:
    (tmp_path / "gen_2").mkdir()
    (tmp_path / "gen_2" / "own.txt").write_text("mine", encoding="utf-8")
    (tmp_path / "gen_3").mkdir()
    (tmp_path / "gen_3" / "sibling.txt").write_text("theirs", encoding="utf-8")

    result = read_file(
        root_dir=tmp_path,
        relative_path="gen_2/own.txt",
        current_gen="gen_2",
    )
    assert result["content"] == "mine"

    with pytest.raises(ValueError, match="forbidden"):
        read_file(
            root_dir=tmp_path,
            relative_path="gen_3/sibling.txt",
            current_gen="gen_2",
        )


def test_read_file_allows_lineage_train_artifacts_without_current_gen(tmp_path: Path) -> None:
    lineage_file = tmp_path / "lineage" / "gen_1" / "paper_review_eval" / "predictions.csv"
    lineage_file.parent.mkdir(parents=True)
    lineage_file.write_text("qid,pred\nq1,1\n", encoding="utf-8")

    result = read_file(
        root_dir=tmp_path,
        relative_path="lineage/gen_1/paper_review_eval/predictions.csv",
    )

    assert "q1" in result["content"]


def test_file_tree_does_not_prune_lineage_gen_dirs(tmp_path: Path) -> None:
    lineage_file = tmp_path / "lineage" / "gen_1" / "agent_output" / "model_patch.diff"
    lineage_file.parent.mkdir(parents=True)
    lineage_file.write_text("diff --git a/x b/x\n", encoding="utf-8")

    result = file_tree(root_dir=tmp_path, relative_path="lineage")
    paths = {entry["path"] for entry in result["entries"]}

    assert "lineage/gen_1" in paths
    assert "lineage/gen_1/agent_output" in paths

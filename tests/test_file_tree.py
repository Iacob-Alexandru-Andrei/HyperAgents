from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.file_tree import file_tree


def test_file_tree_lists_entries(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")

    result = file_tree(root_dir=tmp_path)

    paths = {e["path"] for e in result["entries"]}
    assert "src" in paths
    assert "src/a.py" in paths
    assert "src/b.py" in paths
    assert "README.md" in paths
    assert result["truncated"] is False


def test_file_tree_caps_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for i in range(1000):
        (workspace / f"f{i:04d}.txt").write_text("x", encoding="utf-8")

    result = file_tree(root_dir=tmp_path, relative_path="workspace", max_entries=500)

    assert len(result["entries"]) <= 500
    assert result["truncated"] is True


def test_file_tree_respects_max_depth(tmp_path: Path) -> None:
    # Build a depth-10 chain; depth cap = 2 should yield d1 and d1/d2 only.
    current = tmp_path
    for i in range(1, 11):
        current = current / f"d{i}"
        current.mkdir()
    (current / "deep.txt").write_text("x", encoding="utf-8")

    result = file_tree(root_dir=tmp_path, max_depth=2)
    paths = {e["path"] for e in result["entries"]}
    assert "d1" in paths
    assert "d1/d2" in paths
    assert "d1/d2/d3" not in paths


def test_file_tree_prunes_forbidden_subtrees(tmp_path: Path) -> None:
    (tmp_path / "budget").mkdir()
    (tmp_path / "budget" / "secret.json").write_text("{}", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text("ok", encoding="utf-8")

    result = file_tree(root_dir=tmp_path)
    paths = {e["path"] for e in result["entries"]}
    assert "src" in paths
    assert "src/ok.py" in paths
    assert "budget" not in paths
    assert all("budget" not in p.split("/") for p in paths)


def test_file_tree_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_tree(root_dir=tmp_path, relative_path="nope")


def test_file_tree_raises_on_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        file_tree(root_dir=tmp_path, relative_path="a.txt")

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.py_check import py_check


def test_py_check_passes_clean_file(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
    result = py_check(root_dir=tmp_path, relative_path="clean.py")
    assert result["ok"] is True
    assert result["message"] == "ok"
    assert result["path"] == "clean.py"


def test_py_check_fails_syntax_error(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def foo(:\n    pass\n", encoding="utf-8")
    result = py_check(root_dir=tmp_path, relative_path="broken.py")
    assert result["ok"] is False
    assert "SyntaxError" in result["message"]


def test_py_check_rejects_non_python_file(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("not python\n", encoding="utf-8")
    result = py_check(root_dir=tmp_path, relative_path="notes.md")
    assert result["ok"] is False
    assert "not a Python file" in result["message"]


def test_py_check_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        py_check(root_dir=tmp_path, relative_path="missing.py")


def test_py_check_raises_on_directory(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    with pytest.raises(IsADirectoryError):
        py_check(root_dir=tmp_path, relative_path="pkg")


def test_py_check_does_not_execute_module(tmp_path: Path) -> None:
    # If py_check executed this file it would raise RuntimeError. Since
    # compile() only parses, the call must succeed with ok=True.
    sentinel = tmp_path / "sentinel.txt"
    (tmp_path / "danger.py").write_text(
        f'open({str(sentinel)!r}, "w").write("EXECUTED")\n',
        encoding="utf-8",
    )
    result = py_check(root_dir=tmp_path, relative_path="danger.py")
    assert result["ok"] is True
    assert not sentinel.exists(), "py_check must not execute the source"

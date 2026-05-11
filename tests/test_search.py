from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.search import (
    DEFAULT_MAX_RESULTS,
    MAX_LINE_CHARS,
    search,
)


def test_search_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hello world\nfoo bar\nhello again\n", encoding="utf-8")

    result = search(root_dir=tmp_path, pattern="hello")

    lines = [(r["path"], r["line_number"], r["line"]) for r in result["results"]]
    assert ("a.py", 1, "hello world") in lines
    assert ("a.py", 3, "hello again") in lines
    assert result["truncated"] is False


def test_search_caps_results(tmp_path: Path) -> None:
    target = tmp_path / "many.txt"
    target.write_text("\n".join("match" for _ in range(500)) + "\n", encoding="utf-8")

    result = search(root_dir=tmp_path, pattern="match", max_results=DEFAULT_MAX_RESULTS)

    assert len(result["results"]) <= DEFAULT_MAX_RESULTS
    assert result["truncated"] is True


def test_search_truncates_long_lines(tmp_path: Path) -> None:
    long_line = "x" * 200
    (tmp_path / "long.txt").write_text(f"prefix {long_line} match\n", encoding="utf-8")

    result = search(root_dir=tmp_path, pattern="prefix")

    assert len(result["results"]) == 1
    assert len(result["results"][0]["line"]) <= MAX_LINE_CHARS


def test_search_skips_forbidden_subtrees(tmp_path: Path) -> None:
    (tmp_path / "budget").mkdir()
    (tmp_path / "budget" / "secret.txt").write_text("password=hunter2\n", encoding="utf-8")
    (tmp_path / "src.txt").write_text("password=public\n", encoding="utf-8")

    result = search(root_dir=tmp_path, pattern="password")

    paths = {r["path"] for r in result["results"]}
    assert "src.txt" in paths
    assert all("budget" not in p.split("/") for p in paths)


def test_search_rejects_bad_regex(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        search(root_dir=tmp_path, pattern="(unclosed")


def test_search_total_byte_cap(tmp_path: Path) -> None:
    # 1024 entries each ~80 chars should blow past 16 KiB before max_results.
    target = tmp_path / "fat.txt"
    line = "match " + ("y" * 70)
    target.write_text("\n".join(line for _ in range(1024)) + "\n", encoding="utf-8")

    result = search(root_dir=tmp_path, pattern="match", max_total_bytes=2048)

    assert result["truncated"] is True
    # Conservative: must not exceed the cap by more than one entry's worth.
    assert len(result["results"]) < 200

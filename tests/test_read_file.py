from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.read_file import MAX_BYTES, read_file


def test_read_file_returns_small_content(tmp_path: Path) -> None:
    file = tmp_path / "src" / "hello.py"
    file.parent.mkdir(parents=True)
    file.write_text("print('hi')\n", encoding="utf-8")

    result = read_file(root_dir=tmp_path, relative_path="src/hello.py")

    assert result["content"] == "print('hi')\n"
    assert result["size_bytes"] == len(b"print('hi')\n")
    assert result["truncated"] is False
    assert result["path"] == "src/hello.py"


def test_read_file_returns_content_with_truncation_marker(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    payload = b"A" * (32 * 1024)  # 32 KiB
    big.write_bytes(payload)

    result = read_file(root_dir=tmp_path, relative_path="big.txt")

    assert result["truncated"] is True
    assert result["size_bytes"] == 32 * 1024
    assert result["content"].startswith("A" * MAX_BYTES)
    assert "[truncated: total=32768 bytes]" in result["content"]
    # The body up to the cap is exactly MAX_BYTES; the marker is the only
    # text appended past the cap.
    head, _, _tail = result["content"].partition("\n[truncated:")
    assert len(head) == MAX_BYTES


def test_read_file_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_file(root_dir=tmp_path, relative_path="missing.txt")


def test_read_file_raises_on_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with pytest.raises(IsADirectoryError):
        read_file(root_dir=tmp_path, relative_path="sub")

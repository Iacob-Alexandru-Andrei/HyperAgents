"""Read-only ``read_file`` tool for the meta-agent.

Returns the contents of ``workspace_root / relative_path`` after path-safety
checks, capped at 16 KiB. When the file is larger than the cap, only the
first ``MAX_BYTES + 1`` bytes are streamed off disk (not the whole file)
and the first ``MAX_BYTES`` are returned along with a
``[truncated: total=<N> bytes]`` marker and ``truncated=True``.

Bounds enforcement: the workspace root is injected at load time by
``hyperagents/agent/tools/__init__.py:load_tools`` via ``functools.partial``,
so this tool's ``tool_function`` always receives the validated workspace
root. Path-safety violations raise ``ValueError`` so ``process_tool_call``
folds the error into the meta-agent's chat as a recoverable tool-output
message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools._path_safety import resolve_safe_path

MAX_BYTES = 16 * 1024  # 16 KiB


def tool_info(*, workspace_root=None):
    return {
        "name": "read_file",
        "description": (
            "Read a UTF-8 file inside the meta-agent's workspace. Returns the "
            "first 16 KiB of content plus size_bytes / truncated flags. Paths "
            "must be relative to the workspace root; absolute paths, '..' "
            "escapes, and forbidden subtrees (budget internals, held-out eval, "
            "hard pools, sibling generations, '.git') are refused."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to the workspace root.",
                },
            },
            "required": ["relative_path"],
        },
    }


def _read_file_impl(
    *,
    root_dir: Path,
    relative_path: str,
    current_gen: str | None = None,
) -> dict[str, Any]:
    """Return contents of ``root_dir / relative_path``.

    Returns a dict with keys ``path``, ``size_bytes``, ``content``,
    ``truncated``. Raises ``ValueError`` (with the word ``"forbidden"``) on
    path-safety violation; ``FileNotFoundError`` when the file does not
    exist; ``IsADirectoryError`` when the target is a directory.

    Bounded read: only ``MAX_BYTES + 1`` bytes are read from disk, so a
    multi-gigabyte file does not load fully into memory.
    """
    target = resolve_safe_path(
        root_dir=root_dir,
        relative_path=relative_path,
        current_gen=current_gen,
    )

    if not target.exists():
        raise FileNotFoundError(f"no such file: {relative_path!r}")
    if target.is_dir():
        raise IsADirectoryError(f"path is a directory, not a file: {relative_path!r}")

    size_bytes = target.stat().st_size
    with target.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    truncated = len(data) > MAX_BYTES
    if truncated:
        data = data[:MAX_BYTES]

    if truncated:
        body = data.decode("utf-8", errors="replace")
        content = f"{body}\n[truncated: total={size_bytes} bytes]"
    else:
        content = data.decode("utf-8", errors="replace")

    return {
        "path": str(target.relative_to(root_dir.resolve())),
        "size_bytes": size_bytes,
        "content": content,
        "truncated": truncated,
    }


def read_file(
    *,
    root_dir: Path,
    relative_path: str,
    current_gen: str | None = None,
) -> dict[str, Any]:
    """Public, dependency-injected entry point used by tests and callers
    that already hold the workspace root.

    Hyperagents' ``load_tools`` discovers ``tool_function`` (below) and
    binds ``workspace_root`` via ``functools.partial``; this function is
    the underlying implementation and is called directly by both
    ``tool_function`` and the test suite.
    """
    return _read_file_impl(
        root_dir=root_dir,
        relative_path=relative_path,
        current_gen=current_gen,
    )


def tool_function(relative_path, current_gen=None, *, workspace_root=None):
    if workspace_root is None:
        raise ValueError("read_file is disabled: no workspace_root configured.")
    return _read_file_impl(
        root_dir=Path(workspace_root),
        relative_path=relative_path,
        current_gen=current_gen,
    )

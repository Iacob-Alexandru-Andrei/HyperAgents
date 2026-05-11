"""Read-only ``search`` tool for the meta-agent.

Performs a regex search across all files under ``workspace_root`` (excluding
forbidden subtrees). The pattern is taken as a regular expression; the
caller is responsible for ``re.escape`` if literal matching is desired.

Output is capped three ways:

* at most 200 matching lines (``max_results``);
* each matching line truncated to 80 characters;
* total textual payload capped at 16 KiB (``max_total_bytes``).

Bounds enforcement: the workspace root is injected at load time by
``hyperagents/agent/tools/__init__.py:load_tools`` via ``functools.partial``.
Path-safety violations raise ``ValueError`` so ``process_tool_call`` folds
the error into the meta-agent's chat as a recoverable tool-output
message.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from agent.tools._path_safety import is_forbidden_component, resolve_safe_path

DEFAULT_MAX_RESULTS = 200
MAX_LINE_CHARS = 80
MAX_TOTAL_BYTES = 16 * 1024  # 16 KiB

_MAX_SCAN_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB


def tool_info(*, workspace_root=None):
    return {
        "name": "search",
        "description": (
            "Regex-search files under a workspace-relative path. Returns at "
            "most 200 matching lines, each clipped to 80 characters, total "
            "payload capped at 16 KiB. Forbidden subtrees are skipped. Invalid "
            "regex raises ValueError."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression to match.",
                },
                "relative_path": {
                    "type": "string",
                    "description": (
                        "Directory or file relative to the workspace root. "
                        "Empty string searches the whole workspace."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Cap on matches returned (default 200).",
                },
                "max_total_bytes": {
                    "type": "integer",
                    "description": "Cap on total textual payload (default 16 KiB).",
                },
            },
            "required": ["pattern"],
        },
    }


def _search_impl(
    *,
    root_dir: Path,
    pattern: str,
    relative_path: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    current_gen: str | None = None,
) -> dict[str, Any]:
    """Search files under ``root_dir / relative_path`` for ``pattern``.

    Returns a dict with keys ``pattern``, ``results``, ``truncated``,
    ``max_results``. Each result is a dict with ``path`` (relative to
    ``root_dir``), ``line_number`` (1-indexed), and ``line`` (truncated to
    ``MAX_LINE_CHARS``).

    Raises ``ValueError`` (containing ``"forbidden"``) on path-safety
    violations or on an invalid regex; ``FileNotFoundError`` when the
    target does not exist.
    """
    max_results = max(1, min(int(max_results), DEFAULT_MAX_RESULTS))
    max_total_bytes = max(1, min(int(max_total_bytes), MAX_TOTAL_BYTES))

    target = resolve_safe_path(
        root_dir=root_dir,
        relative_path=relative_path,
        current_gen=current_gen,
    )

    if not target.exists():
        raise FileNotFoundError(f"no such path: {relative_path!r}")

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"forbidden: invalid regex pattern: {exc}") from exc

    root_resolved = root_dir.resolve()
    results: list[dict[str, Any]] = []
    total_bytes = 0
    truncated = False

    iter_target = target if target.is_dir() else target.parent

    def _scan_file(path: Path) -> bool:
        """Scan one file; return True if caller should stop walking.

        Re-resolves the walked path before opening to defeat symlink-file
        escapes: ``os.walk(..., followlinks=False)`` does not follow
        symlink directories but DOES list symlink files. A symlink under
        ``root_dir`` whose target is ``/etc/passwd`` would otherwise be
        opened and its contents leaked. Resolving + containment check
        below rejects such escapes.
        """
        nonlocal total_bytes, truncated
        try:
            resolved = path.resolve()
        except OSError:
            return False
        if not (resolved == root_resolved or resolved.is_relative_to(root_resolved)):
            return False
        try:
            if path.stat().st_size > _MAX_SCAN_FILE_BYTES:
                return False
        except OSError:
            return False

        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.rstrip("\n")
                    if not regex.search(line):
                        continue
                    snippet = line[:MAX_LINE_CHARS]
                    result_path = str(path.relative_to(root_resolved))
                    entry = {
                        "path": result_path,
                        "line_number": line_number,
                        "line": snippet,
                    }
                    entry_bytes = len(result_path) + len(snippet) + 16
                    if total_bytes + entry_bytes > max_total_bytes:
                        truncated = True
                        return True
                    results.append(entry)
                    total_bytes += entry_bytes
                    if len(results) >= max_results:
                        truncated = True
                        return True
        except OSError:
            return False
        return False

    if target.is_file():
        _scan_file(target)
    else:
        for dirpath, dirnames, filenames in os.walk(iter_target, followlinks=False):
            parent_parts = Path(dirpath).relative_to(root_resolved).parts
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not is_forbidden_component(d, current_gen, parent_parts)
            )
            for fname in sorted(filenames):
                if is_forbidden_component(fname, current_gen):
                    continue
                file_path = Path(dirpath) / fname
                if _scan_file(file_path):
                    break
            if truncated:
                break

    return {
        "pattern": pattern,
        "results": results,
        "truncated": truncated,
        "max_results": max_results,
    }


def search(
    *,
    root_dir: Path,
    pattern: str,
    relative_path: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    current_gen: str | None = None,
) -> dict[str, Any]:
    """Public, dependency-injected entry point used by tests and callers
    that already hold the workspace root.

    Hyperagents' ``load_tools`` discovers ``tool_function`` (below) and
    binds ``workspace_root`` via ``functools.partial``; this function is
    the underlying implementation and is called directly by both
    ``tool_function`` and the test suite.
    """
    return _search_impl(
        root_dir=root_dir,
        pattern=pattern,
        relative_path=relative_path,
        max_results=max_results,
        max_total_bytes=max_total_bytes,
        current_gen=current_gen,
    )


def tool_function(
    pattern,
    relative_path="",
    max_results=DEFAULT_MAX_RESULTS,
    max_total_bytes=MAX_TOTAL_BYTES,
    current_gen=None,
    *,
    workspace_root=None,
):
    if workspace_root is None:
        raise ValueError("search is disabled: no workspace_root configured.")
    return _search_impl(
        root_dir=Path(workspace_root),
        pattern=pattern,
        relative_path=relative_path,
        max_results=max_results,
        max_total_bytes=max_total_bytes,
        current_gen=current_gen,
    )

"""Read-only ``file_tree`` tool for the meta-agent.

Walks ``workspace_root / relative_path`` and returns a flat list of entries
relative to the workspace root. Entries inside forbidden subtrees are
pruned. The walk stops once ``max_entries`` is reached or ``max_depth``
levels are exceeded.

Bounds enforcement: the workspace root is injected at load time by
``hyperagents/agent/tools/__init__.py:load_tools`` via ``functools.partial``.
Path-safety violations raise ``ValueError`` so ``process_tool_call`` folds
the error into the meta-agent's chat as a recoverable tool-output
message.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent.tools._path_safety import is_forbidden_component, resolve_safe_path

DEFAULT_MAX_ENTRIES = 500
DEFAULT_MAX_DEPTH = 6


def tool_info(*, workspace_root=None):
    return {
        "name": "file_tree",
        "description": (
            "List files and directories under a workspace-relative path. "
            "Returns up to 500 entries (configurable downward) up to 6 levels "
            "deep. Forbidden subtrees (budget internals, held-out eval, hard "
            "pools, sibling generations, '.git') are pruned silently; "
            "path-escapes raise."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": (
                        "Directory relative to the workspace root. Empty "
                        "string lists the workspace root itself."
                    ),
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Cap on entries returned (default 500).",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Cap on tree depth (default 6).",
                },
            },
            "required": [],
        },
    }


def _file_tree_impl(
    *,
    root_dir: Path,
    relative_path: str = "",
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    current_gen: str | None = None,
) -> dict[str, Any]:
    """Return a flat tree listing rooted at ``root_dir / relative_path``.

    Returns a dict with keys ``path``, ``entries``, ``truncated``,
    ``max_entries``, ``max_depth``. Each entry is a dict with ``path``
    (relative to ``root_dir``), ``kind`` (``"file"`` or ``"dir"``), and
    ``depth`` (1-indexed, relative to the listed root).

    Raises ``ValueError`` (containing ``"forbidden"``) on path-safety
    violations; ``FileNotFoundError`` when the target does not exist;
    ``NotADirectoryError`` when the target is not a directory.
    """
    max_entries = max(1, min(int(max_entries), DEFAULT_MAX_ENTRIES))
    max_depth = max(1, min(int(max_depth), DEFAULT_MAX_DEPTH))

    target = resolve_safe_path(
        root_dir=root_dir,
        relative_path=relative_path,
        current_gen=current_gen,
    )

    if not target.exists():
        raise FileNotFoundError(f"no such path: {relative_path!r}")
    if not target.is_dir():
        raise NotADirectoryError(f"path is not a directory: {relative_path!r}")

    root_resolved = root_dir.resolve()
    entries: list[dict[str, Any]] = []
    truncated = False

    base_depth = len(target.relative_to(root_resolved).parts)

    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        depth_from_target = len(Path(dirpath).relative_to(target).parts)

        if depth_from_target >= max_depth:
            dirnames[:] = []

        parent_parts = Path(dirpath).relative_to(root_resolved).parts
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not is_forbidden_component(d, current_gen, parent_parts)
        )
        filenames = sorted(filenames)

        for d in dirnames:
            entry_path = (Path(dirpath) / d).relative_to(root_resolved)
            entries.append(
                {
                    "path": str(entry_path),
                    "kind": "dir",
                    "depth": depth_from_target + 1,
                }
            )
            if len(entries) >= max_entries:
                truncated = True
                break

        if truncated:
            break

        for f in filenames:
            if is_forbidden_component(f, current_gen):
                continue
            entry_path = (Path(dirpath) / f).relative_to(root_resolved)
            entries.append(
                {
                    "path": str(entry_path),
                    "kind": "file",
                    "depth": depth_from_target + 1,
                }
            )
            if len(entries) >= max_entries:
                truncated = True
                break

        if truncated:
            break

    return {
        "path": str(target.relative_to(root_resolved)) or ".",
        "entries": entries,
        "truncated": truncated,
        "max_entries": max_entries,
        "max_depth": max_depth,
        "base_depth": base_depth,
    }


def file_tree(
    *,
    root_dir: Path,
    relative_path: str = "",
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    current_gen: str | None = None,
) -> dict[str, Any]:
    """Public, dependency-injected entry point used by tests and callers
    that already hold the workspace root.

    Hyperagents' ``load_tools`` discovers ``tool_function`` (below) and
    binds ``workspace_root`` via ``functools.partial``; this function is
    the underlying implementation and is called directly by both
    ``tool_function`` and the test suite.
    """
    return _file_tree_impl(
        root_dir=root_dir,
        relative_path=relative_path,
        max_entries=max_entries,
        max_depth=max_depth,
        current_gen=current_gen,
    )


def tool_function(
    relative_path="",
    max_entries=DEFAULT_MAX_ENTRIES,
    max_depth=DEFAULT_MAX_DEPTH,
    current_gen=None,
    *,
    workspace_root=None,
):
    if workspace_root is None:
        raise ValueError("file_tree is disabled: no workspace_root configured.")
    return _file_tree_impl(
        root_dir=Path(workspace_root),
        relative_path=relative_path,
        max_entries=max_entries,
        max_depth=max_depth,
        current_gen=current_gen,
    )

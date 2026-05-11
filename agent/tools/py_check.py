"""Read-only ``py_check`` tool for the meta-agent.

Compiles a single Python file in-process via the stdlib ``compile``
builtin (the same logic ``python -m py_compile`` uses to drive parsing).
The source is NEVER executed: ``compile`` parses to bytecode but does not
run it. The result is a short syntax-check verdict.

Bounds enforcement: the workspace root is injected at load time by
``hyperagents/agent/tools/__init__.py:load_tools`` via ``functools.partial``.
Path-safety violations raise ``ValueError`` so ``process_tool_call`` folds
the error into the meta-agent's chat as a recoverable tool-output
message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools._path_safety import resolve_safe_path

_MAX_SOURCE_BYTES = 4 * 1024 * 1024  # 4 MiB
_MAX_ERROR_CHARS = 1024


def tool_info(*, workspace_root=None):
    return {
        "name": "py_check",
        "description": (
            "Syntax-check a Python file by parsing it with compile() -- the "
            "source is NEVER executed. Returns ok=True for clean parses and "
            "ok=False with a short error message otherwise. Only .py files; "
            "anything else returns ok=False."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path to a .py file relative to the workspace root.",
                },
            },
            "required": ["relative_path"],
        },
    }


def _py_check_impl(
    *,
    root_dir: Path,
    relative_path: str,
    current_gen: str | None = None,
) -> dict[str, Any]:
    """Syntax-check the Python file at ``root_dir / relative_path``.

    Returns a dict with keys ``path``, ``ok``, ``message``. ``ok`` is
    ``True`` when the file parses cleanly; otherwise ``ok=False`` and
    ``message`` carries a short error description (truncated to
    ``_MAX_ERROR_CHARS``).

    Raises ``ValueError`` (containing ``"forbidden"``) on path-safety
    violations; ``FileNotFoundError`` when the file does not exist;
    ``IsADirectoryError`` when the target is a directory.
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
    if target.suffix != ".py":
        return {
            "path": str(target.relative_to(root_dir.resolve())),
            "ok": False,
            "message": f"not a Python file: {target.suffix!r}",
        }

    try:
        size = target.stat().st_size
    except OSError as exc:
        return {
            "path": str(target.relative_to(root_dir.resolve())),
            "ok": False,
            "message": f"stat failed: {exc}"[:_MAX_ERROR_CHARS],
        }

    if size > _MAX_SOURCE_BYTES:
        return {
            "path": str(target.relative_to(root_dir.resolve())),
            "ok": False,
            "message": f"source too large for syntax check ({size} bytes)",
        }

    try:
        source = target.read_bytes()
    except OSError as exc:
        return {
            "path": str(target.relative_to(root_dir.resolve())),
            "ok": False,
            "message": f"read failed: {exc}"[:_MAX_ERROR_CHARS],
        }

    try:
        compile(source, str(target), "exec", dont_inherit=True)
    except SyntaxError as exc:
        message = f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})"
        return {
            "path": str(target.relative_to(root_dir.resolve())),
            "ok": False,
            "message": message[:_MAX_ERROR_CHARS],
        }
    except ValueError as exc:
        return {
            "path": str(target.relative_to(root_dir.resolve())),
            "ok": False,
            "message": f"ValueError: {exc}"[:_MAX_ERROR_CHARS],
        }

    return {
        "path": str(target.relative_to(root_dir.resolve())),
        "ok": True,
        "message": "ok",
    }


def py_check(
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
    return _py_check_impl(
        root_dir=root_dir,
        relative_path=relative_path,
        current_gen=current_gen,
    )


def tool_function(relative_path, current_gen=None, *, workspace_root=None):
    if workspace_root is None:
        raise ValueError("py_check is disabled: no workspace_root configured.")
    return _py_check_impl(
        root_dir=Path(workspace_root),
        relative_path=relative_path,
        current_gen=current_gen,
    )

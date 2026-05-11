"""Shared path-safety helpers for read-only meta-agent tools.

Every public tool that consumes a workspace-relative path routes the
caller-supplied path through ``resolve_safe_path`` before touching the
filesystem. Two protections are applied:

* Path containment: the resolved target must stay under ``root_dir``.
* Forbidden subpaths: certain repo-internal directories (budget internals,
  held-out eval outputs, hard pools, evaluator telemetry, sibling gen
  outputs, atomic-publication staging dirs, other workers' scratch dirs,
  ``.git/``) are never readable through these tools.

The forbidden list is centralized so every tool refuses the same set.

Module name has no leading underscore so it is discovered as a tool
module by ``hyperagents/agent/tools/__init__.py``'s file glob — however,
this module exports no ``tool_info`` / ``tool_function`` symbols, so
``load_tools`` raises if asked to load it by name. It is imported by the
sibling tool modules (``read_file``, ``file_tree``, ``search``,
``py_check``) for the shared safety helpers only.
"""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_COMPONENTS: frozenset[str] = frozenset(
    {
        "budget",
        "_eval_val",
        "_eval_test",
        "paper_review_hard",
        "epoch_data",
        "_pending",
        "_rs_scratch",
        ".git",
    }
)

_GEN_DIR_PATTERN = re.compile(r"^gen_[A-Za-z0-9_-]+$")


def _lineage_gen_allowed(parts: tuple[str, ...], index: int) -> bool:
    return index > 0 and parts[index - 1] == "lineage"


def _check_forbidden_components(parts: tuple[str, ...], current_gen: str | None) -> str | None:
    """Return the first forbidden component encountered, or ``None``.

    Domain-prefixed held-out dirs like ``paper_review_eval_val`` and
    ``paper_review_eval_test`` are real production paths under each
    ``gen_<id>/``; the bare ``_eval_val`` / ``_eval_test`` entries in
    ``FORBIDDEN_COMPONENTS`` only match exact-component names. The suffix
    check below catches the domain-prefixed variants so a meta-agent with
    ``current_gen`` set cannot read held-out eval outputs of its own gen
    or any sibling.
    """
    for index, part in enumerate(parts):
        if part in FORBIDDEN_COMPONENTS:
            return part
        if (
            _GEN_DIR_PATTERN.match(part)
            and not _lineage_gen_allowed(parts, index)
            and (current_gen is None or part != current_gen)
        ):
            return part
        if part.endswith(("_eval_val", "_eval_test")):
            return part
    return None


def resolve_safe_path(
    *,
    root_dir: Path,
    relative_path: str,
    current_gen: str | None = None,
) -> Path:
    """Resolve ``root_dir / relative_path`` after checking safety.

    Returns the resolved absolute path. Raises ``ValueError`` with the word
    "forbidden" in the message when the path violates either containment or
    the forbidden-component list.
    """
    if not isinstance(relative_path, str):
        raise ValueError(f"forbidden: relative_path must be a string, got {type(relative_path)!r}")

    root_resolved = root_dir.resolve()
    candidate = (root_resolved / relative_path).resolve() if relative_path else root_resolved

    if not (candidate == root_resolved or candidate.is_relative_to(root_resolved)):
        raise ValueError(f"forbidden: path {relative_path!r} resolves outside the workspace root")

    try:
        rel_parts = candidate.relative_to(root_resolved).parts
    except ValueError as exc:
        raise ValueError(f"forbidden: path {relative_path!r} is not under workspace root") from exc

    offender = _check_forbidden_components(rel_parts, current_gen)
    if offender is not None:
        raise ValueError(
            f"forbidden: path {relative_path!r} contains forbidden component {offender!r}"
        )

    return candidate


def is_forbidden_component(
    name: str,
    current_gen: str | None = None,
    parent_parts: tuple[str, ...] = (),
) -> bool:
    """Return ``True`` if a single path component is forbidden.

    Used by tree walking and search to prune subtrees without raising.
    Mirrors :func:`_check_forbidden_components` -- including the
    domain-prefixed ``_eval_val`` / ``_eval_test`` suffix check that
    catches held-out dirs like ``paper_review_eval_val``.
    """
    if name in FORBIDDEN_COMPONENTS:
        return True
    if _GEN_DIR_PATTERN.match(name):
        if parent_parts and parent_parts[-1] == "lineage":
            return False
        return current_gen is None or name != current_gen
    return name.endswith(("_eval_val", "_eval_test"))

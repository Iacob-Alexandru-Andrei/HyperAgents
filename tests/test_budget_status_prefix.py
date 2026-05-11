from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _load_llm_withtools_module():
    """Import ``agent.llm_withtools`` with the LLM/tool deps stubbed.

    The module imports ``agent.llm.get_response_from_llm`` and
    ``agent.tools.load_tools`` at top level; both are unrelated to the
    ``_budget_status_prefix`` helper under test. We install lightweight
    fakes so the import succeeds without pulling the full provider stack.
    """
    sys.modules.pop("agent.llm_withtools", None)
    if "agent.llm" not in sys.modules:
        agent_llm = types.ModuleType("agent.llm")
        agent_llm.get_response_from_llm = lambda **_kwargs: ("", [], {})  # type: ignore[attr-defined]
        sys.modules["agent.llm"] = agent_llm
    if "agent.tools" not in sys.modules:
        agent_tools = types.ModuleType("agent.tools")
        agent_tools.load_tools = lambda **_kwargs: []  # type: ignore[attr-defined]
        sys.modules["agent.tools"] = agent_tools

    import agent.llm_withtools as module  # noqa: PLC0415

    return module


def test_budget_status_prefix_returns_empty_when_path_falsy() -> None:
    module = _load_llm_withtools_module()

    assert module._budget_status_prefix(None) == ""
    assert module._budget_status_prefix("") == ""


def test_budget_status_prefix_raises_when_provided_path_missing(tmp_path: Path) -> None:
    module = _load_llm_withtools_module()

    with pytest.raises(FileNotFoundError):
        module._budget_status_prefix(str(tmp_path / "missing.md"))


def test_budget_status_prefix_returns_empty_for_empty_file(tmp_path: Path) -> None:
    module = _load_llm_withtools_module()
    status_path = tmp_path / "status.md"
    status_path.write_text("", encoding="utf-8")

    assert module._budget_status_prefix(str(status_path)) == ""


def test_budget_status_prefix_returns_empty_for_whitespace_only_file(tmp_path: Path) -> None:
    module = _load_llm_withtools_module()
    status_path = tmp_path / "status.md"
    status_path.write_text("   \n\t\n", encoding="utf-8")

    assert module._budget_status_prefix(str(status_path)) == ""


def test_budget_status_prefix_returns_text_plus_blank_line(tmp_path: Path) -> None:
    module = _load_llm_withtools_module()
    status_path = tmp_path / "status.md"
    status_path.write_text("Budget lease: expand-gen-3.\n", encoding="utf-8")

    prefix = module._budget_status_prefix(str(status_path))

    assert prefix == "Budget lease: expand-gen-3.\n\n"

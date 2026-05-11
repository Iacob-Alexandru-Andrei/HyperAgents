from __future__ import annotations

from pathlib import Path

from meta_agent import MetaAgent


def test_meta_agent_passes_workspace_root_to_chat_tools(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_chat_with_agent(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("meta_agent.chat_with_agent", fake_chat_with_agent)
    chat_history = tmp_path / "chat.md"
    agent = MetaAgent(
        model="fake",
        chat_history_file=str(chat_history),
        budget_status_path="/rqgm_budget/status.md",
    )

    agent.forward(
        repo_path="/rqgm/",
        eval_path=str(tmp_path),
        iterations_left=3,
    )

    assert captured["workspace_root"] == "/rqgm/"
    assert captured["budget_status_path"] == "/rqgm_budget/status.md"

from __future__ import annotations

from pathlib import Path

import pytest

import generation_step


class _ExecResult:
    exit_code = 0


class _FakeContainer:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def start(self) -> None:
        pass

    def exec_run(self, cmd, workdir=None):
        if isinstance(cmd, list):
            self.commands.append(cmd)
        return _ExecResult()


def _install_generation_fakes(
    monkeypatch: pytest.MonkeyPatch,
    patch_text: str,
) -> tuple[_FakeContainer, list[str]]:
    container = _FakeContainer()
    eval_calls: list[str] = []

    monkeypatch.setattr(generation_step, "build_container", lambda *args, **kwargs: container)
    monkeypatch.setattr(generation_step, "cleanup_container", lambda container: None)
    monkeypatch.setattr(generation_step, "setup_logger", lambda path: None)
    monkeypatch.setattr(generation_step, "safe_log", lambda message: None)
    monkeypatch.setattr(generation_step, "log_container_output", lambda result: None)
    monkeypatch.setattr(generation_step, "apply_diffs_container", lambda container, patches: "base")
    monkeypatch.setattr(
        generation_step,
        "copy_prev_eval_to_container",
        lambda *args, **kwargs: "/tmp/prev_eval",
    )
    monkeypatch.setattr(generation_step, "run_commands_to_check_compilation", lambda container: None)
    monkeypatch.setattr(generation_step, "get_score", lambda *args, **kwargs: None)

    def fake_copy_from_container(container, source_path, dest_path):
        agent_output = Path(dest_path)
        agent_output.mkdir(parents=True, exist_ok=True)
        (agent_output / "model_patch.diff").write_text(patch_text, encoding="utf-8")

    def fake_eval_produced_agent(*args, **kwargs):
        eval_calls.append(kwargs["domain"])

    monkeypatch.setattr(generation_step, "copy_from_container", fake_copy_from_container)
    monkeypatch.setattr(generation_step, "eval_produced_agent", fake_eval_produced_agent)
    return container, eval_calls


def _meta_agent_command(container: _FakeContainer) -> list[str]:
    return next(command for command in container.commands if "run_meta_agent.py" in command)


@pytest.mark.parametrize(("iterations_left", "expected"), [(7, "7"), (-3, "0")])
def test_generation_step_forwards_clamped_iterations_left(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    iterations_left: int,
    expected: str,
) -> None:
    container, _ = _install_generation_fakes(
        monkeypatch,
        "diff --git a/task_agent.py b/task_agent.py\n",
    )

    generation_step.run_generation_step(
        docker_client=object(),
        domains=["paper_review"],
        output_dir=str(tmp_path),
        run_id="unit",
        current_genid=1,
        parent_genid="initial",
        root_dir=str(tmp_path / "root"),
        root_commit="root",
        eval_samples=[1],
        eval_workers=1,
        eval_subsets=[""],
        parent_patch_files=[],
        run_eval_after_meta_agent=False,
        skip_staged_eval=True,
        iterations_left=iterations_left,
        model="fake",
    )

    command = _meta_agent_command(container)
    assert command[command.index("--iterations_left") + 1] == expected


def test_generation_step_can_skip_eval_after_meta_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, eval_calls = _install_generation_fakes(
        monkeypatch,
        "diff --git a/task_agent.py b/task_agent.py\n",
    )

    metadata = generation_step.run_generation_step(
        docker_client=object(),
        domains=["paper_review"],
        output_dir=str(tmp_path),
        run_id="unit",
        current_genid=1,
        parent_genid="initial",
        root_dir=str(tmp_path / "root"),
        root_commit="root",
        eval_samples=[1],
        eval_workers=1,
        eval_subsets=[""],
        parent_patch_files=[],
        run_eval_after_meta_agent=False,
        skip_staged_eval=True,
        iterations_left=1,
        model="fake",
    )

    assert metadata["parent_agent_success"] is True
    assert metadata["run_eval"] is False
    assert eval_calls == []

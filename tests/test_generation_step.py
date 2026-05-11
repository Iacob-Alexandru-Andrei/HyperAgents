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
    monkeypatch.setattr(generation_step, "safe_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(generation_step, "log_container_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(generation_step, "apply_diffs_container", lambda container, patches: "base")
    monkeypatch.setattr(generation_step, "run_commands_to_check_compilation", lambda container: None)
    monkeypatch.setattr(generation_step, "get_score", lambda *args, **kwargs: None)

    def fake_copy_from_container(container, source_path, dest_path):
        if str(dest_path).endswith("/"):
            agent_output = Path(dest_path)
            agent_output.mkdir(parents=True, exist_ok=True)
            (agent_output / "model_patch.diff").write_text(patch_text, encoding="utf-8")
        else:
            target = Path(dest_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")

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


# ----- F2l (recursive-scientist deviation) ----------------------------------


class _RichExecResult:
    """Exec result with a configurable exit_code + output, for F2l tests."""

    def __init__(self, exit_code: int = 0, output: bytes = b"") -> None:
        self.exit_code = exit_code
        self.output = output


class _RecordingContainer:
    """Container fake that records every (cmd, workdir) pair so tests can
    assert the F2l snapshot helper issued the expected shell sequence."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, str | None]] = []

    def start(self) -> None:
        pass

    def exec_run(self, cmd, workdir=None):
        self.calls.append((cmd, workdir))
        return _RichExecResult(exit_code=0, output=b"")


def test_f2l_snapshot_helper_issues_expected_shell_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_snapshot_train_lineage_in_container`` should run three exec_run
    commands inside ``/<REPO_NAME>``: a cp/mkdir block, ``git add -A``,
    then a ``git commit`` with deterministic author and message.
    """
    monkeypatch.setattr(generation_step, "log_container_output", lambda *a, **k: None)
    monkeypatch.setattr(generation_step, "safe_log", lambda *a, **k: None)

    container = _RecordingContainer()
    generation_step._snapshot_train_lineage_in_container(
        container,
        current_genid=42,
        base_commit="abcdef1",
        metadata={"current_genid": 42, "parent_genid": 7},
        verbose=False,
    )

    assert len(container.calls) == 3
    cp_cmd, cp_workdir = container.calls[0]
    add_cmd, add_workdir = container.calls[1]
    commit_cmd, commit_workdir = container.calls[2]

    # All three run inside /<REPO_NAME>.
    expected_workdir = f"/{generation_step.REPO_NAME}"
    assert cp_workdir == expected_workdir
    assert add_workdir == expected_workdir
    assert commit_workdir == expected_workdir

    # Step 1: mkdir + cp train eval dirs + cp agent_output + write metadata.json.
    cp_payload = cp_cmd[2]
    assert "mkdir -p" in cp_payload
    assert "lineage/gen_42" in cp_payload
    assert "/tmp/*_eval" in cp_payload
    assert "cp -r /tmp/agent_output" in cp_payload
    assert "metadata.json" in cp_payload
    # Stub payload contains the metadata we passed in.
    assert '"current_genid": 42' in cp_payload
    assert '"parent_genid": 7' in cp_payload

    # Step 2: stage everything.
    assert add_cmd == ["/bin/sh", "-c", "git add -A"]

    # Step 3: commit with deterministic author + message.
    commit_payload = commit_cmd[2]
    assert "user.name='rqgm'" in commit_payload
    assert "user.email='rqgm@local'" in commit_payload
    assert "--allow-empty" in commit_payload
    assert "code + train lineage for gen_42" in commit_payload


def test_f2l_snapshot_helper_raises_on_copy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-zero exit from the cp/mkdir step must surface as RuntimeError —
    the caller in run_generation_step catches & logs, but the helper
    itself stays strict so failures are detectable in unit tests.
    """
    monkeypatch.setattr(generation_step, "log_container_output", lambda *a, **k: None)
    monkeypatch.setattr(generation_step, "safe_log", lambda *a, **k: None)

    class _FailingContainer:
        def exec_run(self, cmd, workdir=None):
            return _RichExecResult(exit_code=1, output=b"cp: permission denied")

    with pytest.raises(RuntimeError, match="F2l lineage snapshot copy failed"):
        generation_step._snapshot_train_lineage_in_container(
            _FailingContainer(),
            current_genid=7,
            base_commit="base",
            verbose=False,
        )


def test_f2l_snapshot_helper_raises_on_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_step, "log_container_output", lambda *a, **k: None)
    monkeypatch.setattr(generation_step, "safe_log", lambda *a, **k: None)

    class _CommitFailingContainer:
        def __init__(self) -> None:
            self.calls = 0

        def exec_run(self, cmd, workdir=None):
            self.calls += 1
            # commit is the 3rd call; everything before succeeds.
            if self.calls >= 3:
                return _RichExecResult(exit_code=128, output=b"commit failed")
            return _RichExecResult(exit_code=0)

    with pytest.raises(RuntimeError, match="F2l lineage commit failed"):
        generation_step._snapshot_train_lineage_in_container(
            _CommitFailingContainer(),
            current_genid=7,
            base_commit="base",
            verbose=False,
        )


def test_run_generation_step_preserves_code_only_and_overwrites_model_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end: after the meta-agent + eval block, run_generation_step
    should (1) copy the original (pre-eval, code-only) ``model_patch.diff``
    to ``code_only_patch.diff`` and (2) overwrite ``model_patch.diff`` with
    the richer post-snapshot diff captured inside the container.
    """
    code_only_text = (
        "diff --git a/task_agent.py b/task_agent.py\n"
        "@@\n-old\n+new\n"
    )
    container, _ = _install_generation_fakes(monkeypatch, code_only_text)

    # Make the F2l snapshot helper a no-op (we test its internals
    # separately above); the call site is what we exercise here.
    monkeypatch.setattr(
        generation_step,
        "_snapshot_train_lineage_in_container",
        lambda *a, **k: "base",
    )

    # When the call site executes ``git diff --binary ...`` inside the
    # container, our fake_copy_from_container captures the destination
    # and we replace the file with a synthesized "richer" diff.
    richer_text = (
        code_only_text
        + "diff --git a/lineage/gen_1/paper_review_eval/predictions.csv "
          "b/lineage/gen_1/paper_review_eval/predictions.csv\n"
          "new file mode 100644\n"
    )

    original_copy_from = generation_step.copy_from_container
    captured_destinations: list[str] = []

    def tracking_copy_from(container, source_path, dest_path):
        captured_destinations.append(str(dest_path))
        # The first call (existing path, line ~548) writes the
        # code-only patch into agent_output/. Mirror the original fake.
        if dest_path.endswith("agent_output/"):
            original_copy_from(container, source_path, dest_path)
            return
        # The F2l call (after eval) overwrites model_patch.diff with the
        # richer diff. Simulate that.
        if source_path.endswith("model_patch.diff"):
            Path(dest_path).write_text(richer_text, encoding="utf-8")
            return
        original_copy_from(container, source_path, dest_path)

    monkeypatch.setattr(generation_step, "copy_from_container", tracking_copy_from)

    # Force the eval branch to run so the F2l block at the end fires.
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
        run_eval_after_meta_agent=True,
        skip_staged_eval=True,
        iterations_left=1,
        model="fake",
    )

    agent_output = tmp_path / "gen_1" / "agent_output"
    code_only_path = agent_output / "code_only_patch.diff"
    model_patch_path = agent_output / "model_patch.diff"

    assert code_only_path.exists(), "code_only_patch.diff must be preserved"
    assert code_only_path.read_text() == code_only_text

    assert model_patch_path.exists()
    assert model_patch_path.read_text() == richer_text
    assert "lineage/gen_1/paper_review_eval/predictions.csv" in model_patch_path.read_text()


def test_run_generation_step_f2l_failure_does_not_abort_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the F2l snapshot helper raises, the surrounding flow must still
    complete (the additive scaffolding cannot break production)."""
    container, _ = _install_generation_fakes(
        monkeypatch,
        "diff --git a/task_agent.py b/task_agent.py\n@@\n-x\n+y\n",
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated F2l failure")

    monkeypatch.setattr(generation_step, "_snapshot_train_lineage_in_container", _boom)

    # Capture safe_log calls so we can assert the failure was logged.
    logged: list[str] = []
    monkeypatch.setattr(
        generation_step,
        "safe_log",
        lambda message, *args, **kwargs: logged.append(str(message)),
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
        run_eval_after_meta_agent=True,
        skip_staged_eval=True,
        iterations_left=1,
        model="fake",
    )

    # Flow completed.
    assert metadata["parent_agent_success"] is True
    # F2l failure was logged.
    assert any("F2l lineage snapshot skipped" in msg for msg in logged), (
        f"expected F2l skip log, got: {logged}"
    )


# ----- F2l Phase 3 (no copy-tar fallback) -----------------------------------


def test_phase3_copy_prev_eval_to_container_is_gone() -> None:
    """The host-side tar of output_dir is deleted in Phase 3. The
    module must no longer expose ``copy_prev_eval_to_container`` or
    its helpers (``_lineage_gen_dirs``, ``_non_lineage_prune_cmds``,
    ``_read_parent_genid``)."""
    for name in (
        "copy_prev_eval_to_container",
        "_lineage_gen_dirs",
        "_non_lineage_prune_cmds",
        "_read_parent_genid",
    ):
        assert not hasattr(generation_step, name), (
            f"{name} should be deleted in F2l Phase 3 — found it still "
            f"exported from generation_step"
        )


def test_phase3_meta_agent_evals_folder_points_at_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The meta-agent command must pass
    ``--evals_folder /<REPO_NAME>/lineage`` -- the lineage tree
    delivered by patches is the SOLE source of ancestor artifacts."""
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
        iterations_left=1,
        model="fake",
    )

    meta_cmd = _meta_agent_command(container)
    evals_idx = meta_cmd.index("--evals_folder")
    assert meta_cmd[evals_idx + 1] == f"/{generation_step.REPO_NAME}/lineage"


def test_phase3_snapshot_passes_metadata_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end: the F2l snapshot helper receives the live metadata
    dict so the in-container metadata.json stub reflects what the host
    will eventually persist for this gen."""
    _install_generation_fakes(
        monkeypatch,
        "diff --git a/task_agent.py b/task_agent.py\n",
    )

    captured_kwargs: dict[str, object] = {}

    def _capturing_snapshot(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return kwargs.get("base_commit")

    monkeypatch.setattr(
        generation_step,
        "_snapshot_train_lineage_in_container",
        _capturing_snapshot,
    )

    generation_step.run_generation_step(
        docker_client=object(),
        domains=["paper_review"],
        output_dir=str(tmp_path),
        run_id="unit",
        current_genid=3,
        parent_genid=2,
        root_dir=str(tmp_path / "root"),
        root_commit="root",
        eval_samples=[1],
        eval_workers=1,
        eval_subsets=[""],
        parent_patch_files=[],
        run_eval_after_meta_agent=True,
        skip_staged_eval=True,
        iterations_left=1,
        model="fake",
    )

    assert "metadata" in captured_kwargs
    md = captured_kwargs["metadata"]
    assert isinstance(md, dict)
    assert md["current_genid"] == 3
    assert md["parent_genid"] == 2
    assert md["parent_agent_success"] is True
    assert md["run_eval"] is True


def test_phase3_llm_calls_extracted_from_tmp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The cost tracker writes ``/tmp/llm_calls.jsonl`` (outside the
    git repo) so the file does not get swept into the lineage commit.
    The host's copy_from_container call must use that path, not a path
    inside the lineage tree."""
    _install_generation_fakes(
        monkeypatch,
        "diff --git a/task_agent.py b/task_agent.py\n",
    )

    captured_sources: list[str] = []
    real_copy_from = generation_step.copy_from_container

    def tracking(container, source_path, dest_path):
        captured_sources.append(str(source_path))
        real_copy_from(container, source_path, dest_path)

    monkeypatch.setattr(generation_step, "copy_from_container", tracking)

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
        iterations_left=1,
        model="fake",
    )

    assert "/tmp/llm_calls.jsonl" in captured_sources, (
        f"expected /tmp/llm_calls.jsonl in extracted sources, got: {captured_sources}"
    )

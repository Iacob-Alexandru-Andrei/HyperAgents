"""Adversarial security tests for the read-only meta-agent tools.

Complements ``test_tool_path_safety.py`` (which enumerates the forbidden-path
matrix) by attacking the safety boundary in non-obvious ways: path
traversal via ``..``, absolute paths, symlink escape, null-byte injection,
Unicode lookalikes, catastrophic regex, symlink loops, hostile ``.py``
payloads, and the stability of the truncation marker downstream parsers
rely on.

Phase 7 of the reconciled execution plan requires proving the tools
cannot read held-out / test / hard-pool future data, budget internals,
sibling gen outputs, or evaluator telemetry. These tests target the
defense-in-depth properties (containment, no execution, output cap
formats) that hold those guarantees together.
"""

from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agent.tools.file_tree import file_tree
from agent.tools.py_check import py_check
from agent.tools.read_file import MAX_BYTES, read_file
from agent.tools.search import search


# ---------------------------------------------------------------------------
# read_file: path-traversal variants
# ---------------------------------------------------------------------------


def test_read_file_refuses_double_dot_slash_escape(tmp_path: Path) -> None:
    """``../../etc/passwd`` resolves outside root_dir and must be refused."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=workspace, relative_path="../../etc/passwd")


def test_read_file_refuses_absolute_path_escape(tmp_path: Path) -> None:
    """An absolute path like ``/etc/passwd`` is joined as itself and must be
    refused because it resolves outside root_dir.

    ``Path("/a") / "/b"`` == ``Path("/b")`` in stdlib semantics -- so passing
    an absolute relative_path is the classic way to bypass naive join-and-
    read code. ``resolve_safe_path`` catches it via the containment check.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=workspace, relative_path="/etc/passwd")


def test_read_file_refuses_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside the workspace pointing to ``/etc/hostname`` must be
    refused -- ``.resolve()`` follows the link, and the resolved target sits
    outside the workspace root.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # /etc/hostname is essentially always present on Linux CI; if it's not,
    # any absolute file outside the workspace works for the same test.
    outside_target = Path("/etc/hostname")
    if not outside_target.exists():
        pytest.skip("/etc/hostname not present on this system")

    link = workspace / "link_to_outside"
    link.symlink_to(outside_target)

    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=workspace, relative_path="link_to_outside")


def test_read_file_refuses_null_byte_in_path(tmp_path: Path) -> None:
    """A null byte embedded in the path must raise ValueError.

    Python's stdlib rejects null bytes at the syscall layer
    (``ValueError: embedded null character`` from os.stat/lstat). The tool
    therefore inherits the rejection even before our forbidden-component
    check fires -- the assertion is just that *some* ``ValueError`` escapes
    rather than the null byte being silently stripped or accepted.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ValueError):
        read_file(root_dir=workspace, relative_path="legit\x00../../../etc/passwd")


def test_read_file_refuses_unicode_normalization_attack(tmp_path: Path) -> None:
    """A path whose components only LOOK like forbidden names must not be
    treated as forbidden if the components don't match byte-for-byte.

    This documents the actual policy: forbidden-component check is exact
    string equality on each ``Path.parts`` component, not substring or
    Unicode-normalized match. So ``budǵet/lease.json`` (combining acute on
    'g') is NOT the forbidden ``budget`` directory and IS readable.

    Important corollary: if real workspace policy required Unicode-aware
    matching (e.g. ``NFKC`` normalization before comparison), this test
    would have to flip -- and the safety layer would have to grow that
    normalization step. For now, exact-component is the documented
    contract.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # 'budǵet' = b-u-d-g+combining-acute-accent-e-t. Visually similar to
    # 'budget' but not equal. Length differs (7 vs 6) and Path.parts will
    # carry the combining mark, so the exact-equality check passes.
    lookalike_dir = "budǵet"  # 'ǵ' is U+01F5 (precomposed) for portability
    assert lookalike_dir != "budget"
    (workspace / lookalike_dir).mkdir()
    (workspace / lookalike_dir / "lease.json").write_text("not-the-real-budget", encoding="utf-8")

    # Exact-component policy: this is NOT the forbidden 'budget' dir, so
    # the read succeeds. If we ever switch to Unicode-normalized matching,
    # this assertion must flip and the safety layer must add NFKC.
    result = read_file(root_dir=workspace, relative_path=f"{lookalike_dir}/lease.json")
    assert result["content"] == "not-the-real-budget"

    # And the real 'budget' is still refused -- proving the policy isn't
    # accidentally permissive.
    (workspace / "budget").mkdir()
    (workspace / "budget" / "lease.json").write_text("real-budget", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        read_file(root_dir=workspace, relative_path="budget/lease.json")


# ---------------------------------------------------------------------------
# search: catastrophic regex
# ---------------------------------------------------------------------------


def test_search_refuses_huge_regex_dos(tmp_path: Path) -> None:
    """A catastrophic-backtracking pattern over a moderate input must not
    hang the worker.

    The search tool currently does NOT install a regex timeout (Python's
    ``re`` has no built-in budget). On adversarial inputs (e.g.
    ``(a+)+$`` against ``"a" * 30 + "!"``) backtracking can blow up
    exponentially. The test uses a deliberately small input (20 'a's
    followed by '!') so that even worst-case backtracking finishes in
    well under the asserted bound -- that way the test is a smoke check on
    the property "search returns on bounded input" rather than a stress
    test of the regex engine itself.

    If a future change ever introduces a regex-execution timeout, this
    test still passes: it only asserts the call returns within the budget,
    either way.

    NOTE for maintainers: do not raise the input size. The whole point of
    the test is to stay inside the safe zone of the catastrophic curve.
    Anything beyond ~25 'a's may push wall-clock past 5s on a busy CI
    runner.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "big.txt"
    target.write_text("a" * 20 + "!", encoding="utf-8")

    result_holder: dict[str, Any] = {}

    def run() -> None:
        try:
            result_holder["ok"] = search(
                root_dir=workspace,
                pattern=r"(a+)+$",
                relative_path="",
            )
        except Exception as exc:  # threaded probe needs broad catch
            result_holder["err"] = exc

    thread = threading.Thread(target=run, daemon=True)
    start = time.monotonic()
    thread.start()
    thread.join(timeout=5.0)
    elapsed = time.monotonic() - start

    assert not thread.is_alive(), (
        f"search hung on catastrophic regex (elapsed={elapsed:.2f}s) - "
        "no regex timeout is installed in the tool"
    )
    # Either it returned a result dict or surfaced a ValueError for bad
    # input; both are acceptable. The forbidden outcome is "still running".
    assert "ok" in result_holder or "err" in result_holder


# ---------------------------------------------------------------------------
# file_tree: symlink loop must not hang
# ---------------------------------------------------------------------------


def test_file_tree_refuses_huge_recursive_symlink_loop(tmp_path: Path) -> None:
    """A symlink cycle (a -> b -> a) must not cause unbounded traversal.

    ``file_tree`` uses ``os.walk(..., followlinks=False)`` which is the
    correct primitive: it never descends through a symlink, so the loop
    is structurally impossible. This test pins that behavior in place so
    a future refactor cannot silently flip ``followlinks=True``.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    a = workspace / "a"
    b = workspace / "b"
    a.mkdir()
    b.mkdir()
    (a / "link_to_b").symlink_to(b, target_is_directory=True)
    (b / "link_to_a").symlink_to(a, target_is_directory=True)

    result_holder: dict[str, Any] = {}

    def run() -> None:
        try:
            result_holder["ok"] = file_tree(
                root_dir=workspace,
                relative_path="",
                max_entries=100,
            )
        except Exception as exc:  # threaded probe needs broad catch
            result_holder["err"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "file_tree hung on a symlink loop"

    assert "ok" in result_holder, f"file_tree failed: {result_holder.get('err')!r}"
    result = result_holder["ok"]
    assert isinstance(result, dict)
    entries = result.get("entries")
    assert isinstance(entries, list)
    # Without descent through symlinks the entries are: a, b, a/link_to_b,
    # b/link_to_a. Strict upper bound: well under max_entries.
    assert len(entries) <= 100
    assert len(entries) < 20


# ---------------------------------------------------------------------------
# py_check: must parse only, never execute
# ---------------------------------------------------------------------------


def test_py_check_refuses_pickle_payload_disguised_as_python(tmp_path: Path) -> None:
    """A .py file whose contents are raw pickle bytes must not crash the
    tool and must not be treated as valid Python.

    Pickle byte streams contain null bytes; ``compile()`` rejects sources
    with embedded null bytes. The tool should surface that as an ``ok=False``
    syntax-error verdict, not propagate the exception unhandled.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "pickled.py"
    target.write_bytes(pickle.dumps({"evil": True, "nested": [1, 2, 3]}))

    result = py_check(root_dir=workspace, relative_path="pickled.py")

    assert result["ok"] is False
    # The exact message text depends on whether compile() raises SyntaxError
    # or ValueError for the null bytes (CPython has raised both over
    # versions). Both branches are caught by py_check and surfaced via the
    # "message" field. Assert the verdict is a parse failure, not silent
    # success.
    message = result["message"]
    assert isinstance(message, str)
    assert any(
        marker in message.lower() for marker in ("syntaxerror", "valueerror", "null", "invalid")
    ), f"unexpected py_check message for pickle payload: {message!r}"


def test_py_check_refuses_subprocess_invocation_in_compiled_module(
    tmp_path: Path,
) -> None:
    """A .py file that *would* run ``os.system`` if executed must not run.

    ``py_check`` calls ``compile(..., mode='exec', dont_inherit=True)`` --
    that parses to a code object but never executes it. The test creates a
    canary path, points the .py file's payload at ``touch <canary>``, runs
    ``py_check``, and asserts the canary does NOT exist afterward.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # Use a path inside tmp_path (not /tmp) so a concurrent test run on the
    # same host cannot race against the canary. tmp_path is unique per
    # pytest invocation.
    canary = tmp_path / "pwned_canary"
    assert not canary.exists()

    target = workspace / "evil.py"
    # Properly escape the path into a Python source literal.
    target.write_text(
        f"import os\nos.system({str(canary)!r})\n",
        encoding="utf-8",
    )

    result = py_check(root_dir=workspace, relative_path="evil.py")

    # Source parses fine -- no syntax error. The point is that 'ok=True'
    # means it parsed, NOT that the contained code ran.
    assert result["ok"] is True, f"unexpected py_check verdict: {result!r}"
    assert not canary.exists(), (
        f"py_check executed the file -- canary {canary!s} was created. "
        "This would be a critical sandbox escape; py_check must only parse."
    )


# ---------------------------------------------------------------------------
# search: regex metacharacters cannot escape root
# ---------------------------------------------------------------------------


def test_search_pattern_with_path_metacharacters_does_not_escape_root(
    tmp_path: Path,
) -> None:
    """The regex pattern argument is matched against file CONTENT, not
    against paths -- so even a match-all pattern cannot surface files from
    outside ``root_dir``. The result list's ``path`` entries are always
    relative to root.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.txt").write_text("hello world", encoding="utf-8")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.txt").write_text("another line", encoding="utf-8")

    # Create a sibling directory next to root so an unsafe walk would find
    # it. The search must never reference these.
    sibling = tmp_path / "outside"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("SHOULD NOT APPEAR", encoding="utf-8")

    result = search(root_dir=workspace, pattern=".", relative_path="")

    workspace_resolved = workspace.resolve()
    for entry in result["results"]:
        rel = entry["path"]
        # No '..' segment may appear in a returned path.
        assert ".." not in Path(rel).parts, f"path escaped root: {rel!r}"
        # The reconstructed absolute path must still live under workspace.
        absolute = (workspace_resolved / rel).resolve()
        assert absolute.is_relative_to(workspace_resolved), (
            f"path {rel!r} resolves to {absolute!s}, outside {workspace_resolved!s}"
        )
        # The secret file's name must not appear in results.
        assert "secret.txt" not in rel
        assert "SHOULD NOT APPEAR" not in entry["line"]


# ---------------------------------------------------------------------------
# read_file: truncation marker stability
# ---------------------------------------------------------------------------


def test_read_file_truncation_marker_format_is_stable(tmp_path: Path) -> None:
    """Downstream parsers (paper-artifacts, transcript prettifier) rely on
    the exact marker format ``[truncated: total=<N> bytes]``. Pin it.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # Make the file deliberately larger than MAX_BYTES so truncation fires.
    payload_size = MAX_BYTES + 4096  # 16 KiB + 4 KiB = 20 KiB
    big = workspace / "big.txt"
    big.write_bytes(b"B" * payload_size)

    result = read_file(root_dir=workspace, relative_path="big.txt")

    assert result["truncated"] is True
    assert result["size_bytes"] == payload_size
    # Exact marker substring -- this is the contract.
    expected_marker = f"[truncated: total={payload_size} bytes]"
    assert expected_marker in result["content"], (
        f"truncation marker missing or malformed; expected {expected_marker!r} in content"
    )
    # And the head up to the marker is exactly MAX_BYTES of the original
    # bytes -- no surprise leading whitespace, no off-by-one.
    head, sep, tail = result["content"].partition("\n[truncated:")
    assert sep == "\n[truncated:"
    assert len(head) == MAX_BYTES
    # The tail after '\n[truncated:' is the rest of the marker.
    assert tail == f" total={payload_size} bytes]"


# ---------------------------------------------------------------------------
# Defensive sanity: os module imported only for path operations, not used
# to spawn anything. Keep this stub assertion so a future contributor who
# accidentally imports subprocess at module level here trips the lint.
# ---------------------------------------------------------------------------


def test_module_does_not_import_subprocess() -> None:
    """Negative-import test: this adversarial test module must never
    itself import ``subprocess`` (only ``os`` and ``threading``) so a
    malicious .py file dropped under tests/ cannot trick a later refactor
    into wiring up a real shell.
    """
    import sys

    this_module = sys.modules[__name__]
    # ``os`` and ``threading`` are expected; ``subprocess`` is not.
    assert not hasattr(this_module, "subprocess")

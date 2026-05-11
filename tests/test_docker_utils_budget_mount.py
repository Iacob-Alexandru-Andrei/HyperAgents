from __future__ import annotations

from pathlib import Path

import docker
import pytest

from utils import docker_utils
from utils.docker_utils import BUDGET_STATUS_CONTAINER_DIR, build_container
from utils.constants import REPO_NAME


class _Image:
    tags = ["app"]


class _Images:
    def list(self):
        return [_Image()]


class _Containers:
    def __init__(self) -> None:
        self.run_kwargs: dict[str, object] = {}

    def get(self, _name):
        raise docker.errors.NotFound("missing")

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return object()


class _Client:
    def __init__(self) -> None:
        self.images = _Images()
        self.containers = _Containers()


def test_build_container_mounts_budget_status_dir_read_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    status_path = tmp_path / "gen_1" / "budget" / "status.md"
    status_path.parent.mkdir(parents=True)
    status_path.write_text("remaining: 1\n", encoding="utf-8")
    client = _Client()

    container = build_container(
        client,
        repo_path=str(repo),
        image_name="app",
        container_name="unit",
        budget_status_path=str(status_path),
        verbose=False,
    )

    assert container is not None
    volumes = client.containers.run_kwargs["volumes"]
    assert volumes[str(repo.resolve())] == {"bind": f"/{REPO_NAME}", "mode": "rw"}
    assert volumes[str(status_path.parent.resolve())] == {
        "bind": BUDGET_STATUS_CONTAINER_DIR,
        "mode": "ro",
    }


class _PodmanContainer:
    pass


class _PodmanContainers:
    def __init__(self) -> None:
        self.get_calls = 0

    def get(self, _name):
        self.get_calls += 1
        if self.get_calls == 1:
            raise docker.errors.NotFound("missing")
        return _PodmanContainer()


class _Network:
    attrs = {"IPAM": {"Config": [{"Gateway": "172.31.0.1"}]}}

    def reload(self) -> None:
        pass


class _Networks:
    def get(self, _name):
        return _Network()

    def create(self, *_args, **_kwargs):
        return _Network()


class _PodmanClient:
    def __init__(self) -> None:
        self.images = _Images()
        self.containers = _PodmanContainers()
        self.networks = _Networks()
        self.api = type("Api", (), {"base_url": "unix://podman.sock"})()

    def info(self):
        return {
            "ServerVersion": "podman 5.0",
            "Runtimes": {"crun": {}},
        }


def test_podman_gpu_path_uses_cost_proxy_network_and_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(docker_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_utils, "verify_gpu_in_container", lambda *_args, **_kwargs: True)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")

    container = build_container(
        _PodmanClient(),
        repo_path=str(repo),
        image_name="app",
        container_name="unit",
        domains=["genesis"],
        cost_proxy_enabled=True,
        cost_proxy_network="rs-unit-proxy",
        verbose=False,
    )

    assert isinstance(container, _PodmanContainer)
    cmd = captured["cmd"]
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "rs-unit-proxy"
    assert "--network=host" not in cmd
    assert "--add-host" in cmd
    assert cmd[cmd.index("--add-host") + 1] == "proxy:172.31.0.1"
    assert "OPENAI_API_KEY=secret-key" in cmd

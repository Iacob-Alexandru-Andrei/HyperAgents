import io
import logging
import os
import tarfile
import threading
import warnings
from pathlib import Path
from typing import Optional, Union

import docker
from docker.models.containers import Container
from docker.types import Mount

from utils.constants import REPO_NAME

warnings.filterwarnings(
    "ignore",
    message=r"The default behavior of tarfile extraction has been changed.*",
    category=RuntimeWarning,
)

# Thread-local storage for loggers
_thread_local = threading.local()


def get_thread_logger():
    """Get the logger instance specific to the current thread."""
    return getattr(_thread_local, "logger", None)


def setup_logger(log_file):
    """
    Set up a thread-safe logger with file handler.

    Args:
        log_file (str): Path to the log file

    Returns:
        logging.Logger: Thread-specific logger instance
    """
    # Create logger with thread-specific name
    thread_id = threading.get_ident()
    logger_name = f"docker_logger_{thread_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Create file handler with lock
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.stream.lock = threading.Lock()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Store logger in thread local storage
    _thread_local.logger = logger

    return logger


def safe_log(message: str, level: int = logging.INFO, verbose=True):
    """Thread-safe logging function."""
    if verbose:
        logger = get_thread_logger()
        if logger:
            logger.log(level, message)
        else:
            print(f"Warning: No logger found for thread {threading.get_ident()}")


def log_container_output(exec_result, verbose=True):
    """
    Log output from a Docker container execution, handling both streaming and non-streaming cases.
    """
    # Handle output logging
    if isinstance(exec_result.output, bytes):
        # Handle non-streaming output
        safe_log(f"Container output: {exec_result.output.decode()}", verbose=verbose)
    else:
        # Handle streaming output
        for chunk in exec_result.output:
            if chunk:
                safe_log(f"Container output: {chunk.decode().strip()}", verbose=verbose)

    # Check exit code
    if exec_result.exit_code and exec_result.exit_code != 0:
        error_msg = f"Script failed with exit code {exec_result.exit_code}"
        safe_log(error_msg, logging.ERROR, verbose=verbose)
        raise Exception(error_msg)


def build_container(
    client,
    repo_path="./",
    image_name="app",
    container_name="app-container",
    force_rebuild=False,
    domains=None,
    verbose=True,
    cost_proxy_enabled=None,
    cost_proxy_network=None,
    cost_proxy_health_url=None,
    cost_proxy_deadline_ts=None,
    cost_proxy_budget_usd=None,
):
    """
    Build the Docker image with proxy and host networking, then run it interactively.
    """
    try:
        # F2e: env-var-driven proxy. The original initial-commit hardcoded
        # ``http://fwdproxy:8080`` (Meta-internal); on any other host that
        # proxy is unreachable so ``apt-get update`` inside the build returns
        # exit 100. Honor the host's proxy env vars when set; otherwise pass
        # nothing and let docker's host networking reach the package mirrors
        # directly.
        proxy_env: dict[str, str] = {}
        for build_key, *env_keys in (
            ("https_proxy", "https_proxy", "HTTPS_PROXY"),
            ("http_proxy", "http_proxy", "HTTP_PROXY"),
            ("ftp_proxy", "ftp_proxy", "FTP_PROXY"),
        ):
            for env_key in env_keys:
                if os.environ.get(env_key):
                    proxy_env[build_key] = os.environ[env_key]
                    break
        no_proxy = os.environ.get("no_proxy") or os.environ.get("NO_PROXY")
        if no_proxy:
            proxy_env["http_no_proxy"] = no_proxy

        # Check if we need to rebuild
        image_exists = any(
            image_name in tag for img in client.images.list() for tag in img.tags
        )
        if force_rebuild or not image_exists:
            safe_log(
                "Building the Docker image with proxy and host networking...",
                verbose=verbose,
            )
            image, logs = client.images.build(
                path=repo_path,  # <-- This points to the directory containing Dockerfile
                tag=image_name,
                rm=True,
                network_mode="host",
                nocache=force_rebuild,
                buildargs=proxy_env,
            )
            for log_entry in logs:
                if "stream" in log_entry:
                    safe_log(log_entry["stream"].strip())
            safe_log("Image built successfully.", verbose=verbose)
        else:
            safe_log(
                f"Docker image '{image_name}' already exists. Skipping build.",
                verbose=verbose,
            )
            image = next(
                (img for img in client.images.list() if image_name in img.tags), None
            )

    except Exception as e:
        safe_log(f"Error while building the Docker image: {e}")
        return None

    try:
        # Remove existing container if it exists
        try:
            existing = client.containers.get(container_name)
            existing.remove(force=True)
            safe_log(f"Removed existing container '{container_name}'.", verbose=verbose)
        except docker.errors.NotFound:
            pass

        # Build device_requests for GPU passthrough (conditional)
        device_requests = []
        is_podman = False  # Track if we're using Podman
        # Enable GPU only if domain contains "genesis"
        needs_gpu = domains is not None and any(
            "genesis" in domain.lower() for domain in domains
        )

        if needs_gpu:
            # Check if GPU is available on the host
            safe_log(
                "GPU requested. Checking if GPU is available on the host...",
                verbose=verbose,
            )
            try:
                info = client.info()
                gpu_info = info.get("Runtimes", {})
                has_nvidia_runtime = "nvidia" in gpu_info

                # Podman detection: Check ServerVersion, base_url, or presence of podman-specific runtimes
                # Note: Due to docker alias (alias docker=podman), the Python client may connect via
                # different interfaces that don't expose the podman socket path directly
                server_version = info.get("ServerVersion", "").lower()
                base_url = str(client.api.base_url).lower()
                runtimes = info.get("Runtimes", {})

                # Podman typically uses crun/runc instead of nvidia runtime
                has_podman_runtimes = any(
                    rt in runtimes for rt in ["crun", "crun-vm", "crun-wasm"]
                )
                is_podman = (
                    "podman" in server_version
                    or "podman" in base_url
                    or has_podman_runtimes
                )

                safe_log(
                    f"Debug - ServerVersion: {info.get('ServerVersion', 'N/A')}, has_nvidia_runtime: {has_nvidia_runtime}, is_podman: {is_podman}",
                    verbose=verbose,
                )

                # Enable GPU via device_requests for Docker or devices for Podman
                if has_nvidia_runtime:
                    # Docker with nvidia runtime uses device_requests
                    device_requests = [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]
                    safe_log(
                        "GPU access enabled for container (Docker nvidia runtime).",
                        verbose=verbose,
                    )
                elif is_podman:
                    # Podman needs explicit device mounts (device_requests doesn't work with Podman)
                    # This is a workaround - we'll use device_requests but log a warning
                    device_requests = [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]
                    safe_log(
                        "Detected Podman (via crun/runc runtimes) - attempting GPU via device_requests.",
                        verbose=verbose,
                    )
                    safe_log(
                        "Note: If GPU is not accessible, Podman may need explicit --device flags.",
                        verbose=verbose,
                    )
                else:
                    safe_log(
                        "Warning: GPU requested but neither nvidia-docker runtime nor Podman detected. Running without GPU.",
                        verbose=verbose,
                    )
            except Exception as e:
                safe_log(
                    f"Warning: Could not verify GPU availability: {e}. Running without GPU.",
                    verbose=verbose,
                )
        else:
            safe_log("GPU not requested. Running without GPU.", verbose=verbose)

        # Run the container with host networking and volume mount.
        # For Podman, we need to pass GPU devices explicitly via security_opt or devices.
        #
        # F2c+F2h+F2g+Tier3 environment propagation. ``OPENAI_API_KEY`` is
        # the conventional secret-flow inheritance pattern. Recursive
        # scientist passes cost-proxy settings explicitly so concurrent
        # container launches do not race through process-global env; the
        # legacy env fallback remains for direct upstream usage.
        if cost_proxy_enabled is None:
            cost_proxy_env = {
                key: os.environ.get(key, "")
                for key in (
                    "RS_COST_PROXY_HEALTH_URL",
                    "RS_EXPANSION_DEADLINE_TS",
                    "RS_COST_PROXY_BUDGET_USD",
                )
            }
        else:
            cost_proxy_env = {
                "RS_COST_PROXY_HEALTH_URL": cost_proxy_health_url or "",
                "RS_EXPANSION_DEADLINE_TS": cost_proxy_deadline_ts or "",
                "RS_COST_PROXY_BUDGET_USD": cost_proxy_budget_usd or "",
            }
        run_kwargs = {
            "image": image_name,
            "name": container_name,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "network_mode": "host",
            "volumes": {
                os.path.abspath(repo_path): {"bind": f"/{REPO_NAME}", "mode": "rw"}
            },
            "environment": {
                # F2i (recursive-scientist deviation): alias ``NVIDIA_API_KEY``
                # to ``OPENAI_API_KEY`` here so the host's ``experiments/run.py``
                # never has to write the parent process's ``os.environ``
                # (rs.no-os-environ-anywhere absolute rule). The in-container
                # litellm wrapper at ``hyperagents/agent/llm.py`` still reads
                # ``OPENAI_API_KEY`` exclusively; the alias is scoped to the
                # container's environment dict only.
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")
                or os.environ.get("NVIDIA_API_KEY", ""),
                **cost_proxy_env,
            },
            "command": "tail -f /dev/null",
        }

        # F2g (recursive-scientist Tier-2 cost proxy): when enabled, switch
        # off host networking and confine the container to a custom bridge
        # that resolves ``proxy`` to the bridge gateway.
        _apply_cost_proxy_network_lockdown(
            client,
            run_kwargs,
            enabled=cost_proxy_enabled,
            network_name=cost_proxy_network,
            verbose=verbose,
        )

        # Add GPU support
        if device_requests:
            if is_podman:
                # Podman's Python SDK doesn't support device_requests properly
                # We need to use Podman CLI directly to get --gpus all working
                safe_log(
                    "Using Podman CLI directly for GPU support (Python SDK doesn't support --gpus properly)",
                    verbose=verbose,
                )

                # Build the podman run command with CDI GPU support
                # Podman 5.x uses CDI (Container Device Interface) instead of --gpus
                import subprocess

                volume_mount = f"{os.path.abspath(repo_path)}:/{REPO_NAME}:rw"
                cmd = [
                    "podman",
                    "run",
                    "-d",  # detach
                    "-it",  # interactive + tty
                    "--network=host",
                    "-v",
                    volume_mount,
                    "--device",
                    "nvidia.com/gpu=all",  # CDI format for Podman 5.x
                    # Add environment variables for NVIDIA libraries
                    "-e",
                    "NVIDIA_VISIBLE_DEVICES=all",
                    "-e",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                    "-e",
                    "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
                    "--name",
                    container_name,
                    image_name,
                    "tail",
                    "-f",
                    "/dev/null",
                ]

                safe_log(f"Running Podman command: {' '.join(cmd)}", verbose=verbose)
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    raise Exception(f"Podman run failed: {result.stderr}")

                # Log both stdout and stderr for debugging
                if result.stdout:
                    safe_log(f"Podman stdout: {result.stdout.strip()}", verbose=verbose)
                if result.stderr:
                    safe_log(f"Podman stderr: {result.stderr.strip()}", verbose=verbose)

                # Get the container object from the client
                container = client.containers.get(container_name)
            else:
                # Docker with nvidia runtime
                run_kwargs["device_requests"] = device_requests
                container = client.containers.run(**run_kwargs)
        else:
            container = client.containers.run(**run_kwargs)
        safe_log(f"Container '{container_name}' started successfully.", verbose=verbose)

        # Verify GPU access if GPU was requested
        if needs_gpu and device_requests:
            safe_log("Verifying GPU access inside container...", verbose=verbose)
            gpu_accessible = verify_gpu_in_container(container, verbose=verbose)
            if gpu_accessible:
                safe_log(
                    "GPU verification PASSED - GPU is accessible in container!",
                    verbose=verbose,
                )
            else:
                safe_log(
                    "GPU verification FAILED - GPU is NOT accessible in container!",
                    verbose=verbose,
                )

        return container
    except Exception as e:
        safe_log(f"Error while starting the container: {e}", verbose=verbose)
        return None


def create_archive(path: Union[str, Path], data: Optional[bytes] = None) -> bytes:
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        if data is not None:
            tarinfo = tarfile.TarInfo(name=str(path))
            tarinfo.size = len(data)
            tarinfo.uid = 0
            tarinfo.gid = 0
            tar.addfile(tarinfo, io.BytesIO(data))
        else:
            path = Path(path)
            arcname = path.name
            for item in path.rglob("*"):
                tarinfo = tar.gettarinfo(
                    str(item), arcname=str(item.relative_to(path.parent))
                )
                tarinfo.uid = 0
                tarinfo.gid = 0
                if item.is_file():
                    with open(item, "rb") as f:
                        tar.addfile(tarinfo, f)
                else:
                    tar.addfile(tarinfo)
    tar_stream.seek(0)
    return tar_stream.read()


def copy_to_container(
    container, source_path: Union[str, Path], dest_path: Union[str, Path], verbose=True
) -> None:
    """
    Copy a file or directory from the local system to a Docker container.

    Args:
        container: Docker container object
        source_path (Union[str, Path]): Path to the source file/directory on local system
        dest_path (Union[str, Path]): Destination path in the container

    Raises:
        FileNotFoundError: If source path doesn't exist
        Exception: For other errors during copy operation
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        # Destination directory inside the container
        container_dest_dir = str(dest_path.parent)

        # Build tar stream in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if source_path.is_file():
                # Single file -> archive entry name is dest_path.name
                arcname = dest_path.name
                tarinfo = tar.gettarinfo(str(source_path), arcname=arcname)
                tarinfo.uid = 0
                tarinfo.gid = 0
                with open(source_path, "rb") as f:
                    tar.addfile(tarinfo, f)
            else:
                # Directory -> we want the top-level directory in the archive
                # to be dest_path.name, with its contents under it.
                def _reset_uid_gid(ti: tarfile.TarInfo) -> tarfile.TarInfo:
                    ti.uid = 0
                    ti.gid = 0
                    return ti

                tar.add(
                    str(source_path),
                    arcname=dest_path.name,
                    filter=_reset_uid_gid,
                )

        # Rewind the stream before sending
        tar_stream.seek(0)

        # Ensure destination directory exists in container
        container.exec_run(f"mkdir -p {container_dest_dir}")

        # Pass the file-like object (stream) to put_archive
        success = container.put_archive(container_dest_dir, tar_stream)

        if not success:
            raise Exception(f"Failed to copy {source_path} to container")

        safe_log(
            f"Successfully copied {source_path} to container at {dest_path}",
            verbose=verbose,
        )

    except Exception as e:
        safe_log(f"Error copying to container: {e}", logging.ERROR, verbose=verbose)
        raise


def copy_from_container(
    container, source_path: Union[str, Path], dest_path: Union[str, Path], verbose=True
) -> None:
    """
    Copy a file or directory from a Docker container to the local system.

    Args:
        container: Docker container object
        source_path (Union[str, Path]): Path to the source file/directory in container
        dest_path (Union[str, Path]): Destination path on local system

    Raises:
        FileNotFoundError: If source path doesn't exist in container
        Exception: For other errors during copy operation
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        # Check if source exists in container
        result = container.exec_run(f"test -e {source_path}")
        if result.exit_code and result.exit_code != 0:
            raise FileNotFoundError(
                f"Source path not found in container: {source_path}"
            )

        # Get file type from container
        result = container.exec_run(f"stat -f '%HT' {source_path}")
        is_file = result.output.decode().strip() == "Regular File"

        # Create destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Get archive from container
        bits, stat = container.get_archive(str(source_path))

        # Concatenate all chunks into a single bytes object
        archive_data = b"".join(bits)

        # Extract to temporary stream
        stream = io.BytesIO(archive_data)

        with tarfile.open(fileobj=stream, mode="r") as tar:
            # If extracting a single file
            if is_file:
                member = tar.getmembers()[0]
                source_file = tar.extractfile(member)
                if source_file is not None:
                    with source_file:
                        data = source_file.read()
                        # Write directly to destination file
                        with open(dest_path, "wb") as dest_file:
                            dest_file.write(data)
            else:
                # For directories, extract to parent directory
                tar.extractall(path=str(dest_path.parent))
                # Rename if necessary
                extracted_path = dest_path.parent / Path(stat["name"]).name
                if extracted_path != dest_path and extracted_path.exists():
                    extracted_path.rename(dest_path)

        safe_log(
            f"Successfully copied from container {source_path} to local path {dest_path}",
            verbose=verbose,
        )

    except Exception as e:
        if verbose:
            safe_log(
                f"Error copying from container: {e}", logging.ERROR, verbose=verbose
            )
        raise


def verify_gpu_in_container(container, verbose=True) -> bool:
    """
    Verify GPU is accessible inside a running container.

    Similar to the interactive test:
    - Check if nvidia-smi is available
    - Check if /dev/nvidia* devices are mounted
    - Check if PyTorch can see the GPU

    Args:
        container: Docker container object
        verbose: Whether to log output

    Returns:
        bool: True if GPU is accessible, False otherwise
    """
    try:
        # Test 1: Check if GPU devices are mounted (most important check)
        # Use sh -c to enable shell features like redirection
        result = container.exec_run(
            ["sh", "-c", "ls /dev/nvidia* 2>/dev/null || echo 'not found'"]
        )
        output = result.output.decode().strip()
        if "not found" in output:
            safe_log("GPU check: /dev/nvidia* devices not found", verbose=verbose)
            # Check what's actually in /dev to debug
            dev_result = container.exec_run(["sh", "-c", "ls /dev | head -20"])
            safe_log(
                f"GPU check: /dev contents (first 20): {dev_result.output.decode().strip()}",
                verbose=verbose,
            )
            return False
        safe_log(f"GPU check: GPU devices found - {output}", verbose=verbose)

        # Test 2: Check if nvidia-smi is available (optional - might not be in PATH)
        result = container.exec_run("which nvidia-smi")
        if result.exit_code != 0:
            safe_log(
                "GPU check: nvidia-smi not in PATH, checking /usr/bin/nvidia-smi",
                verbose=verbose,
            )
            result = container.exec_run(
                "test -f /usr/bin/nvidia-smi && echo 'found' || echo 'not found'"
            )
            if "not found" in result.output.decode():
                safe_log(
                    "GPU check: nvidia-smi not found, but GPU devices are present",
                    verbose=verbose,
                )
        else:
            safe_log("GPU check: nvidia-smi found", verbose=verbose)

        # Test 3: Try running nvidia-smi
        result = container.exec_run("nvidia-smi")
        if result.exit_code != 0:
            safe_log(
                f"GPU check: nvidia-smi failed - {result.output.decode()}",
                verbose=verbose,
            )
            return False
        safe_log("GPU check: nvidia-smi executed successfully", verbose=verbose)

        # Test 4: Check LD_LIBRARY_PATH and NVIDIA libraries
        result = container.exec_run(["sh", "-c", "echo $LD_LIBRARY_PATH"])
        ld_library_path = result.output.decode().strip()
        safe_log(f"GPU check: LD_LIBRARY_PATH = {ld_library_path}", verbose=verbose)

        # Check if NVIDIA libraries are accessible
        result = container.exec_run(
            [
                "sh",
                "-c",
                "ls /usr/local/nvidia/lib* 2>/dev/null | head -5 || echo 'NVIDIA lib dirs not found'",
            ]
        )
        safe_log(
            f"GPU check: NVIDIA lib check: {result.output.decode().strip()}",
            verbose=verbose,
        )

        # Test 5: Check PyTorch GPU availability (if PyTorch is installed)
        result = container.exec_run(
            'python -c "import torch; print(torch.cuda.is_available())"'
        )
        if result.exit_code == 0:
            cuda_available = result.output.decode().strip()
            safe_log(
                f"GPU check: PyTorch CUDA available = {cuda_available}", verbose=verbose
            )
            if cuda_available.lower() != "true":
                return False
        else:
            # PyTorch might not be installed, but GPU devices are present
            safe_log(
                "GPU check: PyTorch not available, but GPU devices are present",
                verbose=verbose,
            )

        # Test 6: Check Genesis GPU initialization (if Genesis is installed)
        result = container.exec_run(
            "python -c \"import genesis as gs; gs.init(backend=gs.cuda, logging_level='warning'); print(gs.device)\""
        )
        if result.exit_code == 0:
            genesis_device = result.output.decode().strip()
            safe_log(
                f"GPU check: Genesis initialized with device = {genesis_device}",
                verbose=verbose,
            )
            return "cuda" in genesis_device.lower()
        else:
            # Genesis might not be installed, but GPU devices are present
            safe_log(
                "GPU check: Genesis not available or failed to initialize, but GPU devices are present",
                verbose=verbose,
            )
            return True

    except Exception as e:
        safe_log(f"GPU check failed with error: {e}", verbose=verbose)
        return False


def cleanup_container(container, verbose=True):
    """
    Stop and remove a container cleanly.

    Args:
        container (docker.models.containers.Container): The container to stop and remove.
    """
    try:
        safe_log(f"Stopping container {container.name}...", verbose=verbose)
        container.stop(timeout=10)
    except Exception as e:
        safe_log(
            f"Error while stopping container {container.name}: {e}",
            level=logging.WARNING,
            verbose=verbose,
        )

    try:
        safe_log(f"Removing container {container.name}...", verbose=verbose)
        container.remove(force=True)
    except Exception as e:
        safe_log(
            f"Error while removing container {container.name}: {e}",
            level=logging.ERROR,
            verbose=verbose,
        )


# F2g: recursive-scientist Tier-2 cost-enforcing egress proxy.
#
# When the cost proxy is enabled, this hook rewrites ``run_kwargs`` to:
#
# 1. Drop ``network_mode="host"`` and join the container to a per-run
#    custom bridge network. The bridge is reused across expansions of
#    the same run so we don't churn through docker-network slots.
# 2. Add ``extra_hosts={"proxy": <host_ip>}`` so the in-container
#    catalog's ``@http://proxy:<port>/v1`` entries resolve to the
#    proxy listener on the host. We use the explicit IP rather than
#    ``host.docker.internal`` because Linux Docker historically did
#    not enable that name; explicitly mapping it works on every
#    runtime we support.
#
# Default behavior is unchanged when disabled: the container keeps host
# networking and any catalog ``@<base_url>`` resolves directly to the
# upstream provider.
_DEFAULT_COST_PROXY_NETWORK = "rs-cost-proxy"
_COST_PROXY_NETWORK_LOCK = threading.Lock()


def _apply_cost_proxy_network_lockdown(
    client,
    run_kwargs,
    *,
    enabled=None,
    network_name=None,
    verbose=True,
):
    """F2j (recursive-scientist deviation): attach container to the cost-proxy
    internal bridge and point ``extra_hosts.proxy`` at the **bridge gateway IP**
    (the host's interface on the bridge), NOT the host's external IP.

    ``internal=True`` bridges have no route to the host's external IP — only the
    bridge gateway (e.g. ``172.18.0.1``) is reachable from attached containers.
    Direct callers can still enable this through the legacy
    ``RS_COST_PROXY_HOST_IP`` sentinel; recursive-scientist passes
    ``enabled`` explicitly to avoid process-global env races.
    """
    env_enabled = os.environ.get("RS_COST_PROXY_HOST_IP")
    if enabled is None:
        enabled = bool(env_enabled)
    if not enabled:
        return
    if network_name is None:
        network_name = os.environ.get("RS_COST_PROXY_NETWORK", _DEFAULT_COST_PROXY_NETWORK)
    _ensure_cost_proxy_network(client, network_name, verbose=verbose)
    # Discover the bridge gateway IP after the network exists.
    network = client.networks.get(network_name)
    network.reload()
    ipam_config = (network.attrs.get("IPAM") or {}).get("Config") or []
    gateway_ip = None
    for cfg in ipam_config:
        candidate = cfg.get("Gateway") if isinstance(cfg, dict) else None
        if candidate:
            gateway_ip = str(candidate)
            break
    if not gateway_ip:
        raise RuntimeError(
            f"Cost-proxy network {network_name} has no IPAM gateway; cannot route "
            "internal containers to the host-side proxy. Recreate the network."
        )
    run_kwargs.pop("network_mode", None)
    run_kwargs["network"] = network_name
    extra_hosts = dict(run_kwargs.get("extra_hosts") or {})
    extra_hosts["proxy"] = gateway_ip
    run_kwargs["extra_hosts"] = extra_hosts
    safe_log(
        f"Cost-proxy lockdown active: network={network_name} "
        f"extra_hosts.proxy={gateway_ip} (bridge gateway)",
        verbose=verbose,
    )


def _ensure_cost_proxy_network(client, network_name, *, verbose=True):
    """Create the custom internal bridge network if missing; idempotent.

    Plan-item #2 lockdown: ``internal=True`` drops the network's outbound
    NAT so containers attached to it cannot reach the public internet
    directly -- only the bridge gateway, where the host-bound proxy
    listens, is reachable. The catalog rewrite still names ``proxy`` as
    the upstream; the network-level confinement is the second layer that
    prevents a malicious meta-agent from bypassing the proxy by editing
    catalog model strings to point at a public IP, shelling out to
    ``curl https://api.openai.com``, or otherwise routing around the
    catalog. ``internal=True`` is silently honored by docker on both
    Linux and Podman.
    """
    with _COST_PROXY_NETWORK_LOCK:
        try:
            existing = client.networks.get(network_name)
        except docker.errors.NotFound:
            existing = None
        except Exception as exc:
            safe_log(
                f"Could not query docker network {network_name}: {exc}; attempting create",
                level=logging.WARNING,
                verbose=verbose,
            )
            existing = None
        if existing is None:
            try:
                client.networks.create(network_name, driver="bridge", internal=True)
                safe_log(
                    f"Created docker bridge network {network_name} with internal=True "
                    "(no public egress)",
                    verbose=verbose,
                )
            except Exception as exc:
                try:
                    existing = client.networks.get(network_name)
                except Exception:
                    safe_log(
                        f"Failed to create docker network {network_name}: {exc}",
                        level=logging.ERROR,
                        verbose=verbose,
                    )
                    raise
        if existing is not None:
            # If a previous run created the network as non-internal we can't
            # silently upgrade it (would orphan running containers). Surface
            # the divergence and proceed; the operator can ``docker network
            # rm <name>`` to force a clean re-create.
            attrs = getattr(existing, "attrs", {}) or {}
            is_internal = bool(attrs.get("Internal"))
            if not is_internal:
                safe_log(
                    f"Cost-proxy network {network_name} exists but is NOT internal; "
                    "containers may have egress to public internet. Recreate the "
                    "network to enforce lockdown: `docker network rm "
                    f"{network_name}` and retry.",
                    level=logging.WARNING,
                    verbose=verbose,
                )

# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import docker

from utils.common import file_exist_and_not_empty, load_json_file
from utils.constants import REPO_NAME
from utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    safe_log,
    setup_logger,
)
from utils.domain_utils import (
    get_domain_eval_subset,
    get_domain_splits,
    get_domain_stagedeval_samples,
)
from utils.gl_utils import (
    apply_diffs_container,
    get_patch_files,
    get_score,
    run_commands_to_check_compilation,
    setup_initial_gen,
    update_node_metadata,
    is_starting_node,
    process_meta_patch_files,
)

BUDGET_STATUS_CONTAINER_DIR = "/rqgm_budget"


def _container_budget_status_path(budget_status_path):
    if not budget_status_path:
        return None
    return os.path.join(
        BUDGET_STATUS_CONTAINER_DIR,
        os.path.basename(budget_status_path),
    )


def _rewrite_model_for_proxy(model, cost_proxy_base_url):
    """Rewrite a model string's ``@<url>`` suffix to point at the cost proxy.

    ``cost_proxy_base_url`` is the bridge-internal URL (e.g.
    ``http://proxy:9100/v1``); model strings without an ``@<url>`` suffix pass
    through unchanged.
    """
    if not cost_proxy_base_url or not model or "@" not in model:
        return model
    head, sep, _suffix = model.partition("@")
    if not sep:
        return model
    return f"{head}@{cost_proxy_base_url}"


def _upstream_hostname_from_model(model):
    """Extract the upstream hostname (e.g. ``inference-api.nvidia.com``) from
    a model string with an ``@<base_url>`` suffix. Returns ``None`` when the
    model has no suffix or the suffix is unparseable.
    """
    if not model or "@" not in model:
        return None
    _head, _sep, suffix = model.partition("@")
    if not suffix:
        return None
    from urllib.parse import urlparse

    parsed = urlparse(suffix)
    return parsed.hostname or None


def run_harness_polyglot(root_dir, output_dir, genid, *, model, skip_staged_eval=False, num_samples=-1):
    # NOTE: the harness for polyglot is different because each task instance needs a docker container
    from domains.polyglot.harness import harness as harness_polyglot
    from domains.polyglot.report import report as report_polyglot

    eval_output_dir = os.path.join(output_dir, f"gen_{genid}", "polyglot_eval")
    test_more_threshold = 0.4  # NOTE: same setting as that in DGM
    model_name_or_path = "eval_run"
    patch_files = get_patch_files(output_dir, genid)
    run_next_eval = True

    # Small sample size evaluation for staged eval
    if not skip_staged_eval:
        test_task_list = load_json_file("./domains/polyglot/subsets/small.json")
        dnames = harness_polyglot(
            test_task_list=test_task_list,
            num_samples=-1,
            max_workers=10,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=eval_output_dir,
            output_dir=eval_output_dir,
            root_dir=root_dir,
            model=model,
        )
        report_polyglot(output_dir=eval_output_dir, run_keyword=model_name_or_path, expected_num_tasks=len(test_task_list))
        stagedeval_score = get_score("polyglot", output_dir, genid)
        run_next_eval = stagedeval_score is not None and stagedeval_score >= test_more_threshold

    # Check if additional evaluation should be run
    if run_next_eval:
        test_task_list_more = load_json_file("./domains/polyglot/subsets/medium.json")
        dnames = harness_polyglot(
            test_task_list=test_task_list + test_task_list_more,
            num_samples=num_samples,
            max_workers=10,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=eval_output_dir,
            output_dir=eval_output_dir,
            root_dir=root_dir,
            model=model,
        )
        report_polyglot(output_dir=eval_output_dir, run_keyword=model_name_or_path, expected_num_tasks=len(test_task_list + test_task_list_more))

    # Update metadata
    update_node_metadata(output_dir, genid, {"run_full_eval": run_next_eval})

def eval_produced_agent(
    container,
    container_output_folder,
    gen_output_dir,
    domain,
    model,
    eval_samples=-1,
    eval_workers=10,
    eval_subset="_filtered_100_train",
    eval_test=False,
    reasoning_effort=None,
    splits=None,
    budget_status_path=None,
):
    # When ``splits`` is supplied (a non-None list[str]), iterate over exactly
    # those splits; otherwise fall back to ``get_domain_splits``. The explicit
    # parameter lets concurrent callers with different splits avoid racing on
    # the process-global lookup.
    if splits is None:
        splits = get_domain_splits(domain, eval_test=eval_test)
    for split in splits:  # pyright: ignore
        safe_log(f"Evaluating the produced agent on {domain} {eval_samples} {split}...")
        eval_run_id = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
        container_evaloutput_folder = os.path.join(container_output_folder, eval_run_id)
        command = [
            "timeout",
            "18000",  # 5h timeout
            "python",
            "-m",
            "domains.harness",
            "--agent_path",
            "./task_agent.py",
            "--output_dir",
            container_output_folder,
            "--run_id",
            eval_run_id,
            "--domain",
            domain,
            "--num_samples",
            str(eval_samples),
            "--num_workers",
            str(eval_workers),
            "--subset",
            eval_subset.replace("_train", f"_{split}"),
            "--model",
            model,
        ]
        if reasoning_effort:
            command += ["--reasoning_effort", reasoning_effort]
        if budget_status_path:
            command += ["--budget_status_path", budget_status_path]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        command = [
            "timeout",
            "10800",  # 3h timeout
            "python",
            "-m",
            "domains.report",
            "--domain",
            domain,
            "--dname",
            os.path.join(container_output_folder, eval_run_id),
            "--model",
            model,
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        # Copy container outputs to local, evaluation results
        evaloutput_folder = os.path.join(gen_output_dir, eval_run_id)
        copy_from_container(
            container,
            source_path=container_evaloutput_folder,
            dest_path=evaloutput_folder,
        )


def copy_prev_eval_to_container(
    container,
    prev_eval_path,
    container_output_folder,
    current_genid=None,
    parent_genid=None,
    container_folder_name=None,
):
    """Copy the entire prev_eval_path into the container, then remove unwanted files/dirs in the container"""
    if not os.path.exists(prev_eval_path):
        raise FileNotFoundError(f"Previous eval path not found: {prev_eval_path}")

    # Normalize and construct destination path in container
    prev_eval_path = os.path.normpath(prev_eval_path)
    tail = os.path.join(*prev_eval_path.split(os.sep)[-1:])
    container_prev_eval_path = os.path.join(container_output_folder, tail)

    # Ensure destination parent exists
    container.exec_run(["mkdir", "-p", container_output_folder], workdir="/")

    # Copy the whole tree into the container in one go
    copy_to_container(
        container, source_path=prev_eval_path, dest_path=container_prev_eval_path
    )

    # When the run dir uses the v2 layout (gens/, pending/ subdirs), also
    # copy the ancestor gens from the sibling `gens/` so the meta-agent
    # sees every ancestor's predictions / chat history / metadata under
    # the same eval_path tree. In v1 there is no gens/ subdir; the walk
    # above already covered everything.
    gens_root = os.path.join(os.path.dirname(prev_eval_path), "gens")
    if prev_eval_path.endswith("/pending") and os.path.isdir(gens_root):
        for entry in os.listdir(gens_root):
            if not entry.startswith("gen_"):
                continue
            src = os.path.join(gens_root, entry)
            if not os.path.isdir(src):
                continue
            copy_to_container(
                container,
                source_path=src,
                dest_path=os.path.join(container_prev_eval_path, entry),
            )

    lineage_gen_dirs = _lineage_gen_dirs(prev_eval_path, parent_genid)
    non_lineage_prune_cmds = _non_lineage_prune_cmds(
        prev_eval_path, container_prev_eval_path, lineage_gen_dirs
    )

    # Now prune inside the container
    prune_cmds = [
        *non_lineage_prune_cmds,
        # Remove current genid folder
        f"find '{container_prev_eval_path}' -type d -name 'gen_{current_genid}' -prune -exec rm -rf {{}} +",
        # 1) Remove val/test eval directories
        f"find '{container_prev_eval_path}' -type d -name '*_eval_val*' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type d -name '*_eval_test*' -prune -exec rm -rf {{}} +",
        # 2) Remove any directories containing the repo name (copied worktrees, etc.)
        f"find '{container_prev_eval_path}' -type d -name '*{REPO_NAME}*' -prune -exec rm -rf {{}} +",
        # 3) Remove compiled Python files
        f"find '{container_prev_eval_path}' -type f -name '*.pyc' -delete",
        # 4) Remove files whose base name indicates val/test (with/without extensions)
        #    *_val, *_val.*, *_val_*, and same for _test
        f"find '{container_prev_eval_path}' -type f \\( -name '*_val' -o -name '*_val.*' -o -name '*_val_*' \\) -delete",
        f"find '{container_prev_eval_path}' -type f \\( -name '*_test' -o -name '*_test.*' -o -name '*_test_*' \\) -delete",
    ]

    for cmd in prune_cmds:
        exec_result = container.exec_run(["bash", "-lc", cmd], workdir="/")

    # Confirm files remaining were copied
    exec_result = container.exec_run(
        ["ls", "-l", container_prev_eval_path], workdir="/"
    )
    log_container_output(exec_result)

    # Move the folder to a new name
    if container_folder_name is not None:
        new_container_prev_eval_path = os.path.join(
            container_output_folder, container_folder_name
        )
        container.exec_run(
            ["mv", container_prev_eval_path, new_container_prev_eval_path], workdir="/"
        )
        log_container_output(exec_result)
        container_prev_eval_path = new_container_prev_eval_path

    return container_prev_eval_path


def _lineage_gen_dirs(output_dir, parent_genid):
    if parent_genid is None:
        return set()

    lineage_gen_dirs = set()
    seen = set()
    genid = parent_genid
    while genid is not None and genid not in seen:
        seen.add(genid)
        lineage_gen_dirs.add(f"gen_{genid}")
        genid = _read_parent_genid(output_dir, genid)

    return lineage_gen_dirs


def _non_lineage_prune_cmds(prev_eval_path, container_prev_eval_path, lineage_gen_dirs):
    non_lineage_prune_cmds = []
    if not lineage_gen_dirs:
        return non_lineage_prune_cmds
    for name in os.listdir(prev_eval_path):
        if name not in lineage_gen_dirs:
            non_lineage_prune_cmds.append(
                f"find '{container_prev_eval_path}' -mindepth 1 -maxdepth 1 -name '{name}' -exec rm -rf {{}} +"
            )
    return non_lineage_prune_cmds


def _read_parent_genid(output_dir, genid):
    candidates = [
        os.path.join(output_dir, f"gen_{genid}", "metadata.json"),
        os.path.join(os.path.dirname(output_dir), "gens", f"gen_{genid}", "metadata.json"),
    ]
    metadata_file = next((p for p in candidates if os.path.exists(p)), None)
    if metadata_file is None:
        return None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    parent_genid = metadata.get("parent_genid")
    if parent_genid == -1 or parent_genid == "-1":
        return None
    return parent_genid


def run_generation_step(
    docker_client,
    domains,
    output_dir,
    run_id,
    current_genid,
    parent_genid,
    root_dir,
    root_commit="main",
    eval_samples=-1,
    eval_workers=10,
    eval_subsets="_filtered_100",
    meta_patch_files=None,
    parent_patch_files=None,
    run_meta_agent=True,
    run_eval_after_meta_agent=True,
    eval_test=False,
    skip_staged_eval=False,
    iterations_left=0,
    *,
    model=None,
    meta_agent_model=None,
    task_agent_models=None,
    reasoning_effort=None,
    meta_agent_reasoning_effort=None,
    task_agent_reasoning_efforts=None,
    splits=None,
    cost_proxy_enabled=None,
    cost_proxy_network=None,
    cost_proxy_base_url=None,
    budget_status_path=None,
    produce_patch=True,
):
    """``produce_patch=True`` is the generator pass (train): it writes
    ``model_patch.diff`` (the meta-agent's code-only edit).
    ``produce_patch=False`` is a measurement pass (val/test/cross-eval):
    it applies the gen's full patch chain, runs eval, but does NOT
    touch ``model_patch.diff``."""
    # Per-role model routing: ``model`` is the uniform fallback (required by
    # the polyglot harness path). ``meta_agent_model`` overrides the meta-agent
    # invocation only; ``task_agent_models`` is a per-domain map for task-agent
    # eval calls. Either path falls back to ``model`` when its specific entry
    # is missing. ``reasoning_effort`` mirrors this shape: a uniform value
    # applies to all calls; ``meta_agent_reasoning_effort`` and
    # ``task_agent_reasoning_efforts`` override per call site. ``None`` means
    # "do not pass the parameter through".
    if model is None and meta_agent_model is None and not task_agent_models:
        raise TypeError(
            "run_generation_step requires `model=`, `meta_agent_model=`, or `task_agent_models=`"
        )
    meta_agent_model = meta_agent_model if meta_agent_model is not None else model
    _task_agent_models = dict(task_agent_models or {})
    meta_agent_reasoning_effort = (
        meta_agent_reasoning_effort
        if meta_agent_reasoning_effort is not None
        else reasoning_effort
    )
    _task_agent_reasoning_efforts = dict(task_agent_reasoning_efforts or {})
    # When the cost proxy is enabled, rewrite every ``--model`` arg's
    # ``@<url>`` suffix to point at the bridge-internal proxy URL. The catalog
    # is rewritten separately by ``rewrite_catalog_for_proxy``; the CLI
    # ``--model`` args are an independent channel and must be rewritten here so
    # the container can resolve the proxy hostname over the internal bridge.
    cost_proxy_upstream_hostnames = ()
    if cost_proxy_enabled and cost_proxy_base_url:
        # Collect upstream hostnames BEFORE rewriting so the lockdown can
        # also map them to the bridge gateway as a fail-fast safety net.
        upstream_hostnames = set()
        for source in (
            (model,) if model is not None else (),
            (meta_agent_model,) if meta_agent_model is not None else (),
            tuple(_task_agent_models.values()),
        ):
            for m in source:
                hostname = _upstream_hostname_from_model(m)
                if hostname:
                    upstream_hostnames.add(hostname)
        cost_proxy_upstream_hostnames = tuple(sorted(upstream_hostnames))
        if model is not None:
            model = _rewrite_model_for_proxy(model, cost_proxy_base_url)
        meta_agent_model = _rewrite_model_for_proxy(meta_agent_model, cost_proxy_base_url)
        _task_agent_models = {
            d: _rewrite_model_for_proxy(m, cost_proxy_base_url)
            for d, m in _task_agent_models.items()
        }
    # Setup local output folder
    prev_gen_dir = os.path.join(output_dir, f"gen_{parent_genid}")
    gen_output_dir = os.path.join(output_dir, f"gen_{current_genid}")
    os.makedirs(gen_output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(gen_output_dir, "generate.log"))  # Set up logger
    metadata = {
        "gen_output_dir": gen_output_dir,
        "current_genid": current_genid,
        "parent_genid": parent_genid,
        "prev_patch_files": [],
        "curr_patch_files": [],
        "parent_agent_success": not run_meta_agent,  # meta agent success if not run
    }
    run_eval = not run_meta_agent  # always run eval if not running meta agent
    metadata["run_eval"] = run_eval
    print(metadata)

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    container_name = f"{REPO_NAME}-gl-container-{run_id}"
    container = build_container(
        docker_client,
        root_dir,
        image_name,
        container_name,
        domains=domains,
        cost_proxy_enabled=cost_proxy_enabled,
        cost_proxy_network=cost_proxy_network,
        cost_proxy_upstream_hostnames=cost_proxy_upstream_hostnames,
        budget_status_path=budget_status_path,
    )
    container.start()
    container_output_folder = "/tmp/"
    container_budget_status_path = _container_budget_status_path(budget_status_path)

    try:
        # Apply meta patches (only for starting node, because subsequent generations will inherit the patches from the parent)
        if is_starting_node(current_genid):
            meta_patch_files = meta_patch_files or []
            commit_hash = apply_diffs_container(container, meta_patch_files)
            if produce_patch:
                metadata["prev_patch_files"] += meta_patch_files

        # Apply all lineage diffs
        patch_files = get_patch_files(output_dir, parent_genid) if parent_patch_files is None else parent_patch_files
        # Measurement passes (``produce_patch=False``) apply patches to set up
        # the workspace but must NOT mutate ``prev_patch_files`` -- the train
        # pass already recorded the canonical chain. For producer (train)
        # passes, ``parent_patch_files`` already contains the gen's own patch
        # path (passed as ``prev + curr``); apply the full chain to set up the
        # workspace but exclude self from ``prev_patch_files`` (self belongs in
        # ``curr``). Without this dedupe, ``prev + curr = [parent, self, self]``
        # and the second ``patch -p1`` of self fails with "file already exists".
        self_patch_path = os.path.normpath(
            os.path.join(gen_output_dir, "agent_output", "model_patch.diff")
        )
        if produce_patch:
            parent_chain_only = [
                p for p in patch_files if os.path.normpath(p) != self_patch_path
            ]
            metadata["prev_patch_files"] += parent_chain_only
        commit_hash = apply_diffs_container(container, patch_files)

        local_agentoutput_folder = os.path.join(gen_output_dir, "agent_output/")
        container_agentoutput_folder = os.path.join(
            container_output_folder, "agent_output"
        )
        os.makedirs(local_agentoutput_folder, exist_ok=True)
        container.exec_run(
            ["mkdir", "-p", container_agentoutput_folder], workdir="/"
        )

        if run_meta_agent:
            container_prev_eval_path = copy_prev_eval_to_container(
                container,
                output_dir,
                container_output_folder,
                current_genid=current_genid,
                parent_genid=parent_genid,
            )

            # Run meta agent
            safe_log("Running meta agent...")
            container_chat_history_file = os.path.join(
                container_agentoutput_folder, "meta_agent_chat_history.md"
            )
            command = [
                "timeout",
                "21600",  # 6h timeout
                "python",
                "run_meta_agent.py",
                "--chat_history_file",
                container_chat_history_file,
                "--repo_path",
                f"/{REPO_NAME}/",
                "--evals_folder",
                container_prev_eval_path,
                "--git_dir",
                f"/{REPO_NAME}",
                "--base_commit",
                commit_hash,
                "--outdir",
                container_agentoutput_folder,
                "--iterations_left",
                str(max(0, iterations_left)),
                "--model",
                meta_agent_model,
            ]
            if meta_agent_reasoning_effort:
                command += ["--reasoning_effort", meta_agent_reasoning_effort]
            if container_budget_status_path:
                command += ["--budget_status_path", container_budget_status_path]

            exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
            log_container_output(exec_result)
            metadata["parent_agent_success"] = exec_result.exit_code == 0
            # Persist the meta-agent exit code so the host can distinguish a
            # wall-clock kill (the in-container ``timeout 21600`` wrapper exits
            # 124) from a clean-exit ``parent_agent_success=False``.
            metadata["meta_agent_exit_code"] = int(exec_result.exit_code or 0)

            # Copy container outputs to local
            copy_from_container(
                container,
                source_path=container_agentoutput_folder,
                dest_path=local_agentoutput_folder,
            )

            # Check if agent produced a diff
            local_patch_file = os.path.join(
                local_agentoutput_folder, "model_patch.diff"
            )
            metadata["curr_patch_files"].append(local_patch_file)
            run_eval = file_exist_and_not_empty(local_patch_file) and run_eval_after_meta_agent
            metadata["run_eval"] = run_eval

            # Run commands to check if the agents are compilable
            run_commands_to_check_compilation(container)

        # Evaluate the produced agent
        if run_eval:
            log_path = os.path.join(gen_output_dir, "generate.log")

            def eval_agent_worker(domain, eval_subset, eval_n):
                setup_logger(log_path)  # Re-setup logger because of threading
                eval_produced_agent(
                    container,
                    container_output_folder,
                    gen_output_dir,
                    domain=domain,
                    eval_samples=eval_n,
                    eval_workers=eval_workers,
                    eval_subset=eval_subset,
                    eval_test=eval_test,
                    model=_task_agent_models.get(domain, model),
                    reasoning_effort=_task_agent_reasoning_efforts.get(
                        domain, reasoning_effort
                    ),
                    splits=splits,
                    budget_status_path=container_budget_status_path,
                )

            # Small sample size evaluation for staged eval
            if not skip_staged_eval:
                stagedeval_samples = [
                    get_domain_stagedeval_samples(domain) for domain in domains
                ]
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(eval_agent_worker, d, s, n)
                        for d, s, n in zip(domains, eval_subsets, stagedeval_samples)
                    ]
                    try:
                        for f in futures:
                            f.result()
                    except Exception as e:
                        # Cancel all other futures if any job fails
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        raise
                stagedeval_scores = [
                    get_score(domain, output_dir, current_genid) for domain in domains
                ]
                run_next_eval = all(
                    [x is not None and x > 0 for x in stagedeval_scores]
                )
            else:
                run_next_eval = True

            # Full evaluation
            if run_next_eval:
                _per_domain_eval_samples = eval_samples
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(eval_agent_worker, d, s, n)
                        for d, s, n in zip(
                            domains, eval_subsets, _per_domain_eval_samples
                        )
                    ]
                    try:
                        for f in futures:
                            f.result()
                    except Exception as e:
                        # Cancel all other futures if any job fails
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        raise
                metadata["run_full_eval"] = True

            if produce_patch:
                final_patch = os.path.join(local_agentoutput_folder, "model_patch.diff")
                if (
                    file_exist_and_not_empty(final_patch)
                    and final_patch not in metadata["curr_patch_files"]
                ):
                    metadata["curr_patch_files"].append(final_patch)

    except Exception as e:
        safe_log(f"Error in generate: {e}")
        metadata["run_eval"] = False

    # Even on errors or KeyboardInterrupt
    finally:
        # Reset to the root commit
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)

        # Cleanup container
        cleanup_container(container)

        # Save metadata
        eval_successful = all(
            [
                get_score(domain, output_dir, current_genid) is not None
                for domain in domains
            ]
        )
        metadata["valid_parent"] = metadata["run_eval"] and (eval_successful or meta_patch_files is not None)
        # Measurement passes must not overwrite the gen's canonical
        # metadata.json -- the train pass already wrote it with the
        # authoritative ``prev_patch_files`` / ``curr_patch_files`` chain, and
        # the measurement pass's local ``metadata`` dict would clobber it.
        if produce_patch:
            with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="Run ID")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",  # one or more domains
        choices=[
            "search_arena",
            "paper_review",
            "balrog_babyai",
            "balrog_babaisai",
            "balrog_minihack",
            "balrog_nle",
            "genesis_go2walking",
            "genesis_go2walkback",
            "genesis_go2hop",
            "polyglot",  # separate harness from the rest
            "imo_grading",
            "imo_proof",
        ],
        required=True,
        help="One or more domains to evaluate (must be from the allowed list)",
    )
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="OpenAI-style reasoning_effort applied to meta + task agent calls",
    )
    parser.add_argument("--iterations_left", type=int, default=0)
    parser.add_argument("--budget_status_path", type=str, default=None)
    parser.add_argument(
        "--eval_samples",
        type=int,
        nargs="+",
        default=None,
        help="Evaluation samples per domain (-1 for all). Provide one value per domain.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=10,
        help="Number of evaluation workers in parallel",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the generation output folder",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Prepared root dir for the generation step",
    )
    parser.add_argument(
        "--root_commit",
        type=str,
        default="main",
        help="Base commit for the prepared root dir",
    )
    parser.add_argument(
        "--current_genid",
        type=str,
        required=True,
        help="Generation id to produce",
    )
    parser.add_argument(
        "--parent_genid",
        type=str,
        default=None,
        help="Explicit parent generation id",
    )
    parser.add_argument(
        "--parent_patch_files",
        type=str,
        nargs="*",
        default=[],
        help="Explicit parent lineage patch files to apply",
    )
    parser.add_argument(
        "--meta_patch_files",
        type=str,
        nargs="+",
        default=[],
        help="Meta patch files to apply",
    )
    parser.add_argument(
        "--reset_task_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the task agent (for self-referential self-improvement transfer experiments)",
    )
    parser.add_argument(
        "--reset_meta_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the meta agent (for self-referential self-improvement transfer experiments)",
    )
    parser.add_argument(
        "--copy_root_dir",
        type=str,
        default=None,
        help="Copy root dir for setup_initial_gen",
    )
    parser.add_argument(
        "--eval_test",
        default=False,
        action="store_true",
        help="Always run test set evaluation",
    )
    parser.add_argument(
        "--skip_staged_eval",
        default=False,
        action="store_true",
        help="Skip staged evaluation",
    )
    parser.add_argument("--skip_meta_agent", default=False, action="store_true")
    parser.add_argument("--skip_eval_after_meta_agent", default=False, action="store_true")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        choices=["train", "val", "test"],
        help="Explicit splits to evaluate (overrides get_domain_splits)",
    )
    args = parser.parse_args()

    # Post-parse validation
    if args.eval_samples is None:
        eval_samples = [-1] * len(args.domains)
    elif len(args.eval_samples) == len(args.domains):
        eval_samples = args.eval_samples
    else:
        parser.error("--eval_samples must be a one per domain if provided")

    eval_subsets = [get_domain_eval_subset(d) for d in args.domains]
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if args.run_id is None
        else args.run_id
    )
    current_genid = (
        int(args.current_genid)
        if args.current_genid.isdigit()
        else args.current_genid
    )
    parent_genid = (
        int(args.parent_genid)
        if args.parent_genid and args.parent_genid.isdigit()
        else args.parent_genid
    )
    if args.root_dir is None:
        root_dir, root_commit = setup_initial_gen(
            args.output_dir,
            args.domains,
            copy_root_dir=args.copy_root_dir,
            subsets=eval_subsets,
            resume=False,
            eval_test=args.eval_test,
        )
    else:
        root_dir, root_commit = args.root_dir, args.root_commit
    meta_patch_files = process_meta_patch_files(
        args.meta_patch_files,
        args.output_dir,
        reset_task_agent=args.reset_task_agent,
        reset_meta_agent=args.reset_meta_agent,
    )
    run_generation_step(
        docker.from_env(),
        domains=args.domains,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        run_id=run_id,
        iterations_left=args.iterations_left,
        output_dir=args.output_dir,
        current_genid=current_genid,
        parent_genid=parent_genid,
        root_dir=root_dir,
        root_commit=root_commit,
        eval_samples=eval_samples,
        eval_workers=args.eval_workers,
        eval_subsets=eval_subsets,
        meta_patch_files=meta_patch_files,
        parent_patch_files=args.parent_patch_files,
        run_meta_agent=not args.skip_meta_agent,
        run_eval_after_meta_agent=not args.skip_eval_after_meta_agent,
        eval_test=args.eval_test,
        skip_staged_eval=args.skip_staged_eval,
        splits=args.splits,
        budget_status_path=args.budget_status_path,
    )

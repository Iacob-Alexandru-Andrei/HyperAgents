# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import logging
import os
import shutil
import shlex
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import docker

from utils.common import file_exist_and_not_empty, load_json_file
from utils.constants import REPO_NAME
from utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
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


def _snapshot_train_lineage_in_container(
    container,
    current_genid,
    base_commit,
    metadata=None,
    repo_name=REPO_NAME,
    verbose=True,
):
    """F2l: snapshot per-gen lineage into ``/<repo_name>/lineage/gen_<id>/``
    and commit. This is the SOLE channel through which a child container
    inherits its parent's artifacts — no host-side tar fallback.

    Captured per gen:
      - ``<domain>_eval/`` (train predictions). Val/test eval dirs are
        excluded because their dirname suffix (``_eval_val`` / ``_eval_test``)
        doesn't match the ``/tmp/*_eval`` glob.
      - ``agent_output/`` (full dir — meta_agent_chat_history.md +
        model_patch.diff + any other artifacts the meta-agent wrote).
      - ``metadata.json`` (stub with fields known at snapshot time:
        current_genid, parent_genid, parent_agent_success, run_eval,
        etc. The host writes the authoritative metadata.json on its
        side after this function returns; this stub is what *child*
        containers see for ancestor lookups).

    A single ``git add -A`` + commit captures the lineage delta AND any
    untracked meta-agent code edits. ``--allow-empty`` makes the call
    idempotent for a node that produced no eval output.

    The richer ``model_patch.diff`` covering code + lineage is emitted by
    the caller via ``git diff --binary <base_commit>`` after this
    function returns.
    """
    quoted_id = shlex.quote(str(current_genid))
    lineage_dir = f"/{repo_name}/lineage/gen_{quoted_id}"
    metadata_payload = json.dumps(metadata or {}, indent=2, default=str)
    metadata_quoted = shlex.quote(metadata_payload)
    cmd = (
        f"set -e ; "
        f"mkdir -p {lineage_dir} ; "
        f"for d in /tmp/*_eval ; do "
        f'  if [ -d "$d" ] ; then '
        f'    cp -r "$d" "{lineage_dir}/" ; '
        f"  fi ; "
        f"done ; "
        f"if [ -d /tmp/agent_output ] ; then "
        f"  cp -r /tmp/agent_output {lineage_dir}/agent_output ; "
        f"fi ; "
        f"printf '%s\\n' {metadata_quoted} > {lineage_dir}/metadata.json ; "
        f"echo F2L_SNAPSHOT_COMPLETE"
    )
    exec_result = container.exec_run(
        cmd=["/bin/sh", "-c", cmd], workdir=f"/{repo_name}"
    )
    log_container_output(exec_result, verbose=verbose)
    if exec_result.exit_code != 0:
        raise RuntimeError(
            f"F2l lineage snapshot copy failed with exit {exec_result.exit_code}"
        )
    # Stage everything (lineage + any meta-agent untracked code edits) and
    # commit. ``--allow-empty`` makes the call idempotent for a node that
    # produced no eval output (avoids a "nothing to commit" exit).
    exec_result = container.exec_run(
        cmd=["/bin/sh", "-c", "git add -A"], workdir=f"/{repo_name}"
    )
    log_container_output(exec_result, verbose=verbose)
    commit_msg = f"F2l: code + train lineage for gen_{current_genid}"
    exec_result = container.exec_run(
        cmd=[
            "/bin/sh",
            "-c",
            "git -c user.name='rqgm' -c user.email='rqgm@local' "
            f"commit --allow-empty -m {shlex.quote(commit_msg)}",
        ],
        workdir=f"/{repo_name}",
    )
    log_container_output(exec_result, verbose=verbose)
    if exec_result.exit_code != 0:
        raise RuntimeError(
            f"F2l lineage commit failed with exit {exec_result.exit_code}"
        )
    # base_commit is captured for clarity/audit but not used inside this
    # helper — the caller supplies it because *they* need the right base
    # to ``git diff`` against. We accept it so the call site reads
    # symmetrically with ``apply_diffs_container``'s return value.
    return base_commit


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
    # F2c (recursive-scientist deviation): when ``splits`` is supplied
    # (a non-None list[str]), iterate over exactly those splits instead
    # of consulting the process-global ``get_domain_splits``. The host
    # extension layer drives the train/val/test split selection through
    # this real parameter so concurrent ``run_generation_step`` calls
    # with different splits never race on a global lookup. Default
    # behaviour (``splits=None``) is unchanged.
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
    budget_status_path=None,
):
    # Per-role model routing: the legacy ``model`` keyword is the uniform
    # fallback (still required by the polyglot harness path). ``meta_agent_model``
    # overrides the meta-agent invocation only; ``task_agent_models`` is a
    # per-domain map for the task-agent eval calls. Either path falls back to
    # ``model`` when its specific entry is missing.
    #
    # ``reasoning_effort`` mirrors the per-role model shape: a uniform
    # ``reasoning_effort`` is applied to both the meta-agent and every task-
    # agent path; ``meta_agent_reasoning_effort`` and
    # ``task_agent_reasoning_efforts`` (per-domain map) override the uniform
    # value for their specific call site. ``None`` means "do not pass the
    # parameter through" -- downstream agent.llm silently drops it for models
    # that do not document support for it.
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
            metadata["prev_patch_files"] += meta_patch_files

        # Apply all lineage diffs
        patch_files = get_patch_files(output_dir, parent_genid) if parent_patch_files is None else parent_patch_files
        metadata["prev_patch_files"] += patch_files
        commit_hash = apply_diffs_container(container, patch_files)

        # F2l Phase 3 (recursive-scientist deviation): define agent_output
        # paths and ensure both ends (host + container) exist
        # unconditionally. The bootstrap eval path (run_meta_agent=False)
        # still needs these so F2l can snapshot its eval results into
        # /<REPO_NAME>/lineage/gen_initial/ and emit a model_patch.diff
        # that descendants apply -- without this lift, the F2l block
        # crashes with UnboundLocalError on gen_initial.
        local_agentoutput_folder = os.path.join(gen_output_dir, "agent_output/")
        container_agentoutput_folder = os.path.join(
            container_output_folder, "agent_output"
        )
        os.makedirs(local_agentoutput_folder, exist_ok=True)
        container.exec_run(
            ["mkdir", "-p", container_agentoutput_folder], workdir="/"
        )

        if run_meta_agent:
            # F2l Phase 3 (recursive-scientist deviation): the parent's
            # train-eval lineage already landed in /<REPO_NAME>/lineage/
            # via the patches applied above. No host-side tar of
            # output_dir is needed -- predictions, agent_output/, and
            # metadata.json all flow via patches. The meta-agent's
            # eval_path simply points at the lineage tree.
            container_prev_eval_path = f"/{REPO_NAME}/lineage"
            container.exec_run(
                ["mkdir", "-p", container_prev_eval_path], workdir="/"
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
            # F2i (recursive-scientist Tier 3 two-axis kill): persist the
            # exit code so the host can distinguish a wall-clock kill
            # (the in-container ``timeout 21600`` shell wrapper exits 124)
            # from a clean-exit ``parent_agent_success=False`` (e.g. the
            # meta-agent gave up on its own). The host writes
            # ``killed_by="time"`` when this is 124.
            metadata["meta_agent_exit_code"] = int(exec_result.exit_code or 0)

            # Copy container outputs to local. ``local_agentoutput_folder``
            # is defined unconditionally above the ``if run_meta_agent``
            # block so the F2l flow can also reach it on the bootstrap
            # eval path.
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

            # F2l (recursive-scientist deviation): snapshot train-eval
            # outputs, agent_output/, and a metadata.json stub into
            # ``/<REPO_NAME>/lineage/gen_<id>/`` and rewrite
            # model_patch.diff so it carries code + lineage as one
            # patch. Child containers that ``patch -p1`` this richer
            # patch get the full parent state in their working tree
            # without any host-side tar-of-output_dir.
            #
            # Phase 3: this is now the LOAD-BEARING path.
            # ``copy_prev_eval_to_container`` has been deleted; the
            # meta-agent's eval_path points at
            # ``/<REPO_NAME>/lineage/`` and reads ancestor data from
            # the per-gen subdirs delivered via patches.
            #
            # On failure of the snapshot itself we log + suppress (it
            # is best-effort) but production flow continues — the host
            # already has the train/val outcomes from the eval block
            # above; what we lose is the child's view of this parent.
            try:
                _snapshot_train_lineage_in_container(
                    container,
                    current_genid=current_genid,
                    base_commit=commit_hash,
                    metadata=metadata,
                    verbose=False,
                )
                # Preserve the existing (pre-eval, code-only) patch under
                # ``code_only_patch.diff`` for analysis before we
                # overwrite the canonical ``model_patch.diff`` with the
                # richer code+lineage diff. The original was already
                # copied to the host by the earlier ``copy_from_container``
                # call above (line ~548-553), so this is a host-side
                # rename.
                existing_model_patch = os.path.join(
                    local_agentoutput_folder, "model_patch.diff"
                )
                code_only_path = os.path.join(
                    local_agentoutput_folder, "code_only_patch.diff"
                )
                if os.path.exists(existing_model_patch) and not os.path.exists(
                    code_only_path
                ):
                    shutil.copy(existing_model_patch, code_only_path)
                # Emit the richer model_patch.diff INSIDE the container,
                # then copy it back over the host's existing file.
                # ``--binary`` so any non-text harness artifact still
                # round-trips through patch -p1. ``--no-color`` to keep
                # the output script-safe.
                in_container_patch = (
                    f"{container_agentoutput_folder}/model_patch.diff"
                )
                cmd = (
                    f"git diff --binary --no-color {shlex.quote(commit_hash)} "
                    f"> {shlex.quote(in_container_patch)}"
                )
                exec_result = container.exec_run(
                    cmd=["/bin/sh", "-c", cmd], workdir=f"/{REPO_NAME}"
                )
                log_container_output(exec_result, verbose=False)
                if exec_result.exit_code != 0:
                    raise RuntimeError(
                        f"F2l git diff failed with exit {exec_result.exit_code}"
                    )
                copy_from_container(
                    container,
                    source_path=in_container_patch,
                    dest_path=existing_model_patch,
                )
            except Exception as lineage_exc:  # noqa: BLE001 -- F2l is additive
                safe_log(
                    f"F2l lineage snapshot skipped: {lineage_exc}",
                    level=logging.WARNING,
                )

            # F2l Phase 3: register the (possibly F2l-regenerated)
            # ``model_patch.diff`` in ``curr_patch_files`` when it is
            # genuinely present and non-empty. This is the canonical
            # place for the bootstrap eval path (run_meta_agent=False)
            # to publish its F2l-generated patch -- the meta-agent
            # branch above already appended the path eagerly. Idempotent:
            # we don't duplicate if the path is already in the list.
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
        help="F2c: explicit splits to evaluate (overrides get_domain_splits)",
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

# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def __init__(
        self,
        model,
        chat_history_file='./outputs/chat_history.md',
        reasoning_effort=None,
        budget_status_path=None,
    ):
        super().__init__(model=model, chat_history_file=chat_history_file)
        self.reasoning_effort = reasoning_effort
        self.budget_status_path = budget_status_path

    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        # F-class: load the model catalog written by the host before the
        # container started (write_catalog_artifacts at run init). When the
        # catalog is present, query_model is enabled with this list bound
        # via functools.partial inside load_tools; when absent the tool
        # advertises itself as disabled and refuses to fire.
        catalog_path = os.path.join(eval_path, "model_catalog.json")
        catalog = []
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path) as f:
                    catalog = json.load(f) or []
            except (OSError, json.JSONDecodeError):
                catalog = []

        instruction = (
            f"Modify any part of the codebase at `{repo_path}` to improve "
            "performance on the active evaluation domains. This includes your "
            "OWN code -- the meta-agent's loop, prompts, tool surface, and the "
            "task-agent harness. Improving your own self-improvement abilities "
            "compounds across future generations and may lead to better "
            "outcomes in the long run than only optimizing the per-domain "
            "task agents."
            f"\n\nYou have {iterations_left} expansions remaining."
            f"\n\nAvailable LLM models for one-shot delegation are listed in "
            f"`{eval_path}/MODEL_CATALOG.md`. Use the `query_model` tool to "
            "fire a one-shot call to a different model with `(prompt, id, "
            "effort)` -- useful for offloading bulk text generation to a "
            "cheaper model, or sanity-checking your reasoning against a "
            "different model. The call has no chat history and no tools; "
            "it's tokens in, tokens out."
            f"\n\nRunning LLM cost from prior generations is summarized at "
            f"`{eval_path}/cost.md`. Live remaining budget for this "
            "expansion is prepended to every chat turn by the host harness."
            f"\n\nPrior generation artifacts are at `{eval_path}`. Each "
            "ancestor `gen_<id>/` contains:"
            "\n  - `<domain>_eval/predictions.csv` -- task-agent outputs "
            "paired with ground-truth labels on the TRAIN split. The only "
            "sample-level outcome data you can see directly."
            "\n  - `agent_output/meta_agent_chat_history.md` -- the full "
            "transcript of that predecessor's reasoning. The highest-signal "
            "artifact: what was tried, what worked, what failed. Future "
            "generations will read YOUR chat history, so explain your "
            "reasoning clearly."
            "\n  - `agent_output/model_patch.diff` -- the diff each "
            "predecessor produced. Useful so you don't redo work and can "
            "build on it."
            "\n  - `metadata.json` -- parent_genid, parent_agent_success, "
            "lineage info."
            "\n\nCRITICAL: your utility is computed on a HELD-OUT validation "
            "set you NEVER see. The train predictions you can read are "
            "feedback for study only. Do NOT hardcode answers, memorize "
            "specific question ids, or otherwise overfit to the train "
            "samples -- those tricks score perfectly on train and 0 on "
            "validation, which is what actually drives selection."
            "\n\nOutcomes are binary (1=correct, 0=incorrect) and a node's "
            "utility is the mean of its validation outcomes per (role, "
            "task). The evaluator is external -- improving your agents to "
            "genuinely produce better outputs is the path forward; gaming "
            "the parser or breaking the harness scores 0."
            "\n\nEach `<domain>_eval/report.json` distinguishes `malformed_qids` (LLM "
            "output couldn't be parsed -> outcome=0 by convention; the agent failed to "
            "produce parseable output) from `question_ids_failed_real` (parsed but "
            "wrong prediction). High `malformed_count` is an actionable signal: add "
            "output-validation scaffolding to the relevant agent (wrap its "
            "`chat_with_agent` call in a retry-with-corrective-prompt loop, tighten "
            "the output schema in `task_agent.py:OUTPUT_FORMATS`, or use `query_model` "
            "to sanity-check suspect outputs). Reducing malformations is a reliable "
            "way to lift utility because each malformed row is currently a "
            "guaranteed outcome=0."
        )

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            reasoning_effort=self.reasoning_effort,
            catalog=catalog,
            budget_status_path=self.budget_status_path,
            workspace_root=repo_path,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        instruction = f"Modify any part of the codebase at `{repo_path}`."
        # F2g (recursive-scientist Tier-2 cost proxy): when the host has
        # spawned a per-expansion proxy and exported the budget via
        # ``RS_COST_PROXY_BUDGET_USD``, surface that to the meta-agent
        # so it can self-regulate. Absent var = original behavior.
        budget_str = os.environ.get("RS_COST_PROXY_BUDGET_USD")
        if budget_str:
            instruction = (
                f"{instruction}\n\n"
                f"Your budget for this expansion is ${budget_str}. "
                f"The proxy will refuse further LLM calls when the budget is exceeded; "
                f"check `{eval_path}/cost.md` to see your running spend."
            )

        new_msg_history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all')

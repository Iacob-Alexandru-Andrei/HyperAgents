# Minimal IMO-proof TaskAgent. Heavy multi-round doctrine prompts
# (step1_prompt + self_improvement + verify + refine) have moved to
# ``src/recursive_scientist/baselines/prompt_registry.py::PROVER_BASELINE``;
# they activate only when the rs supervisor passes
# ``system_prompt_override=<PROVER_BASELINE text>`` to the constructor.
# Default behavior is one minimal LLM call.

from typing import Any, Dict, List, Optional, Tuple

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


DEFAULT_PROMPT = (
    "Prove the following IMO problem rigorously. Output your final proof inside "
    "<SOLUTION>...</SOLUTION> tags."
)


class TaskAgent(AgentSystem):
    """Minimal IMO solver. One LLM call.

    Inputs expected in ``forward()``:
        inputs["problem"]: str (required) - the problem statement

    The optional ``system_prompt_override`` constructor kwarg (forwarded by
    ``hyperagents/domains/harness.py``) replaces ``DEFAULT_PROMPT``; the
    rs supervisor sets it from ``roles.prover.prompt_override`` via the
    prompt registry. Reasoning-effort and budget-status plumbing match
    the canonical ``hyperagents/task_agent.py``.
    """

    def __init__(
        self,
        model: str,
        chat_history_file: str = "./outputs/chat_history.md",
        reasoning_effort: Optional[str] = None,
        budget_status_path: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, chat_history_file=chat_history_file)
        self.reasoning_effort = reasoning_effort
        self.budget_status_path = budget_status_path
        self.system_prompt_override = system_prompt_override

    def forward(
        self,
        inputs: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        problem_statement: str = inputs["problem"]
        system_prompt = self.system_prompt_override or DEFAULT_PROMPT
        instruction = (
            f"{system_prompt}\n\n"
            f"### Problem ###\n\n{problem_statement}\n\n"
            "Output your final proof between <SOLUTION> and </SOLUTION>."
        )

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available=[],
            reasoning_effort=self.reasoning_effort,
            budget_status_path=self.budget_status_path,
        )

        try:
            prediction = new_msg_history[-1].get("text", "")
        except (IndexError, AttributeError) as e:
            self.log(f"Error extracting LLM output: {e}")
            prediction = ""

        return prediction, new_msg_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to use")
    args = parser.parse_args()

    agent = TaskAgent(model=args.model)

    import pandas as pd

    csv_path = "./domains/imo/proofbench.csv"
    df = pd.read_csv(csv_path)

    row = df.iloc[0]
    print(row)

    from domains.imo.proof_utils import GROUND_TRUTH_KEY, format_input_dict

    input_dict = format_input_dict(row)
    prediction, new_msg_history = agent.forward(input_dict)
    print(f"Prediction: {repr(prediction[:500])}...")

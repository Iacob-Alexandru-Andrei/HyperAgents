# Minimal AI-reviewer TaskAgent. The heavy sakana NeurIPS reviewer doctrine
# (reviewer_system_prompt_base + template_instructions + neurips_form) has
# moved to ``src/recursive_scientist/baselines/prompt_registry.py``
# (``SAKANA_NEURIPS_REVIEWER``); it activates only when the rs supervisor
# passes ``system_prompt_override=<that text>`` via the registry.
# Default behavior is one minimal LLM call.

from typing import Any, Dict, List, Optional, Tuple

from agent.base_agent import AgentSystem
from agent.llm import get_response_from_llm
from utils.common import extract_jsons


DEFAULT_PROMPT = (
    "You are reviewing a paper submitted to a top ML venue. "
    "Decide Accept or Reject and give a short rationale. "
    "Respond with one JSON block of the form "
    '{"Decision": "Accept" | "Reject", "Rationale": "<short rationale>"}.'
)


class TaskAgent(AgentSystem):
    """Minimal paper reviewer. One LLM call.

    Inputs expected in ``forward()``:
        inputs["paper_text"]: str - the paper to review

    The optional ``system_prompt_override`` constructor kwarg (forwarded by
    ``hyperagents/domains/harness.py``) replaces ``DEFAULT_PROMPT``; the
    rs supervisor sets it from ``roles.critic.prompt_override`` /
    ``roles.paper_writer.prompt_override`` via the prompt registry
    (e.g. ``sakana_neurips_reviewer``, ``critic_baseline``).
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

    def forward(self, inputs: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        system_prompt = self.system_prompt_override or DEFAULT_PROMPT
        instruction = (
            f"{system_prompt}\n\n"
            "Here is the paper you are asked to review:\n"
            "```\n"
            f"{inputs['paper_text']}\n"
            "```"
        )

        self.log(f"Input: {repr(instruction)}")
        response, new_msg_history, _ = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )
        self.log(f"Output: {repr(response)}")

        prediction = "None"
        try:
            extracted_jsons = extract_jsons(new_msg_history[-1]["text"])
            prediction = extracted_jsons[-1]["Decision"]
        except (KeyError, IndexError, TypeError) as e:
            self.log(f"Error extracting prediction: {e}")
            prediction = "None"

        return prediction, new_msg_history

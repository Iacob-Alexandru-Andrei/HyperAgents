from agent.base_agent import AgentSystem
from agent.llm import get_response_from_llm
from baselines.ai_reviewer.agent import neurips_form, reviewer_system_prompt_base
from domains.paper_review.prompt_packet import build_review_packet_toon
from utils.common import extract_jsons


class TaskAgent(AgentSystem):
    def forward(self, inputs):
        packet_toon = build_review_packet_toon(inputs["paper_text"])
        instruction = reviewer_system_prompt_base + neurips_form
        instruction += f"""
Here is a compact TOON review packet derived from the paper.
Use the retained title, abstract, core sections, and key tables as the paper evidence.
Evaluate the evidence on its merits; do not infer a rejection solely from cost-reduced formatting.

paper_packet_toon:
```toon
{packet_toon}
```"""

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
        except Exception as e:
            self.log(f"Error extracting prediction: {e}")
            prediction = "None"

        return prediction, new_msg_history

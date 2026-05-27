from agent.base_agent import AgentSystem
from agent.llm import get_response_from_llm
from baselines.ai_reviewer.agent import neurips_form, reviewer_system_prompt_neg
from domains.paper_review.prompt_packet import build_core_text_packet_parts
from utils.common import extract_jsons


class TaskAgent(AgentSystem):
    def forward(self, inputs):
        packet_parts = build_core_text_packet_parts(inputs["paper_text"])
        instruction = reviewer_system_prompt_neg + neurips_form
        instruction += f"""
Here is a cost-reduced paper view.
The TOON sidecar contains metadata, included section names, dropped section names, and key tables.
The raw core text preserves selected title, abstract, method, experiment, result, discussion, limitation, and conclusion prose.

paper_metadata_toon:
```toon
{packet_parts["metadata_toon"]}
```

paper_core_text:
```
{packet_parts["core_text"]}
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

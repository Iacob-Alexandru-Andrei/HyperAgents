from agent.base_agent import AgentSystem
from agent.llm import get_response_from_llm
from baselines.ai_reviewer.agent import neurips_form, reviewer_system_prompt_base
from domains.paper_review.prompt_packet import build_retrieval_packet_toon
from utils.common import extract_jsons


EVIDENCE_INSTRUCTIONS = """
Extract concise decision-relevant review evidence from this paper packet.
Use only the retained abstract, evidence chunks, and key tables.
Prefer specific claims about novelty, soundness, results, limitations, and missing support.
Do not make a final accept/reject decision.

Respond as JSON with these fields:
- "Summary"
- "Novelty Evidence"
- "Soundness Evidence"
- "Results Evidence"
- "Limitations Evidence"
- "Missing Evidence"
- "Decision-Relevant Evidence"
"""


class TaskAgent(AgentSystem):
    def forward(self, inputs):
        packet_toon = build_retrieval_packet_toon(inputs["paper_text"])
        evidence_instruction = f"""{EVIDENCE_INSTRUCTIONS}

evidence_packet_toon:
```toon
{packet_toon}
```"""

        self.log(f"Evidence Input: {repr(evidence_instruction)}")
        evidence_response, evidence_history, _ = get_response_from_llm(
            msg=evidence_instruction,
            model=self.model,
            msg_history=[],
        )
        self.log(f"Evidence Output: {repr(evidence_response)}")

        review_instruction = reviewer_system_prompt_base + neurips_form
        review_instruction += f"""
Here is extracted evidence from a cost-reduced paper packet.
Use this evidence to complete the review. Treat explicit missing-evidence notes as uncertainty, but do not reject solely because the packet was cost-reduced.

extracted_evidence:
```
{evidence_response}
```"""

        self.log(f"Review Input: {repr(review_instruction)}")
        response, review_history, _ = get_response_from_llm(
            msg=review_instruction,
            model=self.model,
            msg_history=[],
        )
        self.log(f"Review Output: {repr(response)}")

        prediction = "None"
        try:
            extracted_jsons = extract_jsons(review_history[-1]["text"])
            prediction = extracted_jsons[-1]["Decision"]
        except Exception as e:
            self.log(f"Error extracting prediction: {e}")
            prediction = "None"

        return prediction, evidence_history + review_history

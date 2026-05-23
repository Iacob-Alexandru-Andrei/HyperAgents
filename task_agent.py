from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons


class TaskAgent(AgentSystem):
    def __init__(
        self,
        model,
        chat_history_file='./outputs/chat_history.md',
        reasoning_effort=None,
        budget_status_path=None,
        system_prompt_override=None,
    ):
        super().__init__(model=model, chat_history_file=chat_history_file)
        self.reasoning_effort = reasoning_effort
        self.budget_status_path = budget_status_path
        self.system_prompt_override = system_prompt_override

    MAX_PARSE_RETRIES = 1

    def forward(self, inputs):
        """
        An agent that solves a given task.

        Args:
            inputs (dict): A dictionary with input data for the task.

        Returns:
            tuple:
                - prediction (str): The prediction made by the agent.
                - new_msg_history (list): A list of messages representing the message history of the interaction.
        """
        domain = inputs['domain']

        output_format, extract_field = self.OUTPUT_FORMATS.get(domain, ('Respond in JSON format with the following schema:\n<json>\n{\n    "response": ...\n}\n</json>', "response"))
        # ``system_prompt_override``: when set (via recursive_scientist's
        # ``RoleConfig.prompt_override`` registry), the role-specific baseline
        # prompt replaces the generic "You are an agent." preamble. The
        # per-domain ``output_format`` (which carries the parseable JSON
        # contract the harness relies on) is preserved after it -- baselines
        # like ``sakana_neurips_reviewer`` carry their own format block, but
        # appending the harness's contract still guarantees the JSON shape
        # downstream extraction looks for. ``extract_field`` is unchanged so
        # extraction logic stays in sync with ``OUTPUT_FORMATS``.
        if self.system_prompt_override:
            instruction = f"""{self.system_prompt_override}

Task input:
```
{inputs}
```

{output_format}"""
        else:
            instruction = f"""You are an agent.

Task input:
```
{inputs}
```

{output_format}"""

        # Bounded retry on parse failure: up to MAX_PARSE_RETRIES calls total;
        # each retry sends a corrective re-prompt that restates the required
        # format. History accumulates across attempts so the model sees its own
        # bad output. On exhaustion, fall through to the sentinel ``"None"``
        # that downstream eval code already handles.
        new_msg_history = []
        current_instruction = instruction
        prediction = "None"
        for attempt in range(self.MAX_PARSE_RETRIES):
            try:
                new_msg_history = chat_with_agent(
                    current_instruction,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    reasoning_effort=self.reasoning_effort,
                    budget_status_path=self.budget_status_path,
                )
            except RuntimeError as e:
                if "truncated response" not in str(e):
                    raise
                self.log(f"Truncated response treated as malformed output: {e}")
                return prediction, new_msg_history
            try:
                extracted_jsons = extract_jsons(new_msg_history[-1]['text'])
            except Exception as e:
                self.log(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                extracted_jsons = None
            # Prefer the LAST JSON object that contains the expected field —
            # for chain-of-thought replies, the final JSON is the answer.
            if extracted_jsons:
                for obj in reversed(extracted_jsons):
                    if isinstance(obj, dict) and extract_field in obj:
                        prediction = obj[extract_field]
                        return prediction, new_msg_history
            current_instruction = (
                'Your previous reply did not contain a parseable JSON object '
                f'with the required field "{extract_field}". '
                'Reply with a single JSON object and nothing else.\n\n'
                f'{output_format}'
            )

        return prediction, new_msg_history


# Per-domain output-format dispatch. Anything not listed here falls through
# to the default in ``TaskAgent.forward`` (response-shaped JSON).
TaskAgent.OUTPUT_FORMATS = {
    "paper_review": (
        'You are reviewing the paper above for a top ML venue. Read it carefully and decide '
        'whether to Accept or Reject it.\n\n'
        'Respond with one JSON block:\n'
        '{\n'
        '  "Summary": "...", "Strengths": [...], "Weaknesses": [...],\n'
        '  "Decision": "Accept" | "Reject"\n'
        '}\n'
        'For "Decision", use only "Accept" or "Reject".',
        "Decision",
    ),
    "paper_writer_review": (
        'Continue the paper above.\n\n'
        'Respond with one JSON block:\n'
        '{\n'
        '  "response": "..."\n'
        '}',
        "response",
    ),
    "imo_grading": (
        'You are grading an IMO solution. Use the problem, reference solution, grading '
        'guidelines, and proposed solution to assign exactly one label.\n\n'
        'Respond with one JSON block:\n'
        '{\n'
        '  "Grade": "incorrect" | "partial" | "almost" | "correct"\n'
        '}\n'
        'For "Grade", use only "incorrect", "partial", "almost", or "correct".',
        "Grade",
    ),
}

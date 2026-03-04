"""
prompt_utils.py

Utilities for building prompts used by VLA models.
"""


def build_vla_prompt(task_label: str, legacy: bool) -> str:
    """Construct the prompt string used for action prediction."""
    if legacy:
        return f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Match PurePromptBuilder formatting without importing heavy dependencies.
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut: "
    prompt = f"{prompt} </s>"
    # Training drops the terminal EOS; keep the trailing space before it.
    if prompt.endswith("</s>"):
        prompt = prompt[: -len("</s>")]
    return prompt


# Backwards-compatible alias for internal usage
_build_vla_prompt = build_vla_prompt

__all__ = [
    "LLMBackbone",
    "LLaMa2LLMBackbone",
    "MistralLLMBackbone",
    "PhiLLMBackbone",
]


def __getattr__(name):
    if name == "LLMBackbone":
        from .base_llm import LLMBackbone

        return LLMBackbone
    if name == "LLaMa2LLMBackbone":
        from .llama2 import LLaMa2LLMBackbone

        return LLaMa2LLMBackbone
    if name == "MistralLLMBackbone":
        from .mistral import MistralLLMBackbone

        return MistralLLMBackbone
    if name == "PhiLLMBackbone":
        from .phi import PhiLLMBackbone

        return PhiLLMBackbone
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)

"""
Model configuration table - maintains context lengths for major LLM providers.

Since most API providers don't return context length in models.retrieve() or models.list(),
we need to maintain this configuration table manually.

Data sources:
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/overview
- DeepSeek: https://api-docs.deepseek.com/
- Google Gemini: https://ai.google.dev/gemini-api/docs/models
- Others: Official documentation from each provider

Last updated: 2025-12-22
"""

MODEL_CONTEXT_LENGTHS = {
    # OpenAI Models
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    "o1": 128_000,
    "o1-mini": 128_000,
    "o3-mini": 128_000,
    "o3-pro": 200_000,
    "o4-mini": 200_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-5": 400_000,
    "gpt-5.2-xhigh": 400_000,  # Model from Codex

    # Anthropic Claude Models
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-sonnet-4": 200_000,  # Default, expandable to 1M
    "claude-sonnet-4.5": 200_000,  # Default, expandable to 1M
    "claude-opus-4": 200_000,
    "claude-opus-4.5": 200_000,

    # DeepSeek Models
    "deepseek-chat": 64_000,
    "deepseek-coder": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v3": 64_000,
    "deepseek-r1": 64_000,
    "deepseek-r1-0528": 64_000,
    "deepseek-v3.1": 128_000,
    "deepseek-v3.2-exp": 128_000,

    # Google Gemini Models
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-flash-8b": 1_000_000,
    "gemini-2.0-flash-exp": 1_000_000,
    "gemini-3-flash-preview": 1_000_000,  # Assumed similar to 2.0-flash
    "gemini-pro": 32_768,
    "gemini-pro-vision": 16_384,

    # Meta Llama Models
    "llama-3.3-70b": 128_000,
    "llama-3.2-90b": 128_000,
    "llama-3.1-405b": 128_000,
    "llama-3.1-70b": 128_000,
    "llama-3.1-8b": 128_000,
    "llama-3-70b": 8_000,
    "llama-3-8b": 8_000,
    "llama-2-70b": 4_096,
    "llama-2-13b": 4_096,
    "llama-2-7b": 4_096,

    # Mistral Models
    "mistral-large": 128_000,
    "mistral-medium": 32_000,
    "mistral-small": 32_000,
    "mistral-7b": 32_000,
    "mixtral-8x7b": 32_000,
    "mixtral-8x22b": 64_000,

    # Qwen Models
    "qwen-turbo": 32_000,
    "qwen-plus": 32_000,
    "qwen-max": 32_000,
    "qwen2.5-72b": 128_000,
    "qwen2.5-coder": 128_000,

    # Other common models
    "yi-large": 32_000,
    "yi-medium": 16_000,
    "moonshot-v1": 128_000,
    "glm-4": 128_000,
    "glm-3-turbo": 128_000,
}


def get_model_context_length(model_name: str) -> int | None:
    """
    Get context length by model name.

    Args:
        model_name: Model name

    Returns:
        Context length (tokens), or None if not found
    """
    # Exact match
    if model_name in MODEL_CONTEXT_LENGTHS:
        return MODEL_CONTEXT_LENGTHS[model_name]

    # Fuzzy match (handle dated model versions like "gpt-4o-2024-05-13")
    model_base = model_name.split("-")[0:3]  # Take first 3 parts
    for key in MODEL_CONTEXT_LENGTHS:
        if key.startswith("-".join(model_base)):
            return MODEL_CONTEXT_LENGTHS[key]

    # Partial match (for custom deployed model names)
    model_lower = model_name.lower()
    for key, value in MODEL_CONTEXT_LENGTHS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return value

    return None


def list_supported_models() -> list[str]:
    """Return list of all supported models."""
    return sorted(MODEL_CONTEXT_LENGTHS.keys())


def get_provider_models(provider: str) -> dict[str, int]:
    """
    Filter models by provider.

    Args:
        provider: Provider name (openai, anthropic, deepseek, gemini, llama, mistral, qwen)

    Returns:
        Dictionary of models for that provider
    """
    provider_prefixes = {
        "openai": ["gpt-", "o1", "o3", "o4"],
        "anthropic": ["claude-"],
        "deepseek": ["deepseek-"],
        "gemini": ["gemini-"],
        "llama": ["llama-"],
        "mistral": ["mistral-", "mixtral-"],
        "qwen": ["qwen"],
    }

    prefixes = provider_prefixes.get(provider.lower(), [])
    if not prefixes:
        return {}

    return {
        model: length
        for model, length in MODEL_CONTEXT_LENGTHS.items()
        if any(model.startswith(prefix) for prefix in prefixes)
    }

"""
模型配置表 - 维护各大 LLM 提供商的模型上下文长度

由于大多数 API 提供商不在 models.retrieve() 或 models.list() 中返回上下文长度，
我们需要手动维护这个配置表。

数据来源：
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/overview
- DeepSeek: https://api-docs.deepseek.com/
- Google Gemini: https://ai.google.dev/gemini-api/docs/models
- 其他：各提供商官方文档

最后更新：2025-12-22
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
    "gpt-5.2-xhigh": 400_000,  # Codex 提供的模型

    # Anthropic Claude Models
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-sonnet-4": 200_000,  # 默认，可扩展至 1M
    "claude-sonnet-4.5": 200_000,  # 默认，可扩展至 1M
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
    "gemini-3-flash-preview": 1_000_000,  # 假设与 2.0-flash 类似
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

    # 其他常见模型
    "yi-large": 32_000,
    "yi-medium": 16_000,
    "moonshot-v1": 128_000,
    "glm-4": 128_000,
    "glm-3-turbo": 128_000,
}


def get_model_context_length(model_name: str) -> int | None:
    """
    根据模型名称获取上下文长度

    Args:
        model_name: 模型名称

    Returns:
        上下文长度（tokens），如果未找到则返回 None
    """
    # 精确匹配
    if model_name in MODEL_CONTEXT_LENGTHS:
        return MODEL_CONTEXT_LENGTHS[model_name]

    # 模糊匹配（处理带日期的模型版本，如 "gpt-4o-2024-05-13"）
    model_base = model_name.split("-")[0:3]  # 取前3个部分
    for key in MODEL_CONTEXT_LENGTHS:
        if key.startswith("-".join(model_base)):
            return MODEL_CONTEXT_LENGTHS[key]

    # 部分匹配（用于处理自定义部署的模型名）
    model_lower = model_name.lower()
    for key, value in MODEL_CONTEXT_LENGTHS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return value

    return None


def list_supported_models() -> list[str]:
    """返回所有支持的模型列表"""
    return sorted(MODEL_CONTEXT_LENGTHS.keys())


def get_provider_models(provider: str) -> dict[str, int]:
    """
    根据提供商筛选模型

    Args:
        provider: 提供商名称 (openai, anthropic, deepseek, gemini, llama, mistral, qwen)

    Returns:
        该提供商的模型字典
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

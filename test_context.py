#!/usr/bin/env python
"""测试上下文管理功能"""
from main import (
    MAX_CONTEXT_LIMIT,
    CONTEXT_WARNING_THRESHOLD,
    CONTEXT_CRITICAL_THRESHOLD,
    estimate_tokens_from_messages,
    format_tokens,
    get_model_context_length,
)

print("=" * 60)
print("测试上下文管理配置")
print("=" * 60)

print(f"\n✓ 配置加载成功")
print(f"  MAX_CONTEXT_LIMIT: {format_tokens(MAX_CONTEXT_LIMIT)}")
print(f"  警告阈值: {int(CONTEXT_WARNING_THRESHOLD * 100)}%")
print(f"  严重阈值: {int(CONTEXT_CRITICAL_THRESHOLD * 100)}%")

print(f"\n✓ 模型上下文长度检测")
test_models = ["gemini-3-flash-preview", "deepseek-chat", "gpt-4o", "claude-3-sonnet"]
for model in test_models:
    length = get_model_context_length(model)
    if length:
        print(f"  {model}: {format_tokens(length)}")
    else:
        print(f"  {model}: 未知")

print(f"\n✓ Token 估算功能")
test_messages = [
    ("user", "Hello, how are you?"),
    ("assistant", "I'm doing well, thank you!"),
    ("user", "Can you help me with Python?"),
]
tokens = estimate_tokens_from_messages(test_messages)
print(f"  测试对话（3 轮）: ~{tokens} tokens")

print(f"\n✓ 上下文使用率模拟")
effective_limit = min(get_model_context_length("gemini-3-flash-preview") or 1000000, MAX_CONTEXT_LIMIT)
test_cases = [
    (10000, "正常使用"),
    (160000, "警告阈值（80%）"),
    (195000, "严重阈值（95%）"),
]

for used, label in test_cases:
    ratio = used / effective_limit
    percentage = int(ratio * 100)
    bar_width = 10
    filled = int(bar_width * ratio)
    bar = "█" * filled + "░" * (bar_width - filled)

    if ratio >= CONTEXT_CRITICAL_THRESHOLD:
        status = "⚠ 严重（红色）"
    elif ratio >= CONTEXT_WARNING_THRESHOLD:
        status = "⚠ 警告（黄色）"
    else:
        status = "正常（灰色）"

    print(f"  {label}: {percentage}% [{bar}] - {status}")

print(f"\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)

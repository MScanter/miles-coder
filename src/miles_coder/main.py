from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from miles_coder.tools import tools
import os
import asyncio
import math
from openai import OpenAI
from miles_coder.model_config import get_model_context_length
from miles_coder.config import get_api_config, is_configured, setup_config

load_dotenv()

console = Console()

API_KEY, BASE_URL, MODEL = get_api_config()
VERSION = "0.2"
CWD = os.path.basename(os.getcwd())

# 上下文管理配置
MAX_CONTEXT_LIMIT = int(os.getenv("MAX_CONTEXT_LIMIT", "200000"))  # 默认限制 200k tokens
CONTEXT_WARNING_THRESHOLD = 0.8  # 使用超过 80% 时警告
CONTEXT_CRITICAL_THRESHOLD = 0.95  # 使用超过 95% 时严重警告
SUMMARY_MESSAGE_CHAR_LIMIT = 800
SUMMARY_CHUNK_TOKEN_LIMIT = 2500

llm: ChatOpenAI | None = None
summary_llm: ChatOpenAI | None = None
agent = None


def init_llm():
    global llm, summary_llm, agent, API_KEY, BASE_URL, MODEL
    API_KEY, BASE_URL, MODEL = get_api_config()
    llm = ChatOpenAI(
        model=MODEL,
        streaming=True,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    summary_llm = ChatOpenAI(
        model=MODEL,
        streaming=False,
        temperature=0,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    agent = create_agent(llm, tools)


chat_messages: list[object] = []

MODEL_CONTEXT_TOKENS: int | None = None
MODEL_CONTEXT_SOURCE = "uninitialized"

# ASCII Art Logo
LOGO = """    ███╗   ███╗
    ████╗ ████║
    ██╔████╔██║
    ██║╚██╔╝██║
    ██║ ╚═╝ ██║
    ╚═╝     ╚═╝"""

def show_welcome():
    content = f"""[bold white]      Welcome![/bold white]

[bold cyan]{LOGO}[/bold cyan]

  [dim]{MODEL} · [cyan]~/{CWD}[/cyan][/dim]
  [dim]输入 [bold yellow]/help[/bold yellow] 查看命令 · [bold yellow]exit[/bold yellow] 退出[/dim]"""

    panel = Panel(
        content,
        title=f"[bold orange1]Miles Coder[/bold orange1] [dim]v{VERSION}[/dim]",
        border_style="orange1",
        padding=(1, 2),
    )
    console.print(panel)


def find_context_length(payload: object, depth: int = 0) -> int | None:
    if payload is None or depth > 4:
        return None
    if isinstance(payload, dict):
        keys = (
            "context_length",
            "max_context_tokens",
            "max_input_tokens",
            "context_window",
            "context_window_size",
            "n_ctx",
        )
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        for value in payload.values():
            if isinstance(value, (dict, list)):
                found = find_context_length(value, depth + 1)
                if found:
                    return found
        return None
    if isinstance(payload, list):
        for item in payload:
            found = find_context_length(item, depth + 1)
            if found:
                return found
        return None
    for attr in ("model_dump", "dict", "to_dict"):
        if hasattr(payload, attr):
            try:
                data = getattr(payload, attr)()
            except TypeError:
                continue
            return find_context_length(data, depth + 1)
    return None


def resolve_model_context_tokens() -> tuple[int | None, str]:
    if not API_KEY:
        tokens = get_model_context_length(MODEL)
        return tokens, "config" if tokens else "unknown"

    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception:
        # 客户端创建失败时，尝试从配置表获取
        tokens = get_model_context_length(MODEL)
        return tokens, "config" if tokens else "unknown"

    model_info = None
    try:
        model_info = client.models.retrieve(MODEL)
    except Exception:
        model_info = None

    tokens = find_context_length(model_info)

    if not tokens:
        try:
            models = client.models.list()
            data = getattr(models, "data", None)
            if data:
                for item in data:
                    item_id = getattr(item, "id", None)
                    if item_id is None and isinstance(item, dict):
                        item_id = item.get("id")
                    if item_id == MODEL:
                        tokens = find_context_length(item)
                        break
        except Exception:
            tokens = None

    # 如果 API 无法获取上下文长度，使用配置表作为备选方案
    if not tokens:
        tokens = get_model_context_length(MODEL)
        return tokens, "config" if tokens else "unknown"

    return tokens, "api" if tokens else "unknown"


def get_model_context_tokens() -> int | None:
    global MODEL_CONTEXT_TOKENS, MODEL_CONTEXT_SOURCE
    if MODEL_CONTEXT_SOURCE == "uninitialized":
        MODEL_CONTEXT_TOKENS, MODEL_CONTEXT_SOURCE = resolve_model_context_tokens()
    return MODEL_CONTEXT_TOKENS


def format_tokens(value: int) -> str:
    if value >= 100_000:
        return f"{value / 1000:.0f}k"
    if value >= 1_000:
        return f"{value / 1000:.1f}k"
    return str(value)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return llm.get_num_tokens(text)
    except Exception:
        ascii_chars = sum(1 for ch in text if ch.isascii())
        non_ascii_chars = len(text) - ascii_chars
        return max(1, math.ceil(ascii_chars / 4) + non_ascii_chars)


def _stringify_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return str(content)


def _normalize_message(message: object) -> tuple[str, str]:
    if isinstance(message, tuple) and len(message) == 2:
        role, content = message
        return str(role), _stringify_message_content(content)
    if isinstance(message, dict):
        role = message.get("role") or message.get("type") or "unknown"
        content = message.get("content", "")
        return str(role), _stringify_message_content(content)
    role = getattr(message, "type", None)
    if role is not None and hasattr(message, "content"):
        return str(role), _stringify_message_content(getattr(message, "content"))
    return "unknown", _stringify_message_content(message)


def _is_user_role(role: str) -> bool:
    return role in ("user", "human")


def _is_assistant_role(role: str) -> bool:
    return role in ("assistant", "ai")


def _display_role(role: str) -> str:
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return role


def _is_summary_system_message(role: str, content: str) -> bool:
    return role == "system" and content.startswith("[已压缩")


def _get_message_name(message: object) -> str | None:
    if isinstance(message, dict):
        name = message.get("name")
        return name if isinstance(name, str) and name else None
    name = getattr(message, "name", None)
    return name if isinstance(name, str) and name else None


def _format_message_for_summary(message: object) -> str:
    role, content = _normalize_message(message)
    content = content.strip()
    if not content:
        return ""
    if len(content) > SUMMARY_MESSAGE_CHAR_LIMIT:
        content = content[:SUMMARY_MESSAGE_CHAR_LIMIT] + "..."
    label = _display_role(role)
    name = _get_message_name(message)
    if role == "tool" and name:
        label = f"tool:{name}"
    return f"{label}: {content}"


def _chunk_lines(lines: list[str], max_tokens: int) -> list[str]:
    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = estimate_tokens(line)
        if current_lines and current_tokens + line_tokens > max_tokens:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_tokens = line_tokens
        else:
            current_lines.append(line)
            current_tokens += line_tokens
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def _summarize_text(text: str, system_prompt: str) -> str:
    if not text.strip():
        return ""
    try:
        response = summary_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=text)]
        )
    except Exception:
        return ""
    content = getattr(response, "content", "")
    summary = _stringify_message_content(content).strip()
    return summary


def _generate_structured_summary(messages: list[object]) -> str:
    lines = []
    for msg in messages:
        line = _format_message_for_summary(msg)
        if line:
            lines.append(line)
    if not lines:
        return ""

    chunk_prompt = (
        "你是对话总结助手。将下面对话片段提炼为要点，保留事实、决策、约束、"
        "问题、文件/命令。不要推测。输出不超过 6 条要点，每条一行，以 \"- \" 开头。"
    )
    final_prompt = (
        "你是对话总结助手。根据下方对话内容或要点，生成结构化总结（中文，简洁）。\n"
        "格式：\n"
        "【目标/需求】\n"
        "【结论/已完成】\n"
        "【关键约束/配置】\n"
        "【涉及文件/命令】\n"
        "【待解决/下一步】\n"
        "如果没有信息写“无”。"
    )

    chunks = _chunk_lines(lines, SUMMARY_CHUNK_TOKEN_LIMIT)
    if len(chunks) == 1:
        return _summarize_text(chunks[0], final_prompt)

    partials: list[str] = []
    for chunk in chunks:
        partial = _summarize_text(chunk, chunk_prompt)
        if partial:
            partials.append(partial)
    if not partials:
        return ""

    combined = "\n".join(partials)
    return _summarize_text(combined, final_prompt)


def _build_fallback_summary(messages: list[object]) -> str:
    summary_parts = []
    for msg in messages:
        role, content = _normalize_message(msg)
        if not (_is_user_role(role) or _is_assistant_role(role)):
            continue
        preview = content[:50].replace("\n", " ")
        if len(content) > 50:
            preview += "..."
        summary_parts.append(f"  - {_display_role(role)}: {preview}")
    summary = "\n".join(summary_parts[:5])
    if len(summary_parts) > 5:
        summary += f"\n  ... 还有 {len(summary_parts) - 5} 条"
    return summary


def estimate_tokens_from_messages(messages: list[object]) -> int:
    if not messages:
        return 0
    combined = "\n".join(
        f"{role}: {content}" for role, content in (_normalize_message(m) for m in messages)
    )
    return estimate_tokens(combined)


def prompt_user_input() -> str:
    model_context_tokens = get_model_context_tokens()

    if model_context_tokens:
        # 使用配置的上下文限制或模型自身的限制（取较小值）
        effective_limit = min(model_context_tokens, MAX_CONTEXT_LIMIT)
        used_tokens = estimate_tokens_from_messages(chat_messages)
        usage_ratio = used_tokens / effective_limit
        usage_ratio = min(max(usage_ratio, 0.0), 1.0)
        remaining_ratio = max(0.0, 1.0 - usage_ratio)
        percentage = int(remaining_ratio * 100)

        # 根据剩余率设置颜色
        warn_remaining = 1 - CONTEXT_WARNING_THRESHOLD
        critical_remaining = 1 - CONTEXT_CRITICAL_THRESHOLD
        if remaining_ratio <= critical_remaining:
            color = "red bold"
            icon = "⚠"
        elif remaining_ratio <= warn_remaining:
            color = "yellow"
            icon = "⚠"
        else:
            color = "dim"
            icon = ""

        # 显示百分比 + 进度条
        bar_width = 10
        filled = int(bar_width * remaining_ratio)
        bar = "█" * filled + "░" * (bar_width - filled)

        ctx_label = f"[{color}]{icon} ctx left {percentage}% [{bar}][/{color}]"
    else:
        ctx_label = "[dim]ctx unknown[/dim]"

    line = "─" * console.width
    console.print()
    console.print(line, style="dim orange1")  # 上线
    console.print()  # 输入行占位
    console.print(line, style="dim orange1")  # 下线
    console.print(ctx_label)  # 输入框外左下角
    print("\033[3A", end="", flush=True)  # 光标上移到输入行
    user_input = console.input("[green]› [/green]")
    print("\033[2B", end="", flush=True)  # 光标下移到状态行之后
    return user_input


def compact_history(keep_recent: int = 3) -> None:
    """
    压缩历史记录，只保留最近的 N 条对话

    Args:
        keep_recent: 保留最近的对话轮数（默认 3）
    """
    global chat_messages

    user_indices = [
        idx
        for idx, msg in enumerate(chat_messages)
        if _is_user_role(_normalize_message(msg)[0])
    ]
    if len(user_indices) <= keep_recent:
        console.print(f"[dim]对话轮次只有 {len(user_indices)} 条，无需压缩[/dim]")
        return

    cut_index = user_indices[-keep_recent]
    old_messages = chat_messages[:cut_index]
    kept_messages = chat_messages[cut_index:]

    preserved_system_messages = []
    for msg in old_messages:
        role, content = _normalize_message(msg)
        if role == "system" and not _is_summary_system_message(role, content):
            preserved_system_messages.append(msg)

    summary_body = _generate_structured_summary(old_messages)
    if not summary_body:
        summary_body = _build_fallback_summary(old_messages)

    summary = f"[已压缩 {len(old_messages)} 条早期消息]\n{summary_body}".rstrip()

    before_tokens = estimate_tokens_from_messages(chat_messages)
    # 保留最近的对话，并在开头添加汇总
    chat_messages = preserved_system_messages + [("system", summary)] + kept_messages
    after_tokens = estimate_tokens_from_messages(chat_messages)
    saved_tokens = before_tokens - after_tokens

    console.print(
        f"[green]✓[/green] 已压缩历史记录：保留最近 {keep_recent} 条对话，"
        f"节省约 {format_tokens(saved_tokens)} tokens"
    )


def show_help() -> None:
    help_text = """[bold cyan]可用命令：[/bold cyan]

  [yellow]/compact[/yellow]  - 压缩历史记录（保留最近 3 条对话）
  [yellow]/clear[/yellow]    - 清空所有历史记录
  [yellow]/help[/yellow]     - 显示此帮助信息
  [yellow]/[/yellow]         - 显示所有可用命令
  [yellow]exit[/yellow] 或 [yellow]quit[/yellow] - 退出程序

[dim]上下文剩余率说明：[/dim]
  • [dim]100-21%[/dim]  - 正常（灰色）
  • [yellow]20-6%[/yellow]  - 警告（黄色 ⚠）
  • [red bold]5-0%[/red bold] - 严重（红色 ⚠），建议执行 /compact
"""
    console.print(help_text)


async def run_agent(user_input: str, messages: list[object]) -> tuple[str, list[object] | None]:
    content = ""
    last_response = ""
    final_messages: list[object] | None = None
    input_messages = list(messages) + [("user", user_input)]

    with Live(console=console, refresh_per_second=10) as live:
        async for event in agent.astream_events(
            {"messages": input_messages},
            version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_start":
                content = ""
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    content += chunk
                    live.update(Markdown(content))
            elif kind == "on_chat_model_end":
                if content:
                    last_response = content
            elif kind == "on_tool_start":
                tool_name = event["name"]
                live.console.print(f"[dim]🔧 {tool_name}[/dim]")
            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    messages_output = output.get("messages")
                    if isinstance(messages_output, list):
                        if final_messages is None or len(messages_output) >= len(final_messages):
                            final_messages = messages_output

    # 输出结束后添加空行
    console.print()
    if content:
        last_response = content
    return last_response, final_messages


def main():
    global chat_messages

    if not is_configured():
        if not setup_config(console):
            return

    init_llm()
    show_welcome()

    try:
        while True:
            user_input = prompt_user_input()

            if not user_input.strip():
                continue

            cleaned_input = user_input.strip()
            if cleaned_input in {"/", "／"}:
                user_input = "/help"

            # 处理退出命令
            if user_input.lower() in ["exit", "quit"]:
                break

            # 处理压缩命令
            if user_input.lower() == "/compact":
                console.print()
                compact_history(keep_recent=3)
                continue

            # 处理清空命令
            if user_input.lower() == "/clear":
                console.print()
                chat_messages.clear()
                console.print("[green]✓[/green] 已清空所有历史记录")
                continue

            # 处理帮助命令
            if user_input.lower() == "/help":
                console.print()
                show_help()
                continue

            console.print()
            assistant_response, updated_messages = asyncio.run(
                run_agent(user_input, chat_messages)
            )
            if updated_messages is None:
                chat_messages.append(("user", user_input))
                if assistant_response:
                    chat_messages.append(("assistant", assistant_response))
            else:
                chat_messages = updated_messages

    except KeyboardInterrupt:
        pass

    console.print("\n[dim]👋 再见！[/dim]")


if __name__ == "__main__":
    main()

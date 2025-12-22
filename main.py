from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from tools import tools
import os
import asyncio
from langgraph.checkpoint.memory import MemorySaver
import math
from openai import OpenAI
from model_config import get_model_context_length

load_dotenv()

console = Console()

MODEL = os.getenv("MODEL", "deepseek-chat")
VERSION = "0.2"
CWD = os.path.basename(os.getcwd())

# 上下文管理配置
MAX_CONTEXT_LIMIT = int(os.getenv("MAX_CONTEXT_LIMIT", "200000"))  # 默认限制 200k tokens
CONTEXT_WARNING_THRESHOLD = 0.8  # 使用超过 80% 时警告
CONTEXT_CRITICAL_THRESHOLD = 0.95  # 使用超过 95% 时严重警告

llm = ChatOpenAI(model=MODEL, streaming=True)

memory = MemorySaver()
agent = create_agent(llm, tools, checkpointer=memory)
history: list[tuple[str, str]] = []

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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # API key 不存在时，尝试从配置表获取
        tokens = get_model_context_length(MODEL)
        return tokens, "config" if tokens else "unknown"

    base_url = os.getenv("OPENAI_BASE_URL")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
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


def estimate_tokens_from_messages(messages: list[tuple[str, str]]) -> int:
    if not messages:
        return 0
    combined = "\n".join(f"{role}: {content}" for role, content in messages)
    return estimate_tokens(combined)


def prompt_user_input() -> str:
    model_context_tokens = get_model_context_tokens()

    if model_context_tokens:
        # 使用配置的上下文限制或模型自身的限制（取较小值）
        effective_limit = min(model_context_tokens, MAX_CONTEXT_LIMIT)
        used_tokens = estimate_tokens_from_messages(history)
        usage_ratio = used_tokens / effective_limit
        percentage = int(usage_ratio * 100)

        # 根据使用率设置颜色
        if usage_ratio >= CONTEXT_CRITICAL_THRESHOLD:
            color = "red bold"
            icon = "⚠"
        elif usage_ratio >= CONTEXT_WARNING_THRESHOLD:
            color = "yellow"
            icon = "⚠"
        else:
            color = "dim"
            icon = ""

        # 显示百分比 + 进度条
        bar_width = 10
        filled = int(bar_width * usage_ratio)
        bar = "█" * filled + "░" * (bar_width - filled)

        ctx_label = f"[{color}]{icon} ctx {percentage}% [{bar}][/{color}]"
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
    global history

    if len(history) <= keep_recent:
        console.print(f"[dim]历史记录只有 {len(history)} 条，无需压缩[/dim]")
        return

    removed_count = len(history) - keep_recent
    old_history = history[:removed_count]

    # 生成汇总信息
    summary_parts = []
    for role, content in old_history:
        preview = content[:50].replace("\n", " ")
        if len(content) > 50:
            preview += "..."
        summary_parts.append(f"  - {role}: {preview}")

    summary = f"[已压缩 {removed_count} 条早期对话]\n" + "\n".join(summary_parts[:5])
    if len(summary_parts) > 5:
        summary += f"\n  ... 还有 {len(summary_parts) - 5} 条"

    # 保留最近的对话，并在开头添加汇总
    history = [("system", summary)] + history[-keep_recent:]

    before_tokens = estimate_tokens_from_messages(old_history + history[-keep_recent:])
    after_tokens = estimate_tokens_from_messages(history)
    saved_tokens = before_tokens - after_tokens

    console.print(
        f"[green]✓[/green] 已压缩历史记录：保留最近 {keep_recent} 条对话，"
        f"节省约 {format_tokens(saved_tokens)} tokens"
    )


async def run_agent(user_input: str) -> str:
    content = ""
    last_response = ""
    config = {"configurable": {"thread_id": "main"}}

    with Live(console=console, refresh_per_second=10) as live:
        async for event in agent.astream_events(
            {"messages": [("user", user_input)]},
            config=config,
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

    # 输出结束后添加空行
    console.print()
    if content:
        last_response = content
    return last_response


if __name__ == "__main__":
    show_welcome()

    try:
        while True:
            user_input = prompt_user_input()

            if not user_input.strip():
                continue

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
                history.clear()
                console.print("[green]✓[/green] 已清空所有历史记录")
                continue

            # 处理帮助命令
            if user_input.lower() == "/help":
                console.print()
                help_text = """[bold cyan]可用命令：[/bold cyan]

  [yellow]/compact[/yellow]  - 压缩历史记录（保留最近 3 条对话）
  [yellow]/clear[/yellow]    - 清空所有历史记录
  [yellow]/help[/yellow]     - 显示此帮助信息
  [yellow]exit[/yellow] 或 [yellow]quit[/yellow] - 退出程序

[dim]上下文使用率说明：[/dim]
  • [dim]0-79%[/dim]   - 正常（灰色）
  • [yellow]80-94%[/yellow]  - 警告（黄色 ⚠）
  • [red bold]95-100%[/red bold] - 严重（红色 ⚠），建议执行 /compact
"""
                console.print(help_text)
                continue

            console.print()
            assistant_response = asyncio.run(run_agent(user_input))
            history.append(("user", user_input))
            if assistant_response:
                history.append(("assistant", assistant_response))

    except KeyboardInterrupt:
        pass

    console.print("\n[dim]👋 再见！[/dim]")

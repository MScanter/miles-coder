import os
import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "miles-coder"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_api_config() -> tuple[str | None, str | None, str | None]:
    """返回 (api_key, base_url, model)"""
    config = load_config()
    return (
        config.get("api_key") or os.getenv("OPENAI_API_KEY"),
        config.get("base_url") or os.getenv("OPENAI_BASE_URL"),
        config.get("model") or os.getenv("MODEL", "deepseek-chat"),
    )


def is_configured() -> bool:
    api_key, _, _ = get_api_config()
    return bool(api_key)


def setup_config(console) -> bool:
    """首次配置向导，返回是否成功"""
    console.print("\n[bold cyan]首次配置[/bold cyan]\n")
    console.print("[dim]支持 OpenAI 兼容格式的 API（OpenAI、DeepSeek、OpenRouter 等）[/dim]\n")

    base_url = console.input("[yellow]API Base URL[/yellow] (留空使用 OpenAI 默认): ").strip()
    api_key = console.input("[yellow]API Key[/yellow]: ").strip()

    if not api_key:
        console.print("[red]API Key 不能为空[/red]")
        return False

    model = console.input("[yellow]模型名称[/yellow] (默认 deepseek-chat): ").strip()
    if not model:
        model = "deepseek-chat"

    config = {"api_key": api_key, "model": model}
    if base_url:
        config["base_url"] = base_url

    save_config(config)
    console.print("\n[green]✓[/green] 配置已保存到 ~/.config/miles-coder/config.json\n")
    return True

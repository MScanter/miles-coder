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
    """Return (api_key, base_url, model)."""
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
    """First-run configuration wizard, return success status."""
    console.print("\n[bold cyan]First-time Setup[/bold cyan]\n")
    console.print("[dim]Supports OpenAI-compatible APIs (OpenAI, DeepSeek, OpenRouter, etc.)[/dim]\n")

    base_url = console.input("[yellow]API Base URL[/yellow] (leave empty for OpenAI default): ").strip()
    api_key = console.input("[yellow]API Key[/yellow]: ").strip()

    if not api_key:
        console.print("[red]API Key cannot be empty[/red]")
        return False

    model = console.input("[yellow]Model name[/yellow] (default deepseek-chat): ").strip()
    if not model:
        model = "deepseek-chat"

    config = {"api_key": api_key, "model": model}
    if base_url:
        config["base_url"] = base_url

    save_config(config)
    console.print("\n[green]✓[/green] Config saved to ~/.config/miles-coder/config.json\n")
    return True

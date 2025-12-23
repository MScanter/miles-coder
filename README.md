# Miles Coder

A CLI coding assistant agent built with LangChain.

## Features

- File read/write/edit tools
- Multi-turn conversation with memory
- LangChain Agent framework
- **Smart context management**:
  - Displays remaining context as percentage with visual progress bar
  - Color-coded warnings (yellow at 20% left, red at 5% left)
  - Configurable context limit (default 200k tokens)
  - `/compact` command to compress history
  - Automatic fallback to model configuration table when API doesn't provide context info

## Install

```bash
pip install git+https://github.com/MScanter/miles-coder.git
```

## Run

```bash
miles-coder
```

First run will prompt you to configure API URL and Key.

## Commands

Once the CLI is running, you can use these commands:

- `/compact` - Compress conversation history (keeps last 3 turns)
- `/clear` - Clear all conversation history
- `/help` - Show help message with all commands
- `/` - Show all available commands (same as `/help`)
- `exit` or `quit` - Exit the program

## Context Usage Indicators

The CLI displays remaining context with color-coded warnings:

- **100-21%** - Normal (dim gray)
- **20-6%** - Warning (yellow ⚠)
- **5-0%** - Critical (red bold ⚠) - use `/compact` to free up space

Example display: `ctx left 45% [████░░░░░░]`

## Requirements

- Python 3.13+
- OpenAI compatible API (DeepSeek, etc.)

## Configuration

Set your model and API settings in `.env`:

```bash
# Example: DeepSeek
MODEL=deepseek-chat
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.deepseek.com/v1

# Optional: Set maximum context limit (default 200000 = 200k tokens)
# Even if the model supports more, this prevents quality degradation from overly long contexts
MAX_CONTEXT_LIMIT=200000

# Example: Packy API (use a model that is available in your account)
# MODEL=your_available_model
# OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=https://api-slb.packyapi.com/v1
```

### Why Limit Context?

Even if a model supports 1M+ tokens, limiting context to ~200k tokens helps:
- **Reduce hallucinations** - Models perform better with focused context
- **Improve code quality** - Less noise means better responses
- **Prevent performance issues** - Faster response times
- **Cost savings** - Fewer tokens = lower API costs

This follows best practices from production CLI tools like Claude Code.

### Context Length Detection

The context usage indicator (e.g., `ctx left 45% [████░░░░░░]`) shows how much context remains. The system attempts to detect the model's context window in the following order:

1. **API Detection** (preferred): Queries the provider's `models.retrieve()` or `models.list()` endpoint
2. **Hardcoded Config** (fallback): Uses the built-in model configuration table in `model_config.py`

**Note**: Most API providers (OpenAI, DeepSeek, Anthropic) do not expose context length information through their API endpoints, so the fallback configuration table is used in most cases. The table is regularly updated with official model specifications.

Supported models include:
- OpenAI: GPT-4, GPT-3.5, o1/o3/o4 series, GPT-5
- Anthropic: Claude 3/4 series
- DeepSeek: V3, V3.1, V3.2, R1, Coder
- Google: Gemini 1.5/2.0/3.0 series
- Meta: Llama 2/3 series
- Mistral, Qwen, and more

## Troubleshooting

- `model_not_found` or "no available distributor": the selected `MODEL` is not available for the
  configured `OPENAI_BASE_URL`. Pick an available model name for that provider or switch to a
  provider where the model exists.

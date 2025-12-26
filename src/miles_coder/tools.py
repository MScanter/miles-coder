from langchain_core.tools import tool
import os


@tool
def read_file(path: str) -> str:
    """Read file content."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File {path} not found"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Successfully wrote to {path}"


@tool
def list_files(directory: str = ".") -> str:
    """List files in directory."""
    try:
        return "\n".join(os.listdir(directory))
    except FileNotFoundError:
        return f"Error: Directory {directory} not found"


@tool
def edit_file(path: str, old_content: str, new_content: str) -> str:
    """Replace part of file content."""
    try:
        with open(path, "r") as f:
            content = f.read()
        if old_content not in content:
            return "Error: Content to replace not found"
        content = content.replace(old_content, new_content, 1)
        with open(path, "w") as f:
            f.write(content)
        return f"Updated {path}"
    except FileNotFoundError:
        return f"Error: File {path} not found"


@tool
def search_code(pattern: str, directory: str = ".") -> str:
    """Search for keyword in code, return matching files and lines."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", pattern, directory],
        capture_output=True,
        text=True
    )
    if result.stdout:
        return result.stdout[:2000]
    return "No matches found"


@tool
def get_project_overview(directory: str = ".") -> str:
    """Get project overview: file structure + README + entry file."""
    result = []

    # 1. File structure (show first 3 levels, skip irrelevant dirs)
    result.append("## Project Structure\n```")
    skip_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.idea', 'venv'}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        level = root.replace(directory, '').count(os.sep)
        if level > 2:
            continue
        indent = '  ' * level
        folder_name = os.path.basename(root) or directory
        result.append(f"{indent}{folder_name}/")
        for f in sorted(files)[:10]:
            if not f.startswith('.'):
                result.append(f"{indent}  {f}")
    result.append("```")

    # 2. README
    for readme in ['README.md', 'readme.md', 'README.txt', 'README']:
        path = os.path.join(directory, readme)
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()[:2000]
            result.append(f"\n## README\n{content}")
            break

    # 3. Entry file
    for entry in ['main.py', 'app.py', 'index.py', 'index.js', 'index.ts']:
        path = os.path.join(directory, entry)
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()[:1500]
            result.append(f"\n## Entry File ({entry})\n```python\n{content}\n```")
            break

    return '\n'.join(result)


tools = [read_file, write_file, list_files, edit_file, search_code, get_project_overview]

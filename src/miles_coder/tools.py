from langchain_core.tools import tool
import os


@tool
def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"


@tool
def write_file(path: str, content: str) -> str:
    """写入内容到文件"""
    with open(path, "w") as f:
        f.write(content)
    return f"成功写入内容到文件 {path}"


@tool
def list_files(directory: str = ".") -> str:
    """列出目录中的文件"""
    try:
        return "\n".join(os.listdir(directory))
    except FileNotFoundError:
        return f"错误：目录 {directory} 不存在"


@tool
def edit_file(path: str, old_content: str, new_content: str) -> str:
    """替换文件中的部分内容"""
    try:
        with open(path, "r") as f:
            content = f.read()
        if old_content not in content:
            return "错误: 未找到要替换的内容"
        content = content.replace(old_content, new_content, 1)
        with open(path, "w") as f:
            f.write(content)
        return f"已更新 {path}"
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"


@tool
def search_code(pattern: str, directory: str = ".") -> str:
    """在代码中搜索关键词，返回匹配的文件和行"""
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", pattern, directory],
        capture_output=True,
        text=True
    )
    if result.stdout:
        return result.stdout[:2000]
    return "未找到匹配"


@tool
def get_project_overview(directory: str = ".") -> str:
    """获取项目概览：文件结构 + README + 入口文件，帮助理解整个项目"""
    result = []

    # 1. 文件结构（只显示前3层，跳过无关目录）
    result.append("## 项目结构\n```")
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

    # 3. 入口文件
    for entry in ['main.py', 'app.py', 'index.py', 'index.js', 'index.ts']:
        path = os.path.join(directory, entry)
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()[:1500]
            result.append(f"\n## 入口文件 ({entry})\n```python\n{content}\n```")
            break

    return '\n'.join(result)


tools = [read_file, write_file, list_files, edit_file, search_code, get_project_overview]

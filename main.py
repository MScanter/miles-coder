from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from tools import TOOLS_MAP, TOOLS_SCHEMA
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from prompt_toolkit import PromptSession


console = Console()
session = PromptSession()



load_dotenv()
#此agentDEMO是一个Claude code 编程助手一样的简单的编程agent，agent是可以学的么么哒


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL", "gemini-2.5-flash")


def msg_to_dict(msg):
    """将 API 响应的 message 对象转为标准 dict"""
    d = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d


messages = [{"role": "system", "content": "你是一个编程助手，你可以使用下面的工具来帮助用户完成编程任务。请根据用户的需求选择合适的工具，并调用它们来获取所需的信息或执行任务。你需要根据用户的输入，决定是否调用工具，并在调用后处理返回的结果。请确保你的回答简洁明了，直接解决用户的问题。"},]

console.print(f"""
{'='*50}
    Coding Agent v0.1
    模型: {MODEL}
    输入 exit 退出
{'='*50}
""")


try:
    while True:
        user_input = session.prompt(">>> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        messages.append({"role": "user", "content": user_input})
        while True:
            response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            stream=True,
            )

            # 累积数据
            content = ""
            tool_calls_data = {}

            # 流式输出
            with Live(Spinner("dots", text="思考中..."), console=console, refresh_per_second=15) as live:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # 文字内容
                    if delta.content:
                        content += delta.content
                        live.update(Panel(Markdown(content), border_style="dim"))

                    # 工具调用累积
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                tool_calls_data[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_data[idx]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_calls_data[idx]["arguments"] += tc.function.arguments

            # 判断是否有工具调用
            if tool_calls_data:
                tool_calls_list = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls_data.values()
                ]
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls_list})

                for tc in tool_calls_data.values():
                    name = tc["name"]
                    args = json.loads(tc["arguments"])
                    console.print(f"🔧 {name}({args})")
                    try:
                        result = TOOLS_MAP[name](**args)
                        console.print("✅ 执行成功")
                    except Exception as e:
                        result = f"Error: {e}"
                        console.print(f"❌ {e}")
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
            else:
                messages.append({"role": "assistant", "content": content})
                break
except KeyboardInterrupt:
    console.print("\n再见！")
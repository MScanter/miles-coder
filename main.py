from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from tools import TOOLS_MAP, TOOLS_SCHEMA
load_dotenv()
#此agentDEMO是一个Claude code 编程助手一样的简单的编程agent，agent是可以学的么么哒

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

messages = [{"role": "system", "content": "你是一个编程助手，你可以使用下面的工具来帮助用户完成编程任务。请根据用户的需求选择合适的工具，并调用它们来获取所需的信息或执行任务。你需要根据用户的输入，决定是否调用工具，并在调用后处理返回的结果。请确保你的回答简洁明了，直接解决用户的问题。"},]

while True:
    user_input = input ("请输入：")
    if user_input.lower() in ["exit","quit"]:
        break
    messages.append({"role": "user", "content": user_input})
    for msg in messages:
        if hasattr(msg,'reasoning_content'):
            msg.reasoning_content = None
    while True:
        response = client.chat.completions.create(
        model="gemini-2.5-flash-thinking",
        messages=messages,
        tools=TOOLS_SCHEMA,
        )

        resp_msg = response.choices[0].message
        reasoning = getattr(resp_msg, "reasoning_content", None)
        if reasoning:
            print(f"\n🧠 思考过程: {reasoning}\n")
        if resp_msg.tool_calls:
            messages.append(resp_msg)  # 模型的 tool_call 消息
            for tool_call in resp_msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                try:
                    print(f"\n🔧 调用工具: {name}")
                    print(f"   参数: {args}")
                    result = TOOLS_MAP[name](**args)
                    print(f"✅ 执行成功\n")
                except Exception as e:
                    result = f"工具执行失败: {e}"
                    print(f"❌ 执行失败: {e}\n")
            
                messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })
        else:
            print(f"\n{'='*50}")
            print(resp_msg.content)
            print(f"{'='*50}\n")
            messages.append(resp_msg)
            break

    


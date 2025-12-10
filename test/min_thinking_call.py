import os
from dotenv import load_dotenv
from openai import OpenAI


def run():
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model="gemini-2.5-flash-thinking",
        messages=[
            {"role": "system", "content": "你是一个调试助手。"},
            {"role": "user", "content": "测试一下你是否可以响应？"},
        ],
    )
    print(response.choices[0].message)


if __name__ == "__main__":
    run()

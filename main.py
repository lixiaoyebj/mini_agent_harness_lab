import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SCHEMAS, TOOL_REGISTRY

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com"
    ),
    timeout=30.0,
    max_retries=3,
)


MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def run_agent(user_input: str, max_steps: int = 6):
    messages = [
        {
            "role": "system",
            "content": (
                "你是工程数据分析助手。"
                "需要分析csv是调用工具，不得编造数据或统计结果，只处理用户明确要求的数据任务。"
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]


    for step in range(max_steps):

        print("\n========== Messages Debug ==========")

        for i, message in enumerate(messages):
            print(f"\nMessage [{i}]")
            print(json.dumps(
                message,
                ensure_ascii=False,
                indent=2,
                default=str
            ))

        print("====================================\n")


        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or "模型未返回结果"

        for call in msg.tool_calls:
            name = call.function.name
            print(f"[Step {step +1}] Tool: {name}")

            try:
                args = json.loads(call.function.arguments)

                if name not in TOOL_REGISTRY:
                    raise ValueError(f"未知工具：{name}")

                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是json对象")

                if name == "csv_statistics":
                    if set(args) != {"filename"}:
                        raise ValueError("工具参数不合法")
                    if not isinstance(args["filename"], str):
                        raise ValueError("filename必须是字符串")

                result = TOOL_REGISTRY[name](**args)

            except ValueError as exc:
                result = {
                    "error": type(exc).__name__,
                    "message": str(exc)
                }

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str
                )
            })

    return "达到最大执行轮数，任务未完成"


if __name__ == "__main__":
    question = (
        "请分析sample.csv文件，告诉我压力的最大值、平均值和温度的变化。"
    )

    print(run_agent(question, max_steps=6))


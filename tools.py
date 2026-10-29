from pathlib import Path
from pydoc import describe

import pandas as pd


DATA_DIR = (Path(__file__).parent / "data").resolve()


def csv_statistics(filename: str) -> dict:
    """计算csv文件中数值列的统计信息"""
    path = (DATA_DIR / filename).resolve()

    # 文件访问限制在 data 目录内，避免模型任意读取本地文件
    if DATA_DIR not in path.parents:
        raise ValueError("禁止访问数据目录之外的文件")

    if path.suffix.lower() != ".csv":
        raise ValueError("只允许读取csv文件")

    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("文件不能超过5MB")

    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError("csv没有数值列")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "statistics": numeric.describe().round(4).to_dict()
    }


# 告诉模型有哪些工具，以及工具参数格式，这是OpenAI API 约定
TOOL_SCHEMAS = [
    {
        "type": "function",                             #声明工具类型为函数 OpenAI API 约定
        "function": {                                   #函数定义对象 Chat Completions API 格式
            "name": "csv_statistics",
            "description": "读取data目录中的csv文件，并统计数值列",            #告诉模型工具的用途
            "parameters": {                             #函数参数定义
                "type": "object",
                "properties": {                         #描述参数字段
                    "filename": {
                        "type": "string",               #参数类型为字符串
                        "description": "csv文件名，例如sample.csv"
                    }
                },
                "required": ["filename"],               #指定必填参数
                "additionalProperties": False           #禁止传入未声明的额外参数
            }
        }
    }
]


# 将模型输出的工具名称映射到真实Python函数
TOOL_REGISTRY = {
    "csv_statistics": csv_statistics
}
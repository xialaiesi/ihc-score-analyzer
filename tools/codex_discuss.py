#!/usr/bin/env python3
"""
Codex 讨论工具 - 将代码问题发送给 OpenAI Codex/GPT 获取建议
用法: python3 tools/codex_discuss.py "讨论主题" [--code-file ihc_scorer.py] [--lines 100-200]
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import openai
except ImportError:
    print("需要安装 openai 包: pip3 install openai")
    sys.exit(1)


def load_code_context(file_path, line_range=None):
    """加载代码上下文"""
    path = Path(file_path)
    if not path.exists():
        return f"[文件不存在: {file_path}]"

    lines = path.read_text(encoding="utf-8").splitlines()

    if line_range:
        start, end = line_range
        start = max(0, start - 1)
        end = min(len(lines), end)
        selected = lines[start:end]
        header = f"# {file_path} (行 {start+1}-{end})"
    else:
        # 如果文件太大，只取前500行
        if len(lines) > 500:
            selected = lines[:500]
            header = f"# {file_path} (前500行, 共{len(lines)}行)"
        else:
            selected = lines
            header = f"# {file_path} (共{len(lines)}行)"

    return header + "\n" + "\n".join(selected)


def discuss_with_codex(topic, code_context, model="gpt-4o"):
    """发送讨论请求到 OpenAI API"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("=" * 50)
        print("错误: 未设置 OPENAI_API_KEY 环境变量")
        print()
        print("请设置 API Key:")
        print("  export OPENAI_API_KEY='你的API密钥'")
        print()
        print("或在 .env 文件中添加:")
        print("  OPENAI_API_KEY=你的API密钥")
        print("=" * 50)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    system_prompt = """你是一位资深的医学图像处理和Python GUI开发专家。
你正在协助开发一个IHC(免疫组化)评分分析软件，类似ImageJ的功能。
请用中文回答，给出具体的、可操作的建议。
如果涉及代码修改，请给出具体的代码示例。"""

    user_message = f"""## 讨论主题
{topic}

## 相关代码
```python
{code_context}
```

请针对以上主题给出你的分析和建议，包括:
1. 当前实现的优缺点
2. 具体的改进方案（附代码示例）
3. 需要注意的边界情况或潜在问题"""

    print(f"\n正在与 Codex ({model}) 讨论: {topic}")
    print("=" * 50)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=4000,
        )

        reply = response.choices[0].message.content
        print("\n## Codex 回复:\n")
        print(reply)
        print("\n" + "=" * 50)

        # 保存讨论记录
        log_dir = Path(__file__).parent.parent / "docs" / "discussions"
        log_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{timestamp}_{topic[:30].replace(' ', '_')}.md"
        log_file.write_text(
            f"# 讨论: {topic}\n\n"
            f"**时间**: {datetime.now():%Y-%m-%d %H:%M}\n"
            f"**模型**: {model}\n\n"
            f"## 回复\n\n{reply}\n",
            encoding="utf-8"
        )
        print(f"\n讨论记录已保存: {log_file}")

        return reply

    except Exception as e:
        print(f"\nAPI 调用失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="与 Codex 讨论代码问题")
    parser.add_argument("topic", help="讨论主题")
    parser.add_argument("--code-file", default="ihc_scorer.py", help="相关代码文件")
    parser.add_argument("--lines", help="代码行范围, 如 100-200")
    parser.add_argument("--model", default="gpt-4o", help="使用的模型 (默认 gpt-4o)")

    args = parser.parse_args()

    # 解析行范围
    line_range = None
    if args.lines:
        parts = args.lines.split("-")
        line_range = (int(parts[0]), int(parts[1]))

    # 加载代码
    code_context = load_code_context(args.code_file, line_range)

    # 发送讨论
    discuss_with_codex(args.topic, code_context, args.model)


if __name__ == "__main__":
    main()

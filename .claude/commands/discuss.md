调用 Codex 进行技术讨论。将当前问题或需求发送给 Codex，获取第二视角的建议。

讨论主题: $ARGUMENTS

## 工作流程

1. 读取 ihc_scorer.py 中与讨论主题相关的代码
2. 调用 `python3 tools/codex_discuss.py` 将问题和代码发送给 Codex
3. 整理 Codex 的回复，对比自己的分析
4. 向用户展示综合建议:
   - Claude 的分析
   - Codex 的建议
   - 综合推荐方案
5. 询问用户是否采纳某个方案并实施

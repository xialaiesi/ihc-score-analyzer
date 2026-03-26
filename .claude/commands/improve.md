根据用户描述的需求或问题，改进 ihc_scorer.py:

## 工作流程

1. **理解需求**: 确认用户想要改进什么（如果参数中有说明则直接执行）
   - $ARGUMENTS
2. **阅读现有代码**: 读取相关部分的代码，理解当前实现
3. **设计方案**: 简要说明改进方案，得到用户确认
4. **实施修改**: 修改代码
5. **语法检查**: 运行 `python3 -c "import py_compile; py_compile.compile('ihc_scorer.py', doraise=True)"`
6. **记录变更**: 将本次改进追加到 docs/changelog.md
7. **询问打包**: 问用户是否需要重新打包

## 注意事项
- 保持暗色主题风格一致
- 中文 UI，避免中文引号
- matplotlib 图表需确保中文字体配置

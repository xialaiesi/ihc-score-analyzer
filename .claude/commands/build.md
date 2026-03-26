重新打包 IHC Score Analyzer:

1. 运行语法检查: `python3 -c "import py_compile; py_compile.compile('ihc_scorer.py', doraise=True)"`
2. 如果语法检查通过，执行打包:
   ```
   pyinstaller --onefile --windowed --name "IHC_Score_Analyzer" --noconfirm ihc_scorer.py
   ```
3. 确认打包结果: `ls -lh dist/`
4. 报告打包状态

如果需要 Windows 版本，提醒用户推送到 GitHub 并通过 Actions 构建。

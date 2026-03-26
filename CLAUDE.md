# IHC Score Analyzer - 项目指南

## 项目简介
免疫组化(IHC)评分分析桌面软件，基于 PyQt5 + OpenCV + scikit-image，类似 ImageJ 的 IHC 评分工具。

## 技术栈
- Python 3.11+ / PyQt5 (GUI)
- OpenCV (图像处理)
- scikit-image (颜色反卷积)
- matplotlib (图表可视化)
- PyInstaller (打包)

## 核心文件
- `ihc_scorer.py` — 主程序（单文件架构）
- `requirements.txt` — 依赖列表
- `.github/workflows/build.yml` — CI/CD 自动构建 macOS + Windows
- `tools/codex_discuss.py` — Codex 讨论集成工具

## 核心功能
- 颜色反卷积 (H-DAB, H-E, skimage HED)
- H-Score 计算: 1×弱阳% + 2×中阳% + 3×强阳%, 范围 0-300
- ROI 区域选择分析
- 批量分析 + CSV 导出
- 评分叠加可视化

## 开发规范
- 所有 UI 文本使用中文
- matplotlib 图表必须配置中文字体 (PingFang/SimHei)
- 字符串中避免使用中文引号（用方括号或英文引号替代）
- 暗色主题风格 (#1e1e1e 背景)
- 打包前必须通过 `python3 -c "import py_compile; py_compile.compile('ihc_scorer.py', doraise=True)"` 语法检查

## 改进记录
在 `docs/changelog.md` 中记录每次改进的内容和原因。

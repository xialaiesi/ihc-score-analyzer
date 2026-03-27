# IHC Score Analyzer

免疫组化 [IHC] 评分分析桌面软件，基于 `PyQt5 + OpenCV + scikit-image + matplotlib` 开发，面向 H-DAB / H-E 染色图像的半定量分析。软件提供单张分析、ROI 选择、批量统计、CSV 导出和评分叠加可视化，适合做快速阅片辅助和批量初筛。

## 功能特性

- 颜色反卷积: 支持 `H-DAB`、`H-E`、`H-DAB (skimage)` 三种方案
- DAB 通道分析: 自动提取 DAB 灰度通道并绘制直方图
- ROI 分析: 支持框选局部区域单独评分
- H-Score 计算: `1 x 弱阳% + 2 x 中阳% + 3 x 高强阳%`
- IHC Score 计算: `强度评分 x 比例评分`
- 批量分析: 支持批量打开、整文件夹分析、批量 CSV 导出
- 结果可视化: 提供原图、DAB 通道、Hematoxylin 通道、评分叠加图
- 中文界面: 所有主界面文案均为中文，适合直接使用
- Unicode 路径兼容: 可读取中文路径图像

## 技术栈

- Python 3.11+
- PyQt5
- OpenCV
- NumPy
- scikit-image
- matplotlib
- Pillow
- PyInstaller

## 安装

建议使用 Python 3.11 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python3 ihc_scorer.py
```

## 打包前语法检查

项目约定在打包前执行以下检查:

```bash
python3 -c "import py_compile; py_compile.compile('ihc_scorer.py', doraise=True)"
```

## 打包

本地可直接使用 PyInstaller:

```bash
pyinstaller --onefile --windowed --name "IHC_Score_Analyzer" --noconfirm ihc_scorer.py
```

仓库内也提供 GitHub Actions 自动构建，见 `.github/workflows/build.yml`。

## 使用流程

1. 打开单张图像，或通过 [批量打开] / [打开文件夹] 加载样本。
2. 选择染色方案:
   `H-DAB`、`H-E`、`H-DAB (skimage)`。
3. 按需要开启或关闭自动白平衡。
4. 调整阈值:
   强阳性、阳性、弱阳性、背景排除。
5. 可选:
   使用 [选择ROI] 框选局部区域。
6. 点击 [分析] 查看单张结果，或点击 [批量分析] 获取整批统计。
7. 导出 CSV 或保存评分叠加图像。

## 当前评分规则

以下规则与当前代码实现保持一致。

### 1. 灰度分级

DAB 通道按灰度值分级，灰度越低表示染色越深。

- 强阳性: `0-60`
- 阳性: `61-120`
- 弱阳性: `121-180`
- 阴性: `181-235`
- 背景: `236-255`，自动排除

默认滑块值为:

- 强阳性上界: `60`
- 阳性上界: `120`
- 弱阳性上界: `180`
- 背景排除阈值: `236`

### 2. H-Score

```text
H-Score = 1 x 弱阳% + 2 x 中阳% + 3 x 高强阳%
```

理论范围: `0-300`

### 3. 强度评分 [0-3]

当前版本按阳性区域平均灰度估算强度，目的是减少弱串色把所有样本都压成 1 分的问题。

- 无阳性像素: `0`
- 阳性区域平均灰度 `<= 110`: `3`
- 阳性区域平均灰度 `<= 150`: `2`
- 其余阳性情况: `1`

### 4. 比例评分 [0-4]

按阳性率计算:

- `0%`: `0`
- `0% < 阳性率 <= 25%`: `1`
- `25% < 阳性率 <= 50%`: `2`
- `50% < 阳性率 <= 75%`: `3`
- `75% < 阳性率 <= 100%`: `4`

### 5. IHC Score

```text
IHC Score = 强度评分 x 比例评分
```

理论范围: `0-12`

### 6. 临床判定

当前版本包含两套展示:

- 批量结果表 / CSV:
  输出二分类 `Positive / Negative`
- 单图详情:
  输出 `阴性 [-] / 弱阳性 [+] / 阳性 [++] / 强阳性 [+++]`

二分类规则:

- 阳性率 `< 5%`: `Negative`
- 阳性率 `>= 5%`: `Positive`

单图详情规则:

- `IHC Score = 0`: `阴性 [-]`
- `IHC Score <= 3`: `弱阳性 [+]`
- `IHC Score <= 6`: `阳性 [++]`
- 其余: `强阳性 [+++]`

## 批量结果字段

当前批量结果表和 CSV 列为:

- 序号
- 图片名称
- 总像素
- 高强阳(%)
- 中阳(%)
- 低阳(%)
- 阴性(%)
- 临床判定
- 强度评分
- 比例评分
- IHC评分

## 界面说明

左侧为图像显示区，包含四个标签页:

- 原始图像
- DAB通道
- Hematoxylin通道
- 评分结果

右侧为参数和结果面板:

- 颜色反卷积设置
- 阈值设置
- DAB通道直方图
- 评分结果文本和饼图

底部为批量分析结果表，只有执行批量分析后显示。

## 项目结构

```text
ihc_scorer.py              主程序，当前为单文件架构
requirements.txt           Python 依赖列表
docs/changelog.md          变更记录
tools/codex_discuss.py     代码讨论辅助脚本
.github/workflows/build.yml GitHub Actions 自动构建
```

## 已知特点

- 当前主程序为单文件结构，UI、算法、导出逻辑集中在 `ihc_scorer.py`
- 批量判定偏向快速汇总，单图详情保留更细的分级信息
- 阈值和强度映射已经基于桌面 `CALD KI67` 样例图做过一轮校正，但不同实验批次仍可能需要手动微调

## 开发约定

- 所有 UI 文本使用中文
- matplotlib 图表必须配置中文字体
- 字符串中避免使用中文引号
- 默认暗色主题风格
- 每次改动需同步更新 `docs/changelog.md`


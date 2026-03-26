# 变更记录

## v1.0.0 (2026-03-26)
- 初始版本
- 颜色反卷积 (H-DAB, H-E, skimage HED)
- H-Score 计算 + 阳性率
- ROI 区域选择
- 批量分析 + CSV 导出
- 评分叠加可视化 (蓝/绿/橙/红)
- 暗色主题 UI
- macOS + Windows 自动构建 (GitHub Actions)

### 修复
- 修复 matplotlib 中文字体乱码
- 修复字符串中中文引号导致的语法错误
- 添加打开文件夹功能
- 按 IHC_Profiler 标准统一灰度区间定义，明确背景为 236-255 并自动排除
- 调整 IHC Score 计算展示，强度评分改为按主导阳性等级判定并统一中文结果文本

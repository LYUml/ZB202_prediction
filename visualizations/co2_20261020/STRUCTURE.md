# 可视化模块结构

使用 codebase-design 的小 interface、深 module 和 locality 原则。没有找到名称恰好为 codebase-structure 的已安装 skill；本项目此前使用的相近 skill 是 codebase-design。

- `generate.py`：唯一总入口，生成全部图并导出指标。
- `scripts/_figbase.py`：集中读取冻结结果、校验哈希、应用真实标签掩码、统一时间与图形风格；调用者不需重复这些规则。
- `scripts/gen_fig_*.py`：单图运行入口，复用同一数据与指标口径。
- `scripts/export_metrics.py`：导出可检查的指标、真实浓度和来源。
- `figures/`：交付PNG和PDF，不存源数据或模型权重。
- `data/`：从原始冻结结果派生的CSV和来源记录。
- `_utils/`：绘图skill实际使用的字体和风格模块。

预测、训练与物理模拟仍在原来的模块中。可视化仅消费已有预测产物；不导出数据库凭据，不修改训练与模型逻辑。

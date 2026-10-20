# ZB202 Prediction

ZB202 房间 **CO₂ 短期预测与通风情景模拟**。根据过去 4 小时的真实历史，预测未来 1 小时五个点位的浓度；在明确假设下比较通风策略。

当前方案为 **CO₂ 联合 TCN，三个固定种子预测取平均**。新增约四天确认期 MAE **6.57 ppm**，持续值对照 **8.32 ppm**。尚未标定真实风阀响应或接入设备控制。

## 目录

```text
.
├── docs/                   使用方法、架构、假设与评估协议
├── reports/                按日期整理的阶段汇报
├── visualizations/         图册、绘图脚本与指标
├── src/zb202/              预测、物理模型与模拟逻辑
├── scripts/                命令行入口
├── configs/                模型参数与模拟情景
├── data/                   raw 原始 / processed 处理 / synthetic 模拟
├── influxdb/               数据导出与核验工具
├── requirements/           预测环境依赖
└── tests/                  数据、预测与物理模型检查
```

## 文档导航

| 分类 | 入口 |
| --- | --- |
| 阶段汇报 | [2026-10-20 report](<reports/2026-10-20 report.md>) |
| 可视化 | [CO₂ 图册与读图说明](visualizations/co2_20261020/README.md) |
| 使用与复现 | [安装、数据、预测、模拟及检查命令](docs/usage.md) |
| 评估方法 | [评估协议](docs/forecasting-validation.md) · [消融协议](docs/forecasting-ablation.md) |
| 建模说明 | [架构](docs/architecture.md) · [参数与假设](docs/assumptions.md) · [方法方案](docs/method-proposal.md) |
| 数据 | [数据说明](data/README.md) · [处理数据](data/processed/zb202_iot/README.md) · [模拟数据](data/synthetic/README.md) |
| 项目管理 | [项目计划](PROJECT_PLAN.md) · [聊天交接](HANDOFF.md) |

生成的模型权重、处理数据和模拟运行目录默认不纳入 Git；复现所需的本地产物与条件见使用文档。

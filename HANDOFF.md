# ZB202 Prediction — 新聊天交接

更新：2026-10-20。工作区 `/Users/lml/Desktop/ZB202_prediction`。仓库 https://github.com/LYUml/ZB202_prediction.git 。交接前工作区干净，当前 HEAD 为 `9207826`；历史曾被重新整理，不依赖旧聊天中的其他提交编号。

## 用户目标与偏好

用 ZB202 的真实传感器历史、Revit/IFC 空间参考和 AHU 信息，逐步做房间预测、通风情景模拟，最终推荐新风量/阀门开度。当前没有 BA 历史或真实控制能力，允许建立明确标注的假设与合成情景，后续接真实 BA。

用户希望结构模块化、目录干净。Node/InfluxDB 工具集中在 influxdb/；Python 依赖集中在 requirements/。真实原始数据 raw、处理后的真实数据 processed、模拟数据 synthetic 分开。已明确不需要 HongKong_Weather，不要恢复导出天气库。

## 先读

1. PROJECT_PLAN.md：用户已同意据此开工，完整思路与实施顺序。
2. README.md：当前目录、安装与运行命令。
3. docs/method-proposal.md：调研及方法组合方案。
4. docs/architecture.md、docs/assumptions.md：物理模型、接口及证据边界。
5. reports/forecasting/20261018T131219Z-d95c977c.md：首轮真实实验报告。

## 已完成

- 导出并核验 zb202_iot，共 949,752 条字段观测（不是独立上报次数）。
- 原始快照：data/raw/zb202_iot/2026-10-16T09-22-35.747Z/。实际数据约 2026-08-13 至 2026-10-16。
- 保存用户 IFC 和 AHU 截图至 data/raw/reference/，附哈希。
- 简化 CO₂/AHU 模拟器，含固定开度、比例反馈，结果写 synthetic，尚未标定。
- 真实记录因果时间对齐、观测掩码、数据年龄；直接多步 TCN，时间切分、训练段标准化、验证早停及持续值对照。
- 已训练首轮；11 项测试通过。
- LSTM、PIAD-SRNN、MPC、BIM 几何提取、参数辨识尚未实现。不要把调研方案描述成已完成。

## 数据与物理边界

四个 AM103 CO₂ 典型 5 分钟间隔，AM308 为 10 分钟；有接近 12 小时中断。VS341 为位置占用 0/1，不是准确全室人数；门磁和占用事件的持续规则未核实。WS523 为插座功率，不是 AHU 能耗。

IFC2X3 未发现导出 IfcSpace/房间体积数量项，传感器为 23 个 Proxy。空间参数仍待确认；不要把总包围盒当净空气体积，不要相信 MEP 参数自动等于真实工程。

AHU-LG1-10 截图时间未知，服务区标注 ZB202。风阀指令30%/位置11%，总送风8,168 L/s，送风22.3°C/回风23.4°C，水阀指令51%/反馈48%，风机60%/40Hz。这些仅参考，不是同步历史，不可把总送风乘开度推成新风。

当前模拟：充分混合 CO₂ 质量守恒 + 一阶执行器 + 假设新风比例曲线。300 m³体积、0.5 m³/s房间送风等是假设。累计新风体积不等于节能。历史预测无 BA 输入，不能因此声称会预测改变阀门后的因果效果。

## 首轮预测实验

运行 ID：20261018T131219Z-d95c977c。
- src/zb202/forecasting/data.py：读取 annotated CSV，右闭5分钟区间，真实标签不前向填充。
- tcn.py：因果残差扩张卷积，直接多步输出，以持续值为初始残差锚点。
- train.py：70%训练、15%验证、15%测试；训练段归一化；验证早停。
- configs/forecasting/tcn.json：4小时输入、1小时输出、五个CO₂点位、32个基础特征、模型36,572参数。
- 训练/验证/测试起点：12,652 / 2,749 / 2,743；重叠窗口不是独立样本。
- 测试约 2026-10-06 至 2026-10-16；已被查看，不能反复调参后仍称未见测试。

| 时距 | TCN MAE ppm | 持续值 MAE ppm |
| --- | ---: | ---: |
| 10分钟 | 21.27 | 10.51 |
| 30分钟 | 27.05 | 12.89 |
| 60分钟 | 34.00 | 16.80 |

持续值：用当前最新读数预测所有未来时刻。误差是与未来真实记录的预测差，不是传感器测量误差。

结论：本轮 TCN 不合格，所有点位均未优于持续值；不是已经选定主模型，也不足以否定整个 TCN 方法。

后续只读检查（未保存为独立诊断报告）：
- 整体 TCN MAE27.29/RMSE164.98；持续值 MAE13.31/RMSE114.73。
- 绝对误差中位数 TCN7.22 vs 持续值8.00，但99百分位443 vs110 ppm。
- TCN最低预测约-1,851 ppm，物理不合理。
- 测试真实标签最大约9,500 ppm：真实事件或数据异常尚未确认，不能擅自删除。
- 60分钟实际变化>20ppm时，TCN MAE113.34，持续值59.94，变化段同样未胜出。
- 最佳验证MAE9.29，测试明显变差；原因未诊断确认。

## 推荐下一步（尚未执行）

优先诊断，不直接加深网络或盲目调参：
1. 核对最差预测对应原始时间、点位、输入与五点同步情况；检查9,500ppm及负预测。
2. 检查标准化/特征漂移，尤其训练期零方差或近零方差特征的未来放大。build_inputs 使用 scale>=1e-3，这是应审查的候选问题，不是已证实根因。
3. 用训练/验证时段设计诊断与滚动验证，明确已暴露测试段。
4. 再做同输入/预算的 TCN、直接多步 LSTM、PIAD-SRNN 对比。
5. 与此同时按主计划逐步确认空间与物理参数，再推进多情景控制。

用户最后实际要求是交接开新聊天，尚未明确授权“立即开始上述专项诊断”；新聊天结合用户新指令推进。

## 本地环境、命令与产物

Mac：M2 Pro、16GB、12核CPU。首轮CPU训练约3分钟，训练中一次RSS采样约419MB（非峰值）；未使用GPU。当前规模足够，无需云服务器。

```sh
# 项目根目录；.venv 已存在
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/forecast.py prepare
.venv/bin/python scripts/forecast.py train
python3 scripts/simulate.py
```

依赖：requirements/forecasting.txt（通用范围），requirements/forecasting-lock.txt（首轮Python3.13/macOS arm64精确版本）。若安装，使用独立 .venv，不污染系统。

本地未入Git的可重建产物：
- artifacts/forecasting/20261018T131219Z-d95c977c/：tcn.pt、test_predictions.npz、manifest.json。
- data/processed/zb202_iot/2026-10-16T09-22-35.747Z/causal-v1-40549f4b5ad7/：aligned.npz与manifest。

先确认产物是否仍在。预测npz含 origins、target_bin_end_utc、predictions_ppm、actual_ppm、observed_mask、persistence_ppm。时间为区间末端，不是完全同步的原始采样时刻。真实接收延迟未知，当前因果性只按记录时间验证。

InfluxDB真实凭据在 influxdb/.env，禁止显示或提交。导出只读 zb202_iot：

```sh
npm --prefix influxdb run data:export
npm --prefix influxdb run data:verify -- data/raw/zb202_iot/<snapshot>
```

## Git 与交接状态

用户曾要求将项目推送到指定GitHub仓库，代码和说明此前已同步。当前交接仅新增本 HANDOFF.md，本轮没有提交/推送或创建新聊天；由用户自行开启新聊天。不要因旧聊天的提交号与当前重写过的历史不一致而重置仓库。

## 2026-10-20 专项诊断更新

用户本轮已授权诊断与用skill整理结构。已使用codebase-design完成模块提取，新增features/windows/evaluation/diagnostics，train保持编排。详见 reports/forecasting/20261020-first-run-diagnosis.md；diagnose命令可重放。

主因已确认：VS341_04训练段完全缺测，后期1变成1000。冻结权重仅归零此数值通道，测试MAE27.29→12.01且483个负预测全部消失；这是已暴露测试的事后敏感性检查，不能当新模型成绩。验证也有15个负预测。原始9500记录在2026-10-15T07:14:05Z，保留待核实。未调参、未重训、未更改默认编码行为；下一步先修复输入契约并制定训练/验证滚动评估。原始、processed和冻结权重哈希已核验，精确重放差0ppm，11项测试通过。本轮未提交/推送。

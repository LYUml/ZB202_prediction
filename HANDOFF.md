# 新聊天交接

更新：2026-10-25。只记录继续工作所需信息；阶段状态见[项目路线](PROJECT_PLAN.md)，完整结果见[最终技术报告](<reports/2026-10-20 report.md>)。

## 先读

1. [README](README.md)：目录与文档入口。
2. [阶段汇报](<reports/2026-10-20 report.md>)：输入输出、方法对比、假设、结果及局限。
3. [评估协议](docs/forecasting-validation.md)、[物理参数](docs/assumptions.md)：开发与用途约束。
4. [使用说明](docs/usage.md)：现有命令；不要仅为重建上下文重新训练。

## 当前可复用状态

- 精简联合CO₂ TCN，固定种子42/43/44预测平均；输入48×19，输出12×5 ppm。
- 冻结协议与权重路径：`configs/forecasting/confirmation.json`。不改冻结文件来配合新数据调参。
- 新快照：`data/raw/zb202_iot/2026-10-20T09-22-56.494Z/`，1,017,502条字段观测。
- 已处理数据：`data/processed/zb202_iot/2026-10-20T09-22-56.494Z/causal-v1-a91fe5205b53/`。
- 新增确认：`artifacts/forecasting/20261020T092656Z-confirmation-25df244e/`，MAE6.566 vs8.321；该时段现已暴露。
- 无标签推理：`scripts/forecast.py predict`；最新示例`artifacts/forecasting/frozen-tcn-latest-20261020.json`。
- 图册：`visualizations/co2_20261020/README.md`；当前第2图是固定午间三小时浓度/残差双面板。绘图环境依赖与入口见该目录说明。
- 原始数据、processed、模型权重和研究审查产物保留在本地；权重等不随Git发布，新环境需恢复产物或按配置复现。

## 稳健性进展

- 本轮运行：`artifacts/forecasting/20261025T101857Z-robustness-2951db81/`，`validated-summary.json`保留完整汇总与136项产物/误差重算校验。方法和复现见`docs/forecasting-robustness.md`。
- 原冻结模型轻微噪声耐受尚可，但全部传感器偏移50ppm时MAE约43–44ppm；随机缺测5%/15%时仅约78%/44%窗口可预测，单点持续掉线完全拒绝。拒绝不是预测成功。
- 两套UCI真实CO₂数据均单通道；TCN单通道方法在UCI357比持续值差40.6%，UCI864好33.3%。不是原五点联合权重的直接跨楼宇验证。外部测试时段现已暴露，不按其结果调参。
- 45项单元测试通过；本轮原始UCI数据及来源清单随Git保存，模型/预测/processed仍在本地artifacts等目录。

## 不应忘记的限制

训练未见通道放大已通过safe-v2处理；legacy-v1仅作旧权重重放。9500ppm原始观测性质未知，不能为了好成绩删除。持续值仍是有竞争力的对照；TCN短时起伏跟踪不足、AM308略差，四天确认不覆盖极端事件。已有预测阶段记录36项检查通过，不能把旧检查记录冒充本次运行结果。

用户仅授权风阀20–40%模拟，风机固定60%。8.168m³/s为AHU总送风，房间分配未知，0.5m³/s是明确假设；没有同步BA动作历史、真实控制或能耗优化。用户不需要HongKong_Weather数据。

`influxdb/.env`包含真实凭据，禁止显示或提交；导出只读`zb202_iot`。继续工作应先检查`git status`并保留用户未提交修改，不依赖交接中的旧提交号重置仓库。代码提交或远程推送按用户当次授权执行。

## 2026-11-02 补充

- 最新快照 `data/raw/zb202_iot/2026-11-02T13-25-47.919Z/`：1,240,019 条字段观测，服务器与本地计数一致；和 10 月 25 日快照重叠的 1,102,989 条按时间、measurement、field、value 比较一致。因果处理位于 `data/processed/zb202_iot/2026-11-02T13-25-47.919Z/causal-v1-7381ba7c18e8/`。
- 新的单种子 TCN 候选及留出运行 `artifacts/forecasting/20261102T134205Z-latest-holdout-fd6d707f/`，MAE 9.064 vs 原冻结模型 9.438 ppm，但按日重采样的改善区间跨零，部分日期与 AM103_05 变差。该时段已暴露，原冻结模型仍是默认，候选需要新时段确认。详情、可复现命令及未采用的物理/patch 方法见 `docs/forecasting-validation.md` 与 `docs/usage.md`。
- 冻结基线和候选的最新快照预测分别在 `artifacts/forecasting/frozen-tcn-latest-20261102.json` 和 `artifacts/forecasting/candidate-tcn-latest-20261102.json`；权重和派生产物不随 Git 发布，需保留或恢复。

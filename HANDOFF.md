# ZB202 Prediction — 新聊天交接

更新：2026-10-20。工作区 `/Users/lml/Desktop/ZB202_prediction`。仓库 https://github.com/LYUml/ZB202_prediction.git 。交接前工作区干净，当前 HEAD 为 `9207826`；历史曾被重新整理，不依赖旧聊天中的其他提交编号。

## 当前最终状态（优先阅读）

本轮持续优化已达到当前数据条件下的实用优选/收益收敛：精简CO₂联合TCN固定42/43/44平均，开发MAE7.435。读取新数据前冻结configs/forecasting/confirmation.json（协议hash d1918285d1a26c9429412d9d01f22423f3fb115507d462e72c2d8622a4eb0bc3），新只读快照2026-10-20T09-22-56.494Z共1017502字段记录，与旧949752完全重叠一致。run20261020T092656Z-confirmation-25df244e：新增1140窗口/61370观测标签项MAE6.56591 vs持续值8.32113（21.1%改善），RMSE8.63401，无负值；AM308 MAE1.926略差于持续值1.744，新段280–721ppm不涵盖尖峰。36项检查通过；原safe/legacy编码完全一致，旧开发、新确认全窗口和最新单窗重放差0ppm。不要基于现已暴露新段再调参并称未见测试。

新增encode_inputs、inference.py FrozenForecaster与forecast.py predict/confirm，真实最新单窗artifacts/forecasting/frozen-tcn-latest-20261020.json（origin09:20Z）；不读未来标签，预测12×5ppm。采样/周期审查artifacts/research/data-contract-audit-20261020/audit.json；AM308十分钟采样，真实未来标签保留空缺，每日/每周复制差，不继续盲调。总报告reports/forecasting/20261020-optimization-conclusion.md。未来项目仍有更长时段确认、BA/空间/分配实测与MPC等；本轮优化完成不等于这些路线图功能完成。无真实写入、无新增依赖、无提交推送。

## 持续目标阶段记录（历史）

最新追加：structured.py实现按时间维递推/批次不变分解RNN与正值一阶松弛；18次拟合run20261020T091842Z-candidates-d1eba57d，固定种子平均MAE7.735/9.022，均未胜TCN7.435。33项代码检查通过、24组指标重算一致、两类checkpoint完整重放差0ppm。新增sensitivity.py和scripts/sensitivity.py、configs/scenarios/sensitivity.json：144组全因子假设×4策略，房间送风0.25/0.5/1/8.168（最后仅极端压力）、体积/排放/执行器/曲线变化，全部20–40、风机60。fixed40仍43/144组分钟末>示例800目标，不能称真实失败概率。产物data/synthetic/2026-10-20T09-19-48-sensitivity-3a9e623f，报告reports/forecasting/20261020-structured-and-allocation-review.md。用户最新答复“不清楚，保留分配比例假设”已落实；没有待答问题。目标仍active；后续新增时段确认、缺测/采样审查、推理入口与模拟控制比较，不能提前宣称总体完成。

用户已明确要求持续优化，GitHub/arXiv多方法对比，可有证据修订模拟假设，直到形成优良方案/收益收敛。goal仍active，不要因本轮阶段完成标记整体完成。工作账本docs/optimization-program.md列出未完成事项。

新增linear.py、candidates.py、candidates.json及forecast.py compare；3模型×3折×3种子27次拟合完成，run20261020T090306Z-candidates-b8727ca7，固定种子平均开发MAE：NLinear8.167、DLinear7.953、缺测感知线性8.021，未胜TCN7.435。三模型需注明输入信息不同。已保存前四个实验的精确source_snapshot（原manifest不改），新run自动保存。

用户新图保存data/raw/reference/ahu/ahu-lg1-10-user-confirmed-20261020.jpg，确认是当前BA基准。明确授权风阀20–40仅模拟，基准30/位置11；风机高低限60因此固定。房间设定26.4，范围22–27，回风设定23；8.168m³/s总送风房间分配用户不清楚，0.5m³/s保留明确假设，不改成总新风。ahu.json/simulation_envelope、demo固定20/30/40和反馈已更新；模拟器拒绝越界策略。实际模拟data/synthetic/2026-10-20T09-13-42-2a075e44，全部synthetic。

训练期6343个五点真实标签箱，相对AM103_05，中位差AM103_06+135、AM103_08−135；不能直接把五点当一个校准混合状态。room.json明确观测映射未建立。V/N/Q同比缩放等价，不能无约束辨识。证据artifacts/research/assumptions-audit-20261020/audit.json。

PIAD源审查固定commit fa1a8121016dfb8b2fdb2b91a2bdc0c4cceb85f8：SSRNN默认分支沿batch维递推，同一输入受前序批次样本影响；数学正权重特例输出2与3。DSSRNN individual分支还有未定义属性/构造不一致。源文件只读保存artifacts/research/piad-srnn-audit-20261020，未导入执行。下一步实现按时间维递推、批次不变的分解状态RNN和稳定松弛候选，并明确不是PIAD原版复现。

30项检查通过；本轮不再评价旧测试，仍无真实BA历史或写入。报告reports/forecasting/20261020-linear-and-ba-review.md。没有提交/推送。

## 上一阶段：进一步优化已完成

以本块为最新进展。四类辅助组（温湿度/占用/门磁/功率）与5个独立参数局部TCN，三折seed42筛选均未优于CO₂联合基线的内部验证均值，按预定规则不追加种子。单种子开发MAE：联合CO₂7.462，温湿度8.198，占用8.115，门磁8.020，功率8.341，独立点位7.678。

固定三个CO₂ TCN预测平均的开发MAE7.435/RMSE10.479；原三种子单模型指标均值7.547/10.626。固定半Ridge混合7.608，内部验证选混合权重7.483，未胜纯TCN平均。注意平均预测误差与误差均值不同。

9个原CO₂拟合中7个已早停；只对rolling-1 seed42和rolling-3 seed44延长上限到24轮，前12轮历史精确复现，均15轮早停、最佳仍12轮，预测与原完全一致。没有新的训练预算收益。新增共17次拟合，25项代码检查通过，本轮未再评估旧测试。

新增入口：forecast.py ablate/combine/refine，对应configs/forecasting/{ablation,ensemble,refinement}.json。报告：reports/forecasting/20261020-further-optimization.md；完整run：20261020T072049Z-ablation-e72eaea5、20261020T072301Z-ensemble-97152589、20261020T073610Z-refinement-12331a92。本地产物在同名artifacts/forecasting目录。

下一步优先冻结联合CO₂/固定种子平均候选，在新增时间段确认；继续重复旧开发集不能提供新的独立泛化证据。PIAD-SRNN/DLinear仍待实现。没有提交/推送。

## 上一轮状态：2026-10-20 输入修复与模型比较已完成

本块优先于下方首轮历史描述。默认safe-v2编码已修复训练未见通道放大问题；旧diagnose显式使用legacy-v1，原权重重放差0ppm。新增baselines.py（LSTM/Ridge）、benchmark.py及benchmark.json；命令`.venv/bin/python scripts/forecast.py benchmark`。

已完成3折×2特征设置×3种子的TCN/LSTM，另有Ridge和持续值，共42次神经拟合、7次Ridge拟合、4组持续值。开发平均MAE：持续值9.514；完整特征Ridge8.885/TCN8.455/LSTM8.748；精简CO₂版本Ridge8.247/TCN7.547/LSTM8.314。优先继续精简TCN及逐组特征消融，不急着加深网络。

旧测试只作完整特征压力检查：Ridge12.023、TCN11.337、LSTM11.721，持续值13.313 ppm。本轮全部输出无负预测，但尖峰尾部误差仍高。这里不是新的独立最终测试；精简模型按协议未追加旧测试筛选。9500ppm保留，用户反馈不清楚当时特殊活动、没有记录。

19项代码检查通过。报告：reports/forecasting/20261020-input-repair-comparison.md；完整结果：reports/forecasting/20261020T052846Z-benchmark-671d2cad.md；本地产物：artifacts/forecasting/20261020T052846Z-benchmark-671d2cad/。协议：docs/forecasting-validation.md。没有新增依赖；没有提交/推送。后续需新增时段最终验证；PIAD-SRNN、DLinear、MPC尚未实现。

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
- 最新已实现LSTM和滚动比较；PIAD-SRNN、MPC、BIM几何提取、参数辨识仍未实现。不要把调研方案描述成已完成。

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

## 首轮历史建议（已部分执行，以顶部最新状态为准）

优先诊断，不直接加深网络或盲目调参：
1. 核对最差预测对应原始时间、点位、输入与五点同步情况；检查9,500ppm及负预测。
2. 检查标准化/特征漂移，尤其训练期零方差或近零方差特征的未来放大。build_inputs 使用 scale>=1e-3，这是应审查的候选问题，不是已证实根因。
3. 用训练/验证时段设计诊断与滚动验证，明确已暴露测试段。
4. 再做同输入/预算的 TCN、直接多步 LSTM、PIAD-SRNN 对比。
5. 与此同时按主计划逐步确认空间与物理参数，再推进多情景控制。

用户已在本聊天授权专项诊断、输入修复和模型比较，执行结果见顶部最新状态。

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

## 2026-10-20 专项诊断阶段更新（历史，修复试验见顶部）

用户本轮已授权诊断与用skill整理结构。已使用codebase-design完成模块提取，新增features/windows/evaluation/diagnostics，train保持编排。详见 reports/forecasting/20261020-first-run-diagnosis.md；diagnose命令可重放。

主因已确认：VS341_04训练段完全缺测，后期1变成1000。冻结权重仅归零此数值通道，测试MAE27.29→12.01且483个负预测全部消失；这是已暴露测试的事后敏感性检查，不能当新模型成绩。验证也有15个负预测。原始9500记录在2026-10-15T07:14:05Z，保留待核实。未调参、未重训、未更改默认编码行为；下一步先修复输入契约并制定训练/验证滚动评估。原始、processed和冻结权重哈希已核验，精确重放差0ppm，11项测试通过。本轮未提交/推送。

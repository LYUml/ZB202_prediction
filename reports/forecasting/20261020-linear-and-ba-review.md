# 持续优化进展：线性模型、BA基准与假设审查

2026-10-20。持续任务尚未结束，本轮新增27次线性候选拟合，修订BA模拟边界并审查物理假设。

## 线性候选

三折、每模型三种子，24轮上限，内部验证早停；同样的起点和真实标签。完整结果见[运行报告](20261020T090306Z-candidates-b8727ca7.md)。

| 方法 | 三种子单模型MAE均值 | 固定平均预测MAE |
| --- | ---: | ---: |
| NLinear | 8.195 | 8.167 |
| DLinear | 7.997 | 7.953 |
| 缺测感知线性 | 8.295 | 8.021 |
| 既有精简TCN | 7.547 | 7.435 |

原始D/NLinear只读五个标准化CO₂历史，忽略辅助mask/age/日历；缺测输入采用既有训练均值填充，因此不能将差异完全归因于模型家族。DLinear25步平滑是作者起始定义，不代表本房间物理周期。该轮尚未优于TCN；全部输出无负预测。没有重新使用旧测试。

参考：[作者DLinear](https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py)、[作者NLinear](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)。

## PIAD来源审查

[作者SSRNN默认分支](https://github.com/ahmad-shirazi/DSSRNN/blob/fa1a8121016dfb8b2fdb2b91a2bdc0c4cceb85f8/models/SSRNN.py)在转置成batch×channel×history后，沿batch维更新共享状态。默认预测因而可能依赖之前的批次样本。正权重一维特例中，同一输入2单独计算为2，前面加入输入1后为3。这是循环结构的数学反例，不是模型训练成绩；没有执行下载的作者代码。

[DSSRNN源码](https://github.com/ahmad-shirazi/DSSRNN/blob/fa1a8121016dfb8b2fdb2b91a2bdc0c4cceb85f8/models/DSSRNN.py)的individual分支还涉及未初始化属性和构造参数不一致。不能直接整库接入并声称复现成功。后续将实现按时间维递推且批次不变的修正候选，并标明与[论文](https://arxiv.org/abs/2412.00994)及原实现的差异。

固定源文件、SHA和反例在artifacts/research/piad-srnn-audit-20261020/。当前来源代码只作审查资料，没有进入本项目导入路径。

## BA与假设

用户确认截图是当前BA基准，并授权风阀指令20–40仅用于模拟；风机上下限均60，固定。总送风8168L/s=8.168m³/s，用户暂不清楚房间分配，所以0.5m³/s继续标为示例分配，不能当实测新风。

默认情景已改为固定20/30/40及范围内反馈，模型在运行前拒绝越界固定策略和越界反馈声明。新模拟运行data/synthetic/2026-10-20T09-13-42-2a075e44/是synthetic，不是真实BA控制结果。房间温度设定26.4°C、范围22–27°C、回风设定23°C已记录；当前无热湿动态，不据此虚构温控效果。

原训练期6343个五点均有观测的时间箱，相对AM103_05，AM103_06中位差+135ppm、AM103_08为−135ppm。它们不是绝对校准值，也不能单独证明设备故障，但不足以支持五点原始读数直接对应同一个混合状态。room.json已明确观测映射未建立。物理V、N和Q同比缩放可给出完全相同CO₂轨迹，不能一起无约束辨识。

详细证据、限制见[假设说明](../../docs/assumptions.md)。本轮没有为降低误差平移原始标签或把室外420随意改成某点基线。

## 验证与未完成事项

30项代码检查通过；新BA范围模拟已运行，全部策略处于20–40。旧实验源码快照在各run/source_snapshot保存并核验，不修改旧manifest。新候选run自动保存源码快照。物理参数真实标定、修正状态RNN/结构约束候选、新增时段确认尚未完成，目标保持进行。

持续计划：[优化账本](../../docs/optimization-program.md)。

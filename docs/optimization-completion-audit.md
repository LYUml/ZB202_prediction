# 持续优化完成审计

按用户原目标“持续优化、GitHub/arXiv多方法对比、有依据修订模拟假设，直到收益收敛或形成优解”核验。结论是当前数据条件下已形成实用优选，不是数学最优或整个项目路线图完成。

| 要求 | 当前权威证据 | 判定 |
| --- | --- | --- |
| 先诊断，避免盲目调参 | 冻结首轮重放、训练未见输入放大反例；safe-v2及legacy重放，原始数据哈希 | 达成，首轮失控根因已处理 |
| 使用skill改善项目结构 | codebase-design指导的features/windows/evaluation/diagnostics拆分；模型小接口、统一build_model/encode_inputs/FrozenForecaster，README目录与真实文件一致 | 达成 |
| GitHub/arXiv依据而非只堆参数 | 作者源审查固定commit/hash，PIAD批次反例，D/NLinear与修正结构候选，报告链接主源 | 达成，明确不冒充论文复现 |
| 多方法、相同任务与可追溯比较 | benchmark/ablation/ensemble/refinement/candidates完整manifest、checkpoint、预测、源码快照；总体/分点/时距/变化/尾部/成本 | 达成，表示及预算差异已披露 |
| 约束数据泄漏、保留负结果 | 训练段标准化、内部验证选epoch、不挑最优种子；旧测试标记暴露；负结果完整保留；36项检查 | 达成，接收延迟未知为数据限制 |
| 审查数据侧假设 | 实际5/10分钟间隔、AM308标签半覆盖、近12小时中断、15分钟年龄与80%覆盖；日/周复制诊断 | 达成，无证据不继续周期调参 |
| 合理修订模拟假设与用户约束 | BA原图/哈希、20–40仅模拟、风机60；总风量与房间/新风区分；观测偏差及不可辨识性；144组敏感性 | 达成，假设不冒充标定 |
| 形成经确认的实用优选 | 先冻结权重/协议，再导出新时段；1140窗口6.566vs8.321，零负值；逐日及失败点披露；重放差0ppm | 达成，只有四天且无高值压力事件 |
| 可复用交付而非临时结果 | predict/confirm命令实跑；无未来标签接口；schema/权重/processed哈希校验；最新预测JSON；比较/结论报告 | 达成，raw/权重为本地数据产物 |
| 停止搜索有证据 | 特征/独立点位/混合/延长训练/线性/结构连续方向均未胜固定TCN平均，新增期总体确认；推理短测约1.65ms | 实用收益已收敛，不声称无法再提高 |

已检查的本地产物：artifacts/research/new-time-audit-20261020/{overlap,verification,comparison,inference-timing}.json，data-contract-audit-20261020/audit.json；各运行目录与报告见reports/forecasting/20261020-optimization-conclusion.md。最终机器核验记录存artifacts/research/optimization-completion-20261020/audit.json。

仍然存在的项目限制：传感器绝对校准未证实、AM308 MAE略差、极端尖峰来源未知、长期泛化及实际到达延迟未确认、BA动作/风量/空间尚无独立标定。它们已约束用途，并非用假数据掩盖。整个项目未来路线图中的MPC、热湿、真实能耗和BA上线没有完成，也没有作为本轮优化已完成的功能交付。

最后目标回合属于progress：新增真实快照、冻结确认、无标签推理、数据侧审查与综合报告均改变了权威状态；不是等待或状态重述。无需继续在同一已暴露开发/确认数据上扩大模型搜索。

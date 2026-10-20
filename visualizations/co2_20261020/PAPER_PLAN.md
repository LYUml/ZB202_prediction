<!-- BEGIN FIGURE_MANIFEST -->
## 图表清单（FIGURE_MANIFEST）

**数据图（matplotlib；paper-figure）：**
- fig_co2_trend | claim=五个点位真实浓度变化 | source=artifacts/forecasting/20261020T092656Z-confirmation-25df244e/predictions.npz | section=真实确认期可视化 | language=zh | format=pdf,png | layout=single-landscape | recipe:basic.line
- fig_prediction_overlay | claim=30分钟预测与实测，固定中间完整日首个点位12–15点，浓度和残差双面板 | source=artifacts/forecasting/20261020T092656Z-confirmation-25df244e/predictions.npz | section=真实确认期可视化 | language=zh | format=pdf,png | layout=2-panel-stacked | recipe:empirical.prediction_ci
- fig_horizon_error | claim=5至60分钟误差变化 | source=artifacts/forecasting/20261020T092656Z-confirmation-25df244e/predictions.npz | section=真实确认期可视化 | language=zh | format=pdf,png | layout=single-landscape | recipe:empirical.multistep_decay
- fig_sensor_error | claim=五点位误差对比 | source=artifacts/forecasting/20261020T092656Z-confirmation-25df244e/predictions.npz | section=真实确认期可视化 | language=zh | format=pdf,png | layout=single-landscape | recipe:basic.grouped_bar
- fig_error_heatmap | claim=点位与预测时距的误差矩阵 | source=artifacts/forecasting/20261020T092656Z-confirmation-25df244e/predictions.npz | section=真实确认期可视化 | language=zh | format=pdf,png | layout=single-landscape | recipe:basic.heatmap
- fig_residual_distribution | claim=预测偏高偏低的分布 | source=artifacts/forecasting/20261020T092656Z-confirmation-25df244e/predictions.npz | section=真实确认期可视化 | language=zh | format=pdf,png | layout=single-landscape | recipe:empirical.residual_diagnostics
- fig_prediction_scatter | claim=60分钟预测与实测一致性 | source=artifacts/forecasting/20261020T092656Z-confirmation-25df244e/predictions.npz | section=真实确认期可视化 | language=zh | format=pdf,png | layout=2-panel-stacked | recipe:empirical.prediction_ci

**总数：** DATA=7, DRAWIO=0, TIKZ=0, ILLUSTRATION=0, OPTIONAL=0, ALL=7
<!-- END FIGURE_MANIFEST -->

尺寸：横图6×3.8in，引用5.53in，缩放0.92；散点5×4.9in，引用4.55in，缩放0.91。
全部取冻结真实数据；不提供未经估计的置信区间。模板中的置信带因没有相应证据而省略。
颜色由setup_style主题提供；折线按点位取色、比较按方法取色，柱图浅填充原色边框，热图连续渐变。

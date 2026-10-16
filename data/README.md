# zb202_iot 本地历史快照

本目录仅保存 ZB202 房间 `zb202_iot` bucket 的原始数据。

当前快照：`2026-10-16T09-22-35.747Z/`。

- 949,752 条字段观测记录；一行是一个字段值，不是一次完整设备上报。
- 实际时间范围：2026-08-13T09:42:35.323637Z 至 2026-10-16T09:22:20Z。
- 原始文件 `zb202_iot.csv.gz` 已纳入 Git，可在 clone 后直接使用。
- `manifest.json` 记录查询范围、状态、记录数量和 SHA-256。
- `server-counts.json` 保存同一范围的服务器独立计数，与本地一致。
- `zb202_iot-summary.json` 列出 measurement、field 和记录数。

原始格式为 gzip 压缩的 InfluxDB annotated CSV，保留时间精度、字段类型和标签。没有降采样、插值或过滤设备/字段。CSV 可能包含多段表头，不能直接当单表读取；参见 `scripts/inspect-export.py` 的流式读取方式。

查询范围为 Unix epoch 至快照固定截止时间，只包含查询时服务器仍保留的可读取数据，无法恢复已被保留策略删除的记录。

重新导出方式见项目根目录 README.md。

# 数据目录约定

## raw：原始数据

路径：`raw/zb202_iot/<snapshot>/`。快照名为导出开始时刻（UTC），每次导出创建独立目录。

当前快照 `2026-10-16T09-22-35.747Z`：949,752 条字段观测，实际时间范围为 2026-08-13T09:42:35.323637Z 至 2026-10-16T09:22:20Z。

| 文件 | 用途 |
| --- | --- |
| `zb202_iot.csv.gz` | 原始 InfluxDB annotated CSV，保留精度、类型及标签 |
| `manifest.json` | 来源、查询范围、状态、记录数及 SHA-256 |
| `server-counts.json` | 同一范围的服务器独立计数 |
| `zb202_iot-earliest.csv` | 服务器最早时间查询结果 |
| `zb202_iot-summary.json` | 原始数据核验摘要与 measurement/field 清单 |

原始数据文件不做清洗、降采样、插值或覆盖。核验工具可更新同目录的元数据与摘要。CSV 一行是一个字段观测，不是一次完整设备上报；文件含多段表头，读取方法见 `scripts/validation/inspect-export.py`。

查询覆盖 Unix epoch 至统一截止时间，仅包括服务器当时仍保留的可读取记录。

## processed：处理后的数据

路径：`processed/zb202_iot/<snapshot>/<pipeline-version>/`。

目前尚无处理后的数据，只预留目录。后续清洗、时间对齐、重采样、缺失值处理与特征生成的结果放在这里，不能写回 raw。

每份处理结果应附 `manifest.json`，记录输入快照及 SHA-256、处理版本、参数、输出字段、时间范围和行数，以便追溯和重新生成。具体处理规则在建模前根据字段含义确定。

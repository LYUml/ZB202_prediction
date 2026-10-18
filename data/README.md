# 数据分层

| 目录 | 内容 | 规则 |
| --- | --- | --- |
| `raw/zb202_iot/<snapshot>/` | 真实 InfluxDB 导出及核验元数据 | 不清洗、不修改原始 CSV |
| `raw/reference/bim/` | 用户提供的三个 IFC | 空间参考；不保证工程准确性 |
| `raw/reference/ahu/` | 用户提供的 AHU 截图 | 单次读数，观测时间未知 |
| `processed/zb202_iot/` | 真实数据的清洗、对齐及特征结果 | 已实现因果对齐；生成结果需记录输入和处理方法 |
| `synthetic/<run>/` | 模拟模型生成的记录 | 必须标记 synthetic；不能混作实测数据 |

## 原始 IoT 快照

当前快照：`2026-10-16T09-22-35.747Z`，949,752 条字段观测。
实际时间范围：2026-08-13T09:42:35.323637Z 至 2026-10-16T09:22:20Z。

- `zb202_iot.csv.gz`：原始 InfluxDB annotated CSV，保留时间精度、类型与标签。
- `manifest.json`：来源、查询范围、状态、记录数及 SHA-256。
- `server-counts.json`：同一范围的服务器独立计数。
- `zb202_iot-earliest.csv`：最早时间查询结果。
- `zb202_iot-summary.json`：核验摘要与字段清单。

CSV 一行是一个字段值，不是完整设备上报；包含多段表头，读取示例见 `scripts/validation/inspect-export.py`。查询覆盖 Unix epoch 至统一截止时间，仅含服务器当时仍保留的数据。核验工具可更新元数据与摘要，但不改原始压缩文件。

## 衍生数据追溯

processed 推荐使用 `<snapshot>/<pipeline-version>/`，附 manifest 记录输入文件及哈希、处理版本、参数、字段、时间范围和行数。

synthetic 每次运行独立存储，manifest 保存完整配置、代码与输入哈希、输出校验和。默认不上传生成目录；原始参考文件的哈希见 `raw/reference/manifest.json`。

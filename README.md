# ZB202 Prediction

ZB202 房间状态预测与通风情景模拟。当前有真实 IoT 历史数据、参考 IFC/AHU 截图，以及一个使用明确假设的单房间 CO₂ 模拟器。尚未接入 BA、训练预测模型或完成实测标定。

## 目录

```text
configs/                    房间、AHU 和情景参数；记录单位与来源
├── room.json
├── ahu.json
└── scenarios/demo.json
src/zb202/                  Python 模拟核心（仅标准库）
├── adapters/               输入来源；当前为假设占用日程
├── models/                 房间 CO₂ 与简化 AHU 模型
├── controllers.py          固定开度、CO₂ 比例反馈基线
└── simulation.py           模拟编排与结果追溯
scripts/
├── simulate.py             模拟运行入口
└── validation/             原始 IoT 数据文件核验
influxdb/                   独立 Node 工具、依赖与本地连接配置
├── scripts/                数据下载与服务器计数
├── package.json
└── .env.example
data/
├── raw/                    原始 IoT 快照与参考 IFC/AHU 截图
├── processed/              从真实记录处理得到的数据；当前未生成
└── synthetic/              模拟结果；与真实数据分开存储
docs/                       建模边界、参数来源与未来 BA 接入约定
tests/                      物理守恒及模拟行为检查
```

## 运行模拟

需要 Python 3.10+，无需安装第三方 Python 包。在项目根目录运行：

```sh
python3 scripts/simulate.py
python3 -m unittest discover -s tests -v
```

默认情景为 3 小时，比较固定 30%、固定 60% 和 CO₂ 比例反馈三种策略。结果写入 `data/synthetic/<run>/`；CSV 每行表示一个区间，区分区间起止 CO₂、指令、实际开度和平均新风量。

默认体积 300 m³、房间送风 0.5 m³/s 等均为演示假设，不是从模型提取或实测的参数。截图中的 8,168 L/s 只作为 AHU 总送风参考，不用于推定房间新风量。结果只用于流程与敏感性演示，不能解释成真实房间的控制效果。

可编辑 `configs/` 后重新运行，或选择新情景：

```sh
python3 scripts/simulate.py --scenario configs/scenarios/demo.json
```

## InfluxDB 导出与核验

需要 Node.js 20+。在项目根目录运行：

```sh
npm ci --prefix influxdb
# 首次使用时复制模板；已有 .env 不要覆盖
cp influxdb/.env.example influxdb/.env
# 填入真实连接配置和 Token
npm --prefix influxdb run data:export
npm --prefix influxdb run data:verify -- data/raw/zb202_iot/<snapshot>
python3 scripts/validation/inspect-export.py data/raw/zb202_iot/<snapshot>
```

`<snapshot>` 替换为实际快照名。核验路径以项目根目录为基准。导出自动写入项目的 `data/raw/zb202_iot/`，不受运行目录影响。支持用 `ZB202_ENV_FILE` 指定其他配置文件；真实凭据不提交 Git。

当前真实快照有 949,752 条字段观测。详细约定见 [数据说明](data/README.md)、[架构说明](docs/architecture.md) 和 [参数假设](docs/assumptions.md)。

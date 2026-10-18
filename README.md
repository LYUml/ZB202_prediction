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
python3 -m unittest discover -s tests -p test_simulation.py -v
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

## 项目主计划与真实数据预测

完整思路、物理模型、预测方案、数据边界和阶段顺序见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

真实预测模块采用多变量直接多步 TCN：输入过去 4 小时，预测未来 1 小时的五个 CO₂ 点位。当前先实施 TCN 与持续值对照；LSTM、PIAD-SRNN、滚动验证和 MPC 按主计划后续推进。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-forecasting.txt
# 仅准备数据
.venv/bin/python scripts/forecast.py prepare
# 训练、验证早停，最后评估一次测试集
.venv/bin/python scripts/forecast.py train
# 包含预测与物理模型的完整检查
.venv/bin/python -m unittest discover -s tests -v
```

`requirements-forecasting-lock.txt` 记录首轮本机 Python 3.13 / macOS arm64 环境的精确版本；其他平台建议先用通用 requirements 安装。预测配置在 `configs/forecasting/tcn.json`。

处理数据写入 `data/processed/`，权重、预测与 manifest 写入 `artifacts/forecasting/`，审阅报告写入 `reports/forecasting/`。处理数据和模型产物默认不提交 Git。每次训练可能耗时数分钟，首轮单次划分结果不等于最终模型验证。

只运行无额外依赖的物理模型检查时，使用 `python3 -m unittest discover -s tests -p test_simulation.py -v`。

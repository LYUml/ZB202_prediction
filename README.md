# ZB202 Prediction

ZB202 房间 CO₂ 预测与通风情景模拟项目。目标是结合真实 IoT 数据、空间模型和 AHU 物理模型，逐步实现环境预测、新风策略比较，并在未来接入真实 BA 数据。

完整思路与阶段安排见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，方法调研见 [方法方案](docs/method-proposal.md)。

## 当前进度

- 已导出并核验 `zb202_iot` 原始快照：949,752 条字段观测。
- 已保存用户提供的 IFC 与 AHU 截图作为原始参考。
- 已实现使用明确假设的单房间 CO₂ / AHU 模拟器。
- 已实现真实数据因果对齐、多变量直接多步 TCN、时间留出评估与持续值对照。
- 已完成首轮实验和 11 项检查；**首轮 TCN 测试误差高于持续值对照，尚未选定最终预测模型**。见 [首轮实验报告](reports/forecasting/20261018T131219Z-d95c977c.md)。

后续工作：预测诊断、LSTM / PIAD-SRNN 对比、滚动验证、空间参数核实与辨识、多情景 MPC。当前尚未接入 BA，也未完成真实 AHU 响应标定。

## 目录

```text
PROJECT_PLAN.md             项目主计划与实施记录
configs/
├── forecasting/tcn.json    数据处理、TCN 与训练配置
├── room.json               房间物理参数及来源
├── ahu.json                AHU 截图参考与假设参数
└── scenarios/              模拟情景
src/zb202/
├── forecasting/            真实数据预处理、TCN、训练与评估
├── adapters/               模拟输入来源
├── models/                 CO₂ 与简化 AHU 物理模型
├── controllers.py          固定开度、CO₂ 比例反馈
└── simulation.py           模拟编排与结果追溯
scripts/                    预测、模拟及本地数据核验入口
requirements/
├── forecasting.txt         预测模块的通用 Python 依赖
└── forecasting-lock.txt    首轮本机环境的精确依赖版本
influxdb/                   独立 Node 工具、依赖与本地连接配置
data/
├── raw/                    原始 IoT 快照、IFC、AHU 截图
├── processed/              从真实记录处理得到的数据
└── synthetic/              模拟生成数据
artifacts/                  本地模型权重、预测结果与运行记录
reports/forecasting/        可审阅的实验报告
docs/                       方法、架构与假设说明
tests/                      物理模型、因果性和数据切分检查
```

`artifacts/`、生成的 processed 与 synthetic 运行目录默认不提交 Git，可通过原始数据、配置和代码重新生成。首次运行前这些目录可能不存在。

## 真实 CO₂ 预测

在项目根目录运行。首轮验证环境为 macOS arm64、Python 3.13；其他 Python 版本能否安装取决于 NumPy/PyTorch 对应发行包。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/forecasting.txt
# 仅准备数据
.venv/bin/python scripts/forecast.py prepare
# 训练、验证集早停，然后评估测试集
.venv/bin/python scripts/forecast.py train
# 完整检查
.venv/bin/python -m unittest discover -s tests -v
```

TCN 输入过去 4 小时，直接预测未来 1 小时的五个 CO₂ 点位。特征仅使用预测起点前已记录的观测；未来标签只取对应时间区间的真实 CO₂，不做前向填充。标准化仅使用训练时间段，验证集用于早停，测试集用于最终评估。

通用依赖文件给出兼容版本范围。若需复现首轮本机依赖版本，在兼容环境中改用：

```sh
.venv/bin/python -m pip install -r requirements/forecasting-lock.txt
```

lock 文件是首轮 Python 3.13 / macOS arm64 环境的版本快照，不保证跨平台或跨硬件逐位复现结果。

数据写入 `data/processed/`，模型与预测写入 `artifacts/forecasting/<run>/`，报告写入 `reports/forecasting/`。单次时间划分、单种子结果不等于最终验证；当前模型没有学习真实阀门动作的影响。

## 物理情景模拟

需要 Python 3.10+，模拟器只使用标准库，可独立于预测环境运行：

```sh
python3 scripts/simulate.py
python3 scripts/simulate.py --scenario configs/scenarios/demo.json
python3 -m unittest discover -s tests -p test_simulation.py -v
```

默认模拟 3 小时，比较固定 30%、固定 60% 和 CO₂ 比例反馈策略。结果写入 `data/synthetic/<run>/`，包含时间序列、摘要和完整配置/来源记录。

房间体积、送风量、风阀曲线等仍为明确假设。截图中的 8,168 L/s 只作为 AHU 总送风参考，不直接作为房间新风量。模拟结果不是实测控制效果，累计新风体积也不等于能耗或节能。

## InfluxDB 导出与核验

需要 Node.js 20+。在项目根目录运行：

```sh
npm ci --prefix influxdb
# 仅首次创建；已有 .env 不要覆盖
cp influxdb/.env.example influxdb/.env
# 在 .env 填入真实连接配置和 Token
npm --prefix influxdb run data:export
npm --prefix influxdb run data:verify -- data/raw/zb202_iot/<snapshot>
python3 scripts/validation/inspect-export.py data/raw/zb202_iot/<snapshot>
```

`<snapshot>` 替换为实际快照名，核验参数以项目根目录为基准。导出只读取 `zb202_iot`，自动写入项目的 `data/raw/zb202_iot/`。支持通过 `ZB202_ENV_FILE` 指定其他配置文件；真实凭据不提交 Git。

详细约定见 [数据说明](data/README.md)、[架构说明](docs/architecture.md) 和 [参数假设](docs/assumptions.md)。

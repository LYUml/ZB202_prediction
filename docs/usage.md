# 使用与复现

以下命令均在项目根目录执行。冻结模型和原始/处理数据必须在本机存在；`artifacts/` 与派生数据默认不随 Git 发布。

## 真实 CO₂ 预测

在项目根目录运行。首轮验证环境为 macOS arm64、Python 3.13；其他 Python 版本能否安装取决于 NumPy/PyTorch 对应发行包。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/forecasting.txt
# 仅准备数据
.venv/bin/python scripts/forecast.py prepare
# 训练、验证集早停，然后评估测试集
.venv/bin/python scripts/forecast.py train
# 诊断冻结的首轮运行（不训练）
.venv/bin/python scripts/forecast.py diagnose
# 输入修复后的固定预算滚动比较（持续值、Ridge、TCN、LSTM）
.venv/bin/python scripts/forecast.py benchmark
# 进一步开发试验（不再评价旧测试段）
.venv/bin/python scripts/forecast.py ablate
.venv/bin/python scripts/forecast.py combine
.venv/bin/python scripts/forecast.py refine
# 扩展候选（当前含NLinear/DLinear/缺测感知线性）
.venv/bin/python scripts/forecast.py compare
# 结构候选（固定开发比较，不读取新增确认段）
.venv/bin/python scripts/forecast.py compare --config configs/forecasting/structured.json
# 使用已有本地快照的最新完整历史箱，无需未来标签；output须是新文件
.venv/bin/python scripts/forecast.py predict --processed data/processed/zb202_iot/2026-10-20T09-22-56.494Z/causal-v1-a91fe5205b53 --output artifacts/forecasting/my-forecast.json
# 重现已完成的确认（该时段现已暴露；不是又获得一次独立测试）
.venv/bin/python scripts/forecast.py confirm --snapshot data/raw/zb202_iot/2026-10-20T09-22-56.494Z
# 完整检查
.venv/bin/python -m unittest discover -s tests -v
```

TCN 输入过去 4 小时，直接预测未来 1 小时的五个 CO₂ 点位。特征仅使用预测起点前已记录的观测；未来标签只取对应时间区间的真实 CO₂，不做前向填充。标准化仅使用训练时间段，验证集用于早停，测试集用于最终评估。默认使用safe-v2编码：二值事件不做方差放大，训练完全缺测特征显式禁用；首轮重放仍使用legacy-v1。比较协议与已暴露测试边界见 [评估协议](../docs/forecasting-validation.md)。

通用依赖文件给出兼容版本范围。若需复现首轮本机依赖版本，在兼容环境中改用：

```sh
.venv/bin/python -m pip install -r requirements/forecasting-lock.txt
```

lock 文件是首轮 Python 3.13 / macOS arm64 环境的版本快照，不保证跨平台或跨硬件逐位复现结果。

数据写入 `data/processed/`，模型与预测写入 `artifacts/forecasting/<run>/`，日常运行不生成Markdown报告，指标与来源记录写入对应运行目录的`manifest.json`。单次时间划分、单种子结果不等于最终验证；当前模型没有学习真实阀门动作的影响。

`predict`使用configs/forecasting/confirmation.json内已冻结权重及哈希，默认取快照导出时最后一个完整五分钟箱；`--origin-utc`可指定带时区的历史箱。模型读48×19输入，输出12×5ppm，掩码/年龄不是额外目标。缺少本地原始数据/权重时须先恢复或重跑相应基准；这些产物默认不随Git发布。当前最新示例在artifacts/forecasting/frozen-tcn-latest-20261020.json。

## 物理情景模拟

需要 Python 3.10+，模拟器只使用标准库，可独立于预测环境运行：

```sh
python3 scripts/simulate.py
python3 scripts/simulate.py --scenario configs/scenarios/demo.json
python3 -m unittest discover -s tests -p test_simulation.py -v
```

默认模拟 3 小时，按用户确认的BA基准比较固定20%/30%/40%和范围内CO₂反馈，风机固定60%。风阀20–40是用户授权的模拟假设，不是真实设备写入。结果写入 `data/synthetic/<run>/`，包含时间序列、摘要和完整配置/来源记录。

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

详细约定见 [数据说明](../data/README.md)、[架构说明](../docs/architecture.md) 和 [参数假设](../docs/assumptions.md)。

## 输出与汇报约定

训练、比较、消融、组合、预算检查和确认命令只保存本地实验产物，不自动写Markdown报告。对应Python入口返回`(运行目录, manifest.json路径)`；命令行会打印结果路径。`reports/`仅按用户要求维护`YYYY-MM-DD report.md`阶段汇报，同一阶段优先更新已有文件。

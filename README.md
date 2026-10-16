# ZB202 Prediction

ZB202 房间预测与模拟项目。目前完成 `zb202_iot` 历史数据导出和完整性核验，尚未执行建模预处理或训练预测模型。

## 目录

```text
ZB202_prediction/
├── data/
│   ├── raw/zb202_iot/<snapshot>/    # 原始快照及其来源、校验元数据
│   └── processed/zb202_iot/        # 后续清洗、对齐及特征处理结果
├── scripts/
│   ├── ingestion/                 # 从 InfluxDB 导出
│   └── validation/                # 服务器计数与本地完整性核验
├── .env.example                   # 连接配置模板
└── package.json                   # 依赖及命令入口
```

当前原始快照包含 949,752 条字段观测记录。压缩原始数据已纳入 Git，clone 后即可读取。数据约定见 [data/README.md](data/README.md)。

## 使用

需要 Node.js 20+ 和 Python 3。在项目根目录运行：

```sh
npm ci
cp .env.example .env
# 在 .env 填写真实连接配置和 Token
npm run data:export
npm run data:verify -- data/raw/zb202_iot/<snapshot>
npm run data:inspect -- data/raw/zb202_iot/<snapshot>
```

`<snapshot>` 替换为实际快照文件夹名。导出时自动创建新快照，不覆盖旧数据。核验可用于已有快照，无需重新导出。

也可通过 `ZB202_ENV_FILE` 指定现有 `.env`。凭据不提交到 Git。脚本只读取 `zb202_iot`，不操作设备。

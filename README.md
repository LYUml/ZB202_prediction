# ZB202 Prediction

ZB202 房间预测与模拟项目。目前包含 `zb202_iot` 原始历史快照和导出、核验工具，尚未训练预测模型。

## 数据

见 [data/README.md](data/README.md)。当前快照包含 949,752 条字段观测记录，原始 CSV 使用 gzip 压缩，已纳入 Git。

## 重新导出

需要 Node.js 20+ 和 Python 3。

```sh
npm ci
cp .env.example .env
# 在 .env 填写真实连接配置和 Token
npm run export
node scripts/verify-influx-counts.mjs data/<新快照目录>
python3 scripts/inspect-export.py data/<新快照目录>
```

也可通过 `ZB202_ENV_FILE` 指定现有 .env 的路径。凭据不提交到 Git。脚本仅导出 `zb202_iot`，不操作设备。

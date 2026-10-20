"""Single frozen chronological split, validation early stopping, then one test pass."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SENSORS, digest, iso, prepare, sample_origins
from .tcn import TCN
from .features import build_inputs
from .windows import Windows
from .evaluation import predict, scores


def train(project_root, config_path):
    root = Path(project_root)
    config = json.loads(Path(config_path).read_text())
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    processed = prepare(root, config)
    data = dict(np.load(processed / 'aligned.npz'))
    processed_manifest = json.loads((processed / 'manifest.json').read_text())
    names = processed_manifest['feature_names']
    origins, train_end, val_end = sample_origins(data, names, config)
    if any(len(v) == 0 for v in origins.values()):
        raise ValueError('Insufficient eligible samples in one or more chronological splits')
    x, current, y, normalization = build_inputs(data, names, train_end)
    loaders = {key: DataLoader(Windows(x, current, y, idx, config), batch_size=config['batch_size'],
                              shuffle=key == 'train', num_workers=0) for key, idx in origins.items()}
    model = TCN(x.shape[1], len(SENSORS), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    mean, scale = np.array(normalization['target_mean']), np.array(normalization['target_scale'])
    best, stale, best_state, history = float('inf'), 0, None, []
    for epoch in range(config['epochs']):
        model.train()
        total, count = 0, 0
        for inputs, latest, target, mask in loaders['train']:
            prediction = model(inputs, latest)
            loss = torch.abs(prediction - target)[mask].mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * mask.sum().item(); count += mask.sum().item()
        vp, vy, vm, _ = predict(model, loaders['validation'], mean, scale)
        value = scores(vp, vy, vm)['mae_ppm']
        history.append({'epoch': epoch + 1, 'train_normalized_mae': total / count, 'validation_mae_ppm': value})
        print(f"Epoch {epoch+1}: validation MAE {value:.2f} ppm", flush=True)
        if value < best:
            best, stale, best_state = value, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config['patience']:
            break
    if best_state is None:
        raise ValueError('No finite validation checkpoint')
    model.load_state_dict(best_state)
    # Test is accessed for metrics only after model selection is finished.
    pred, truth, mask, baseline = predict(model, loaders['test'], mean, scale)
    metrics = {'overall': {'tcn': scores(pred, truth, mask), 'persistence': scores(baseline, truth, mask)},
               'by_horizon': {}, 'by_sensor': {}}
    for step in [2, 6, 12]:
        if step > config['horizon_steps']:
            continue
        j = step - 1
        metrics['by_horizon'][str(step * config['step_seconds'] // 60)] = {
            'tcn': scores(pred[:, j], truth[:, j], mask[:, j]),
            'persistence': scores(baseline[:, j], truth[:, j], mask[:, j])}
    for j, sensor in enumerate(SENSORS):
        metrics['by_sensor'][sensor] = {'tcn': scores(pred[:, :, j], truth[:, :, j], mask[:, :, j]),
                                         'persistence': scores(baseline[:, :, j], truth[:, :, j], mask[:, :, j])}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    output = root / 'artifacts/forecasting' / run_id
    output.mkdir(parents=True)
    torch.save({'state_dict': best_state, 'config': config, 'input_features': x.shape[1],
                'target_names': SENSORS, 'feature_names': names, 'normalization': normalization}, output / 'tcn.pt')
    np.savez_compressed(output / 'test_predictions.npz', origins=data['timestamps'][origins['test']],
                        target_bin_end_utc=data['timestamps'][origins['test'][:, None] + np.arange(1, config['horizon_steps'] + 1)],
                        predictions_ppm=pred, actual_ppm=np.where(mask, truth, np.nan),
                        observed_mask=mask, persistence_ppm=baseline)
    manifest = {'status': 'first-single-split-experiment', 'run_id': run_id, 'source': 'real-observations',
                'config': config, 'config_sha256': digest(config_path), 'processed': str(processed.relative_to(root)),
                'processed_manifest_sha256': digest(processed / 'manifest.json'), 'raw_sha256': processed_manifest['raw_sha256'],
                'normalization': normalization, 'model_parameters': sum(p.numel() for p in model.parameters()),
                'split_boundaries': {'validation_start_utc': iso(data['timestamps'][train_end]), 'test_start_utc': iso(data['timestamps'][val_end])},
                'splits': {k: {'forecast_origins': len(idx), 'first_origin_utc': iso(data['timestamps'][idx[0]]),
                                'last_origin_utc': iso(data['timestamps'][idx[-1]])} for k, idx in origins.items()},
                'history': history, 'metrics': metrics, 'device': 'cpu', 'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__},
                'code_sha256': {str(p.relative_to(root)): digest(p) for p in sorted((root / 'src/zb202/forecasting').glob('*.py'))},
                'outputs': {p.name: digest(p) for p in output.iterdir()},
                'limitations': ['One seed, one chronological split; not rolling-origin validation or external validation.',
                                'Overlapping forecast windows are correlated, not independent test examples.',
                                'No BA inputs; no inference about effects of new damper actions.',
                                'Target timestamps denote right-closed bins, not exactly synchronized sensor readings.',
                                'Event freshness limits are modelling assumptions; no verified people counts.',
                                'Arrival timestamps unavailable; causality checked against stored measurement timestamps only.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = root / 'reports/forecasting' / (run_id + '.md')
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# 首轮真实 CO₂ 预测实验', '', '这是单次时间划分、单随机种子的实验，不是最终模型选择或控制验证。', '',
             f'运行 ID：`{run_id}`；模型参数数：{manifest["model_parameters"]}；设备：CPU。', '',
             f'原始 SHA-256：`{manifest["raw_sha256"]}`', '',
             '## 数据与划分', '', '| 集合 | 预测起点数 | 首个起点（UTC） | 最后起点（UTC） |', '| --- | ---: | --- | --- |']
    for key, entry in manifest['splits'].items():
        lines.append(f'| {key} | {entry["forecast_origins"]} | {entry["first_origin_utc"]} | {entry["last_origin_utc"]} |')
    lines += ['', '起点间隔 5 分钟；输入 4 小时，预测 1 小时。样本仅选当前五个 CO₂ 输入均有效、历史覆盖足够的时刻；未来标签只取真实观测。重叠窗口不独立，结果仅适用于该筛选范围。', '',
              '## 测试结果', '', '| 预测时距 | 真实标签数 | TCN MAE（ppm） | 持续值 MAE（ppm） | TCN RMSE（ppm） |', '| --- | ---: | ---: | ---: | ---: |']
    for minute, result in metrics['by_horizon'].items():
        a, b = result['tcn'], result['persistence']
        lines.append(f'| {minute} 分钟 | {a["observed_labels"]} | {a["mae_ppm"]:.2f} | {b["mae_ppm"]:.2f} | {a["rmse_ppm"]:.2f} |')
    lines += ['', '## 点位（全部预测时距汇总）', '', '| 点位 | TCN MAE（ppm） | 持续值 MAE（ppm） |', '| --- | ---: | ---: |']
    for sensor, result in metrics['by_sensor'].items():
        lines.append(f'| {sensor} | {result["tcn"]["mae_ppm"]:.2f} | {result["persistence"]["mae_ppm"]:.2f} |')
    lines += ['', '## 限制与后续', '', '未测试 LSTM / PIAD-SRNN、滚动留出、多种随机种子、占用阶段及高值片段表现；不根据本次测试继续调参后仍声称它是未见测试集。', '',
              '权重、原始单位预测与可追溯 manifest 在 `artifacts/forecasting/' + run_id + '/`，生成产物默认不提交 Git。处理结果在 `' + str(processed.relative_to(root)) + '`。', '',
              '模型未接入 BA，也未用于改变阀门后的反事实预测。']
    report.write_text('\n'.join(lines) + '\n')
    print('Report:', report, flush=True)
    return output, report

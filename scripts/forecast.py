"""Prepare, train, audit frozen forecasts, or run a fixed rolling comparison."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'diagnose', 'benchmark'])
    parser.add_argument('--config', default='configs/forecasting/tcn.json')
    parser.add_argument('--run', default='artifacts/forecasting/20261018T131219Z-d95c977c')
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if args.action == 'prepare':
        from zb202.forecasting.data import prepare
        print(prepare(ROOT, json.loads(path.read_text())))
    elif args.action == 'diagnose':
        from zb202.forecasting.diagnostics import diagnose
        print(diagnose(ROOT, ROOT / args.run))
    elif args.action == 'benchmark':
        from zb202.forecasting.benchmark import benchmark
        benchmark(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT / 'configs/forecasting/benchmark.json')
    else:
        from zb202.forecasting.train import train
        train(ROOT, path)

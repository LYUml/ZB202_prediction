"""Prepare real observations or train/evaluate a causal TCN."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train'])
    parser.add_argument('--config', default='configs/forecasting/tcn.json')
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if args.action == 'prepare':
        from zb202.forecasting.data import prepare
        print(prepare(ROOT, json.loads(path.read_text())))
    else:
        from zb202.forecasting.train import train
        train(ROOT, path)

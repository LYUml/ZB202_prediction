"""Prepare, train, audit frozen forecasts, or run a fixed rolling comparison."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'diagnose', 'benchmark', 'ablate', 'combine', 'refine', 'compare', 'confirm', 'predict'])
    parser.add_argument('--config', default='configs/forecasting/tcn.json')
    parser.add_argument('--run', default='artifacts/forecasting/20261018T131219Z-d95c977c')
    parser.add_argument('--snapshot', help='New raw snapshot, relative to project root; required for confirm')
    parser.add_argument('--after-utc', help='Confirmation-only label-period cutoff with explicit timezone; cannot precede frozen cutoff')
    parser.add_argument('--processed', help='Aligned processed directory; required for predict')
    parser.add_argument('--output', help='New output JSON path; required for predict')
    parser.add_argument('--origin-utc', help='Optional aligned historical origin with explicit UTC offset')
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if args.action == 'predict':
        if not args.processed or not args.output:
            parser.error('predict requires --processed and --output')
        from zb202.forecasting.inference import forecast_latest
        forecast_latest(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT/'configs/forecasting/confirmation.json',
                        ROOT/args.processed, ROOT/args.output, args.origin_utc)
    elif args.action == 'confirm':
        if not args.snapshot:
            parser.error('confirm requires --snapshot')
        from zb202.forecasting.confirmation import confirm
        confirm(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT/'configs/forecasting/confirmation.json', args.snapshot, args.after_utc)
    elif args.action == 'prepare':
        from zb202.forecasting.data import prepare
        print(prepare(ROOT, json.loads(path.read_text())))
    elif args.action == 'diagnose':
        from zb202.forecasting.diagnostics import diagnose
        print(diagnose(ROOT, ROOT / args.run))
    elif args.action == 'benchmark':
        from zb202.forecasting.benchmark import benchmark
        benchmark(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT / 'configs/forecasting/benchmark.json')
    elif args.action == 'ablate':
        from zb202.forecasting.ablation import ablate
        ablate(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT / 'configs/forecasting/ablation.json')
    elif args.action == 'combine':
        from zb202.forecasting.ensemble import combine
        combine(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT / 'configs/forecasting/ensemble.json')
    elif args.action == 'refine':
        from zb202.forecasting.refinement import refine
        refine(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT / 'configs/forecasting/refinement.json')
    elif args.action == 'compare':
        from zb202.forecasting.candidates import compare_candidates
        compare_candidates(ROOT, path if args.config != 'configs/forecasting/tcn.json' else ROOT / 'configs/forecasting/candidates.json')
    else:
        from zb202.forecasting.train import train
        train(ROOT, path)

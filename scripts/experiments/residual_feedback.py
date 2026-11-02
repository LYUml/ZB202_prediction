"""Chronological, label-delay-aware feedback experiment for saved forecasts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from zb202.forecasting.benchmark import summarize


def corrected_predictions(origins, target_times, predictions, actual, observed, rate, strength):
    """Update each horizon/point only with outcomes available by the forecast origin.

    Same-bin outcomes are allowed once their recorded bin end has passed. Each
    queued residual is from the original forecast, preventing recursive fitting.
    """
    origins = np.asarray(origins)
    if np.any(np.diff(origins) <= 0):
        raise ValueError('origins must be strictly increasing')
    if not (0 < rate <= 1 and 0 <= strength <= 1):
        raise ValueError('invalid feedback parameters')
    if target_times.shape != predictions.shape[:2] or actual.shape != predictions.shape or observed.shape != predictions.shape:
        raise ValueError('incompatible shapes')
    if np.any(target_times <= origins[:, None]):
        raise ValueError('target must follow origin')
    result = np.empty_like(predictions, dtype=np.float64)
    bias = np.zeros(predictions.shape[1:], dtype=np.float64)
    # A min-heap handles gaps and differing horizons without peeking ahead.
    import heapq
    events = []
    for i, now in enumerate(origins):
        while events and events[0][0] <= now:
            _, _, h, j, error = heapq.heappop(events)
            bias[h, j] += rate * (error - bias[h, j])
        result[i] = predictions[i] + strength * bias
        for h in range(predictions.shape[1]):
            for j in range(predictions.shape[2]):
                if observed[i, h, j]:
                    heapq.heappush(events, (target_times[i, h], i, h, j,
                                            float(actual[i, h, j] - predictions[i, h, j])))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('confirmation', type=Path)
    args = p.parse_args()
    source = args.confirmation / 'predictions.npz'
    manifest = json.loads((args.confirmation / 'manifest.json').read_text())
    z = np.load(source)
    origins, times = z['origins'], z['target_bin_end_utc']
    pred, actual, mask, baseline = (z[k] for k in ('predictions_ppm', 'actual_ppm', 'observed_mask', 'persistence_ppm'))
    cutoff = origins[int(len(origins) * .7)]
    dev = times[:, -1] < cutoff
    test = origins >= cutoff
    if not dev.any() or not test.any():
        raise ValueError('empty chronological split')
    records, predictions = [], {}
    for rate in (.05, .2):
        for strength in (0., .25, .5, 1.):
            out = corrected_predictions(origins, times, pred, actual, mask, rate, strength)
            key = f'{rate}-{strength}'
            predictions[key] = out
            records.append({'rate': rate, 'strength': strength,
                            'development': summarize(out[dev], actual[dev], mask[dev], baseline[dev])})
    best = min(records, key=lambda x: x['development']['mae_ppm'])
    chosen = predictions[f"{best['rate']}-{best['strength']}"]
    run = ROOT / 'artifacts/forecasting' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-feedback-' + uuid4().hex[:8])
    run.mkdir(parents=True)
    record = {'status': 'complete', 'source': str(args.confirmation), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'source_manifest_sha256': hashlib.sha256((args.confirmation / 'manifest.json').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cutoff_utc': datetime.fromtimestamp(float(cutoff), timezone.utc).isoformat(),
              'development_origins': int(dev.sum()), 'holdout_origins': int(test.sum()),
              'grid': records, 'selected': {'rate': best['rate'], 'strength': best['strength']},
              'holdout': {'feedback': summarize(chosen[test], actual[test], mask[test], baseline[test]),
                          'frozen': summarize(pred[test], actual[test], mask[test], baseline[test]),
                          'persistence': summarize(baseline[test], actual[test], mask[test], baseline[test])},
              'limitations': ['Retrospective forecasts, not deployed prospective feedback.',
                              'Observed timestamp assumed available at bin end; actual ingestion times unknown.',
                              'Overlapping windows correlated; all raw labels retained.',
                              'The frozen model has already been evaluated on this holdout period.']}
    np.savez_compressed(run / 'predictions.npz', origins=origins[test], target_bin_end_utc=times[test],
                        feedback_ppm=chosen[test], frozen_ppm=pred[test], actual_ppm=actual[test],
                        observed_mask=mask[test], persistence_ppm=baseline[test])
    record['predictions_sha256'] = hashlib.sha256((run / 'predictions.npz').read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(run)
    print(json.dumps({'selected': record['selected'], 'holdout_mae': {k:v['mae_ppm'] for k,v in record['holdout'].items()}}, indent=2))


if __name__ == '__main__':
    main()

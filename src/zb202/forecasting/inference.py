"""Frozen checkpoint ensemble inference using history only and training statistics."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import torch
from .ablation import feature_columns
from .benchmark import build_model
from .data import SENSORS, digest
from .features import encode_inputs


class FrozenForecaster:
    """Checkpoint paths, hashes and original feature order define the input contract.

    predict accepts causally aligned values/ages/timestamps, never needs labels,
    and rejects origins with insufficient historical coverage or stale current CO2.
    """
    def __init__(self, checkpoints, feature_names, expected_hashes):
        if not checkpoints or len(checkpoints) != len(expected_hashes):
            raise ValueError('Nonempty checkpoints and matching hashes required')
        self.names = list(feature_names)
        self.models = []
        for path, expected in zip(checkpoints, expected_hashes):
            if digest(path) != expected:
                raise ValueError('Checkpoint hash mismatch')
            ck = torch.load(Path(path), map_location='cpu', weights_only=True)
            if ck['normalization'].get('encoding') != 'safe-v2':
                raise ValueError('Frozen inference requires safe-v2 encoding')
            cfg, norm = ck['config'], ck['normalization']
            if 'feature_names' in ck and list(ck['feature_names']) != self.names:
                raise ValueError('Checkpoint feature names/order mismatch')
            cols = ck.get('feature_columns')
            if cols is None:
                cols = (list(range(len(self.names)*3+4)) if ck['feature_set'] == 'all'
                        else feature_columns(self.names, ck['feature_set']))
            if len(cols) != ck['input_features']:
                raise ValueError('Input schema mismatch')
            if self.models:
                if cfg != self.cfg or norm != self.norm or cols != self.cols:
                    raise ValueError('Ensemble members have different contracts')
            else:
                self.cfg, self.norm, self.cols = cfg, norm, cols
            model = build_model(ck['model'], ck['input_features'], len(SENSORS), cfg, norm)
            model.load_state_dict(ck['state_dict']); model.eval()
            self.models.append(model)

    def predict(self, data, feature_names, origins, batch_size=128):
        if list(feature_names) != self.names:
            raise ValueError('Feature names/order differ from frozen schema')
        origins = np.asarray(origins)
        if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer) or not len(origins) or batch_size < 1:
            raise ValueError('Nonempty integer origins and positive batch size required')
        length = self.cfg['lookback_steps']
        if origins.min() < length-1 or origins.max() >= len(data['timestamps']):
            raise ValueError('Origin outside historical context')
        if np.any(np.diff(data['timestamps']) != self.cfg['step_seconds']):
            raise ValueError('Input grid does not match frozen step')
        x, current = encode_inputs(data, self.names, self.norm)
        x = x[:, self.cols]
        indices = origins[:, None]+np.arange(-length+1, 1)
        co2 = [self.names.index(s+'/co2_ppm') for s in SENSORS]
        history_values = data['values'][indices][:, :, co2]
        history_ages = data['ages'][indices][:, :, co2]
        inconsistent = np.isfinite(history_values) & ((history_ages < 0) |
                       ~np.isfinite(history_ages) | (history_ages > self.cfg['environment_max_age_seconds']))
        if inconsistent.any():
            raise ValueError('CO2 values violate frozen causal freshness contract')
        coverage = np.isfinite(history_values).mean(axis=(1, 2))
        if np.any(coverage < self.cfg['minimum_co2_input_coverage']) or not np.isfinite(current[origins]).all():
            raise ValueError('Insufficient current/history CO2 availability')
        if not np.isfinite(x[indices]).all():
            raise ValueError('Nonfinite encoded historical inputs')
        mean, scale = np.asarray(self.norm['target_mean']), np.asarray(self.norm['target_scale'])
        predictions = []
        with torch.no_grad():
            for start in range(0, len(origins), batch_size):
                ix = indices[start:start+batch_size]
                inputs = torch.from_numpy(x[ix])
                latest = torch.from_numpy(current[origins[start:start+batch_size]])
                # Convert each member to ppm before averaging, matching saved benchmarks.
                members = [model(inputs, latest).numpy()*scale+mean for model in self.models]
                predictions.append(np.mean(members, axis=0))
        result = np.concatenate(predictions)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite frozen predictions')
        return result


def forecast_latest(root, protocol_path, processed, output, origin_utc=None):
    """Write one forecast window; requires no future labels or training run."""
    root, processed, output = Path(root), Path(processed), Path(output)
    protocol = json.loads(Path(protocol_path).read_text())
    base = json.loads((root/protocol['reference_run']/'manifest.json').read_text())
    schema_path = root/base['protocol']['processed']/'manifest.json'
    if digest(schema_path) != base['processed_manifest_sha256']:
        raise ValueError('Frozen schema manifest mismatch')
    names = json.loads(schema_path.read_text())['feature_names']
    manifest = json.loads((processed/'manifest.json').read_text())
    if manifest['feature_names'] != names or digest(processed/'aligned.npz') != manifest['aligned_sha256']:
        raise ValueError('Processed schema or data hash mismatch')
    model = FrozenForecaster([root/p for p in protocol['models']], names,
                             [protocol['sha256'][p] for p in protocol['models']])
    for key in ['step_seconds', 'environment_max_age_seconds', 'event_max_age_seconds', 'power_max_age_seconds']:
        if manifest['config'][key] != model.cfg[key]:
            raise ValueError('Processed history rules differ from frozen contract: '+key)
    # Do not read target arrays. Last fully elapsed bin avoids an unfinished origin.
    with np.load(processed/'aligned.npz') as loaded:
        data = {k: loaded[k] for k in ['values', 'ages', 'timestamps']}
    if origin_utc is None:
        stop = json.loads((root/manifest['config']['snapshot']/'manifest.json').read_text())['stop']
        timestamp = np.floor(datetime.fromisoformat(stop.replace('Z', '+00:00')).timestamp()/model.cfg['step_seconds'])*model.cfg['step_seconds']
    else:
        parsed = datetime.fromisoformat(origin_utc.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('Origin requires an explicit UTC offset')
        timestamp = parsed.timestamp()
    matches = np.flatnonzero(data['timestamps'] == timestamp)
    if len(matches) != 1:
        raise ValueError('Origin is not an aligned historical bin')
    origin = int(matches[0])
    data = {k: v[:origin+1] for k, v in data.items()}
    torch.set_num_threads(4)
    prediction = model.predict(data, names, np.array([origin]))[0]
    result = dict(source='forecast-from-real-history', origin_utc=datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                  origin_unix_utc=int(timestamp), target_names=SENSORS,
                  target_bin_end_unix_utc=(timestamp+np.arange(1, model.cfg['horizon_steps']+1)*model.cfg['step_seconds']).astype(int).tolist(),
                  predictions_ppm=prediction.tolist(), checkpoint_sha256=protocol['sha256'],
                  aligned_sha256=manifest['aligned_sha256'], protocol_sha256=digest(protocol_path),
                  limitation='Historical-operation forecast, not changed-damper causal response. No actual future labels used.')
    if output.exists():
        raise ValueError('Refusing to overwrite an existing forecast artifact')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(output)
    return output

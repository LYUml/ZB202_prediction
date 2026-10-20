"""Frozen inference needs no labels and cannot normalize with future data."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from zb202.forecasting.data import SENSORS, digest
from zb202.forecasting.features import build_inputs, encode_inputs
from zb202.forecasting.benchmark import build_model
from zb202.forecasting.inference import FrozenForecaster


class FrozenInferenceTests(unittest.TestCase):
    def fixture(self, directory):
        names = [s+'/co2_ppm' for s in SENSORS]
        values = (450+np.arange(100)[:, None]+np.arange(5)[None, :]).astype(np.float32)
        data = dict(values=values, ages=np.zeros_like(values), targets=values.copy(), timestamps=np.arange(100)*300)
        _, _, _, norm = build_inputs(data, names, 60)
        cfg = dict(lookback_steps=8, horizon_steps=12, channels=4, kernel_size=3,
                   dilations=[1], dropout=0., step_seconds=300, minimum_co2_input_coverage=.8,
                   environment_max_age_seconds=900)
        model = build_model('tcn', 19, 5, cfg)
        # Nonzero head exercises model inference beyond the persistence anchor.
        torch.nn.init.normal_(model.head.weight, std=.01)
        path = Path(directory)/'model.pt'
        torch.save(dict(state_dict=model.state_dict(), model='tcn', config=cfg,
                        normalization=norm, feature_names=names, feature_set='co2_only', input_features=19), path)
        return names, data, FrozenForecaster([path], names, [digest(path)])

    def test_labels_and_future_values_cannot_change_current_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            names, data, model = self.fixture(directory)
            first = model.predict(data, names, np.array([70]))
            history = {k: v[:71].copy() for k, v in data.items() if k != 'targets'}
            np.testing.assert_array_equal(first, model.predict(history, names, np.array([70])))
            data['targets'][:] = 999999
            data['values'][71:] = 999999
            np.testing.assert_array_equal(first, model.predict(data, names, np.array([70])))

    def test_batching_schema_and_missing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            names, data, model = self.fixture(directory)
            together = model.predict(data, names, np.array([70, 71]), batch_size=2)
            alone = model.predict(data, names, np.array([70, 71]), batch_size=1)
            np.testing.assert_allclose(together, alone, atol=1e-4, rtol=1e-6)
            with self.assertRaises(ValueError): model.predict(data, names[::-1], np.array([70]))
            data['values'][70, 0] = np.nan
            with self.assertRaises(ValueError): model.predict(data, names, np.array([70]))
            data['values'][70, 0] = 500
            data['ages'][70, 0] = 901
            with self.assertRaises(ValueError): model.predict(data, names, np.array([70]))

    def test_frozen_encoding_matches_fitted_encoding_without_labels(self):
        names = [s+'/co2_ppm' for s in SENSORS]
        values = np.arange(500, dtype=np.float32).reshape(100, 5)
        data = dict(values=values, ages=np.zeros_like(values), targets=values.copy(), timestamps=np.arange(100)*300)
        x, current, _, norm = build_inputs(data, names, 60)
        del data['targets']
        frozen_x, frozen_current = encode_inputs(data, names, norm)
        np.testing.assert_array_equal(x, frozen_x)
        np.testing.assert_array_equal(current, frozen_current)


if __name__ == '__main__': unittest.main()

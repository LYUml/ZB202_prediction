"""Regression checks for train-unseen channels and fair chronological comparison."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from zb202.forecasting.data import SENSORS
from zb202.forecasting.features import build_inputs
from zb202.forecasting.benchmark import fold_origins, fit_neural
from zb202.forecasting.baselines import DirectLSTM, DirectRidge


class EncodingAndBenchmarkTests(unittest.TestCase):
    def fixture(self):
        names = [s+'/co2_ppm' for s in SENSORS] + ['VS341_04/last_occupancy_event', 'WS523_01/active_power']
        values = np.ones((100, 7), dtype=np.float32)*500
        values[:, 5] = np.nan
        values[60:, 5] = 1
        values[:, 6] = 0
        values[60:, 6] = 100
        return names, {'values': values, 'ages': np.zeros_like(values),
                       'targets': values[:, :5].copy(), 'timestamps': np.arange(100)*300}

    def test_unseen_future_feature_cannot_activate_any_channel(self):
        names, data = self.fixture()
        x, _, _, norm = build_inputs(data, names, 60)
        np.testing.assert_array_equal(x[:, [5, 12, 19]], 0)
        self.assertIn(names[5], norm['disabled_features'])
        legacy, _, _, _ = build_inputs(data, names, 60, 'legacy-v1')
        self.assertAlmostEqual(float(legacy[70, 5]), 1000)

    def test_binary_event_is_not_scaled_by_rarity(self):
        names, data = self.fixture()
        data['values'][:, 5] = 0
        data['values'][10, 5] = 1
        data['values'][70, 5] = 1
        x, _, _, _ = build_inputs(data, names, 60)
        self.assertEqual(x[70, 5], 1)
        self.assertEqual(x[9, 5], 0)

    def test_constant_continuous_future_change_is_frozen(self):
        names, data = self.fixture()
        x, _, _, norm = build_inputs(data, names, 60)
        np.testing.assert_array_equal(x[:, 6], 0)
        self.assertIn(names[6], norm['frozen_constant_values'])

    def test_changing_future_does_not_change_training_encoding(self):
        names, data = self.fixture()
        x1, _, _, norm1 = build_inputs(data, names, 60)
        data['values'][60:, :5] = 100000
        x2, _, _, norm2 = build_inputs(data, names, 60)
        np.testing.assert_array_equal(x1[:60], x2[:60])
        self.assertEqual(norm1['feature_mean'], norm2['feature_mean'])
        self.assertEqual(norm1['feature_scale'], norm2['feature_scale'])

    def test_integer_fold_labels_and_history(self):
        names, data = self.fixture()
        cfg = {'lookback_steps': 5, 'horizon_steps': 4, 'minimum_co2_input_coverage': .8}
        idx = fold_origins(data, names, cfg, [40, 60, 80])
        self.assertLess(idx['train'].max()+4, 40)
        self.assertGreaterEqual(idx['validation'].min(), 40)
        self.assertLess(idx['validation'].max()+4, 60)
        self.assertGreaterEqual(idx['test'].min(), 60)
        self.assertLess(idx['test'].max()+4, 80)

    def test_lstm_initial_persistence_and_shape(self):
        cfg = {'channels': 8, 'horizon_steps': 12}
        m = DirectLSTM(7, 5, cfg)
        latest = torch.randn(2, 5)
        torch.testing.assert_close(m(torch.randn(2, 48, 7), latest), latest[:, None].expand(2, 12, 5))

    def test_ridge_ignores_missing_labels(self):
        x = np.arange(10, dtype=float)[:, None]
        y = (2*x+3).reshape(10, 1, 1)
        mask = np.ones_like(y, dtype=bool); mask[-1] = False
        y[-1] = np.nan
        model = DirectRidge(.001).fit(x, y, mask)
        np.testing.assert_allclose(model.predict(x[:3], np.zeros((3, 1)))[:, 0, 0], [3, 5, 7], atol=.001)

    def test_outer_labels_cannot_change_checkpoint_selection(self):
        names, data = self.fixture()
        cfg = {'lookback_steps': 5, 'horizon_steps': 4, 'minimum_co2_input_coverage': .8,
               'channels': 4, 'kernel_size': 3, 'dilations': [1], 'dropout': 0.,
               'epochs': 2, 'patience': 1, 'batch_size': 16,
               'learning_rate': .001, 'weight_decay': .0001}
        origins = fold_origins(data, names, cfg, [40, 60, 80])
        x, current, y, norm = build_inputs(data, names, 40)
        first, _, history1 = fit_neural('tcn', x, current, y, origins, cfg, norm, 42)
        changed = y.copy(); changed[60:] += 10000
        second, _, history2 = fit_neural('tcn', x, current, changed, origins, cfg, norm, 42)
        self.assertEqual(history1, history2)
        for name, state in first.state_dict().items():
            torch.testing.assert_close(state, second.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

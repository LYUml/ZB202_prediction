"""Leakage, missing-label, normalization and temporal model checks."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zb202.forecasting.data import SENSORS, align_series, sample_origins
from zb202.forecasting.tcn import CausalConv, TCN
from zb202.forecasting.features import build_inputs


class ForecastingTests(unittest.TestCase):
    def test_future_reading_cannot_enter_past_feature(self):
        grid = np.array([300, 600, 900, 1200])
        features, age, labels = align_series(np.array([301, 599]), np.array([10., 20.]), grid, 900)
        self.assertTrue(np.isnan(features[0]))
        self.assertEqual(features[1], 20)
        self.assertEqual(features[2], 20)
        self.assertTrue(np.isnan(labels[2]))
        self.assertTrue(np.isnan(labels[3]))
        self.assertEqual(age[2], 301)

    def test_exact_boundary_observation_is_not_reused_as_future_label(self):
        _, _, labels = align_series(np.array([300., 600.]), np.array([400., 500.]), np.array([300, 600, 900]), 900)
        np.testing.assert_allclose(labels[:2], [400, 500])
        self.assertTrue(np.isnan(labels[2]))

    def test_all_labels_stay_inside_split(self):
        names = [s + '/co2_ppm' for s in SENSORS]
        data = {'timestamps': np.arange(200) * 300, 'values': np.ones((200, 5)), 'targets': np.ones((200, 5))}
        cfg = {'train_fraction': .6, 'validation_fraction': .2, 'lookback_steps': 10,
               'horizon_steps': 12, 'minimum_co2_input_coverage': .8}
        idx, train_end, val_end = sample_origins(data, names, cfg)
        self.assertLess(max(idx['train']) + 12, train_end)
        self.assertGreaterEqual(min(idx['validation']), train_end)
        self.assertLess(max(idx['validation']) + 12, val_end)
        self.assertGreaterEqual(min(idx['test']), val_end)
        self.assertLess(max(idx['test']) + 12, 200)

    def test_normalization_never_uses_future_extremes(self):
        values = np.ones((10, 5), dtype=np.float32) * 500
        values[6:] = 100000
        data = {'values': values, 'targets': values.copy(), 'ages': np.zeros_like(values), 'timestamps': np.arange(10) * 300}
        _, _, _, norm = build_inputs(data, [s + '/co2_ppm' for s in SENSORS], 6)
        self.assertEqual(norm['target_mean'], [500] * 5)

    def test_causal_convolution_ignores_future_changes(self):
        conv = CausalConv(2, 4, 3, 2)
        first = torch.randn(1, 2, 30)
        second = first.clone(); second[:, :, 16:] += 1000
        with torch.no_grad():
            torch.testing.assert_close(conv(first)[:, :, :16], conv(second)[:, :, :16])

    def test_direct_multi_horizon_head_shape_and_initial_persistence(self):
        cfg = {'channels': 8, 'kernel_size': 3, 'dilations': [1, 2], 'dropout': 0, 'horizon_steps': 12}
        model = TCN(7, 5, cfg)
        latest = torch.randn(2, 5)
        out = model(torch.randn(2, 48, 7), latest)
        self.assertEqual(tuple(out.shape), (2, 12, 5))
        torch.testing.assert_close(out, latest[:, None, :].expand(2, 12, 5))


if __name__ == '__main__':
    unittest.main()

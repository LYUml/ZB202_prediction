"""Guard the scientific contract of corruption and external-data evaluation."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from zb202.forecasting.robustness import (
    align_observations, corrupt_history, eligible_history, external_origins, single_inputs,
)


class RobustnessTests(unittest.TestCase):
    def test_external_right_closed_causal_alignment_and_unfinished_bin(self):
        data = align_observations([0, 299, 300, 301, 1800, 1810], [400, 410, 420, 999, 700, 800])
        self.assertEqual(data['targets'][1, 0], 420)  # future t301 cannot enter bin300
        self.assertEqual(data['targets'][2, 0], 999)
        self.assertTrue(np.isnan(data['targets'][3, 0]))  # no target filling
        self.assertTrue(np.isnan(data['values'][5, 0]))  # >900s stale
        self.assertEqual(data['timestamps'][-1], 1800)  # exclude unfinished bin2100

    def test_external_duplicate_timestamps_need_explicit_resolution(self):
        with self.assertRaises(ValueError):
            align_observations([0, 0, 300], [400, 401, 402])

    def test_corruptions_are_input_only_and_repeatable(self):
        data = dict(values=np.full((80, 5), 450., dtype=np.float32),
                    ages=np.zeros((80, 5), dtype=np.float32),
                    targets=np.full((80, 5), 451., dtype=np.float32), timestamps=np.arange(80)*300)
        scenario = dict(kind='noise', sigma_ppm=20)
        first = corrupt_history(data, list(range(5)), scenario, 42)
        second = corrupt_history(data, list(range(5)), scenario, 42)
        np.testing.assert_array_equal(first['values'], second['values'])
        np.testing.assert_array_equal(first['targets'], data['targets'])
        np.testing.assert_array_equal(data['values'], np.full((80, 5), 450.))
        cfg = dict(lookback_steps=48, environment_max_age_seconds=900, minimum_co2_input_coverage=.8)
        self.assertTrue(eligible_history(data, list(range(5)), np.array([60]), cfg)[0])
        outage = corrupt_history(data, list(range(5)), dict(kind='sensor_outage', sensor=4), 42)
        self.assertFalse(eligible_history(outage, list(range(5)), np.array([60]), cfg)[0])
        stale = corrupt_history(data, list(range(5)), dict(kind='stale_current', age_seconds=901), 42)
        self.assertFalse(eligible_history(stale, list(range(5)), np.array([60]), cfg)[0])

    def test_external_future_labels_never_change_training_statistics_or_partition(self):
        data = align_observations(np.arange(1500)*300, 400+np.arange(1500)%50)
        cfg = dict(lookback_steps=48, horizon_steps=12)
        origins, boundaries = external_origins(data, cfg, [.6, .75, 1.])
        first, current, _, norm = single_inputs(data, boundaries[0])
        changed = {k: v.copy() for k, v in data.items()}
        changed['targets'][boundaries[1]:] = 999999
        _, new_current, _, new_norm = single_inputs(changed, boundaries[0])
        self.assertEqual(norm, new_norm)
        np.testing.assert_array_equal(current, new_current)
        new_origins, _ = external_origins(changed, cfg, [.6, .75, 1.])
        for key in origins:
            np.testing.assert_array_equal(origins[key], new_origins[key])
        self.assertLess((origins['train']+12).max(), boundaries[0])
        self.assertGreaterEqual(origins['validation'].min(), boundaries[0])
        self.assertLess((origins['validation']+12).max(), boundaries[1])
        self.assertGreaterEqual(origins['test'].min(), boundaries[1])
        np.testing.assert_array_equal(first[:, 3:], 0)

    def test_future_external_input_shift_cannot_refit_normalization(self):
        data = align_observations(np.arange(200)*300, 400+np.arange(200)%50)
        x, _, _, norm = single_inputs(data, 100)
        changed = {k: v.copy() for k, v in data.items()}
        changed['values'][100:] += 10000
        shifted_x, _, _, shifted_norm = single_inputs(changed, 100)
        self.assertEqual(norm, shifted_norm)
        np.testing.assert_array_equal(x[:100], shifted_x[:100])

    def test_external_grid_gaps_never_stitch_into_complete_windows(self):
        times = np.r_[np.arange(600)*300, (np.arange(600)+3000)*300]
        data = align_observations(times, np.full(1200, 450.))
        cfg = dict(lookback_steps=48, horizon_steps=12)
        origins, _ = external_origins(data, cfg, [.6, .75, 1.])
        for selected in origins.values():
            for origin in selected:
                self.assertTrue(np.isfinite(data['targets'][origin+1:origin+13]).all())
                self.assertGreaterEqual(np.isfinite(data['values'][origin-47:origin+1]).mean(), .8)


if __name__ == '__main__':
    unittest.main()

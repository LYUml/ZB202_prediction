import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('residual_feedback', Path(__file__).resolve().parents[1] / 'scripts/experiments/residual_feedback.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FeedbackTests(unittest.TestCase):
    def test_delayed_labels_and_missing_outcomes(self):
        origins = np.array([0, 1, 2, 3])
        target = origins[:, None] + 2
        pred = np.zeros((4, 1, 1))
        actual = np.array([10., 100., 20., 30.]).reshape(4, 1, 1)
        observed = np.array([True, False, True, True]).reshape(4, 1, 1)
        result = module.corrected_predictions(origins, target, pred, actual, observed, 1., 1.)
        np.testing.assert_array_equal(result[:, 0, 0], [0, 0, 10, 10])

    def test_future_outcomes_cannot_change_past_predictions(self):
        origins = np.arange(8)
        target = origins[:, None] + 3
        pred = np.zeros((8, 1, 1))
        truth = np.ones_like(pred)
        mask = np.ones_like(pred, dtype=bool)
        first = module.corrected_predictions(origins, target, pred, truth, mask, .2, .5)
        truth[4:] = 999
        second = module.corrected_predictions(origins, target, pred, truth, mask, .2, .5)
        np.testing.assert_array_equal(first[:7], second[:7])

    def test_rejects_noncausal_targets(self):
        with self.assertRaises(ValueError):
            module.corrected_predictions(np.array([0]), np.array([[0]]), np.zeros((1, 1, 1)),
                                         np.zeros((1, 1, 1)), np.ones((1, 1, 1), bool), .2, .5)


if __name__ == '__main__':
    unittest.main()

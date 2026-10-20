"""Batch independence and physical structure, including after nonzero training."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from zb202.forecasting.benchmark import build_model


class StructuredForecastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.cfg = {'lookback_steps': 48, 'horizon_steps': 12}
        self.norm = {'target_mean': [450., 580., 450., 315., 440.],
                     'target_scale': [40., 50., 45., 30., 60.]}

    def test_time_recurrence_does_not_depend_on_other_batch_samples(self):
        model = build_model('decomp_srnn', 19, 5, self.cfg)
        # Nonzero heads prevent initial persistence from hiding a recurrence bug.
        torch.nn.init.normal_(model.seasonal_head.weight)
        torch.nn.init.normal_(model.trend_head.weight)
        x = torch.randn(3, 48, 19); current = x[:, -1, :5]
        with torch.no_grad():
            together = model(x, current)
            alone = model(x[1:2], current[1:2])
            reverse = model(x.flip(0), current.flip(0)).flip(0)
            changed = x.clone(); changed[0] += 100
            perturbed = model(changed, current)
        torch.testing.assert_close(together[1:2], alone)
        torch.testing.assert_close(together, reverse)
        torch.testing.assert_close(together[1:], perturbed[1:])
        self.assertEqual(tuple(together.shape), (3, 12, 5))

    def test_recurrence_uses_history_and_has_finite_training_gradients(self):
        model = build_model('decomp_srnn', 19, 5, self.cfg)
        torch.nn.init.normal_(model.seasonal_head.weight)
        x = torch.randn(2, 48, 19, requires_grad=True)
        model(x, x[:, -1, :5]).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[:, :-1].abs().sum()), 0.)

    def test_relaxation_stays_positive_and_monotone_after_large_head_updates(self):
        model = build_model('relaxation', 19, 5, self.cfg, self.norm)
        torch.nn.init.normal_(model.head.weight, std=10.)
        x = torch.randn(3, 48, 19)
        raw = torch.tensor([[0., 100., 500., 1500., 9500.]]).expand(3, -1)
        current = (raw-model.mean)/model.scale
        out = model(x, current); ppm = out*model.scale+model.mean
        self.assertTrue(torch.isfinite(ppm).all())
        self.assertGreaterEqual(float(ppm.detach().min()), -1e-3)
        increments = torch.diff(torch.cat((raw[:, None, :], ppm), dim=1), dim=1)
        self.assertTrue(((increments >= -1e-3).all(dim=1) |
                         (increments <= 1e-3).all(dim=1)).all())
        out.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

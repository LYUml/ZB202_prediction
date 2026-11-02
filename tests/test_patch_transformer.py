import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from zb202.forecasting.patch_transformer import PatchTransformer


class PatchTransformerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'lookback_steps':48, 'horizon_steps':12, 'dropout':0.0}

    def test_initial_output_is_persistence_and_gradients_exist(self):
        model = PatchTransformer(19, 5, self.cfg)
        x = torch.randn(2, 48, 19)
        current = torch.randn(2, 5)
        output = model(x, current)
        self.assertEqual(tuple(output.shape), (2, 12, 5))
        torch.testing.assert_close(output, current[:, None, :].expand_as(output))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())

    def test_rejects_invalid_geometry_and_schema(self):
        with self.assertRaises(ValueError):
            PatchTransformer(18, 5, self.cfg)
        with self.assertRaises(ValueError):
            PatchTransformer(19, 5, dict(self.cfg, patch_length=7))


if __name__ == '__main__':
    unittest.main()

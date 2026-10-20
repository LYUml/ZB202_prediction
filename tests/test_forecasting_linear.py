"""Window-only decomposition, translation behavior, and missing-information contracts."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zb202.forecasting.linear import NLinear,DLinear,MaskAwareLinear


class LinearForecastTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'lookback_steps':48,'horizon_steps':12,'decomposition_kernel':25}

    def test_nlinear_is_translation_equivariant_and_ignores_auxiliary_columns(self):
        model=NLinear(19,5,self.cfg)
        x=torch.randn(2,48,19);current=x[:,-1,:5].clone()
        changed=x.clone();changed[:,:,:5]+=100;changed[:,:,5:]+=10000
        with torch.no_grad(): first=model(x,current);second=model(changed,current+100)
        torch.testing.assert_close(second,first+100,atol=2e-5,rtol=1e-5)

    def test_dlinear_decomposition_reconstructs_last_input(self):
        model=DLinear(19,5,self.cfg)
        for layer in [model.trend,model.residual]:
            torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
            with torch.no_grad():layer.weight[:,-1]=1
        x=torch.randn(2,48,19);current=x[:,-1,:5].clone()
        torch.testing.assert_close(model(x,current),current[:,None].expand(2,12,5))
        with self.assertRaises(ValueError):DLinear(19,5,dict(self.cfg,decomposition_kernel=4))

    def test_mask_aware_initial_persistence_and_auxiliary_gradient(self):
        model=MaskAwareLinear(19,5,self.cfg)
        x=torch.randn(2,48,19);current=x[:,-1,:5].clone()
        out=model(x,current)
        torch.testing.assert_close(out,current[:,None].expand(2,12,5))
        out.sum().backward()
        self.assertGreater(float(model.map.weight.grad.abs().sum()),0)
        self.assertEqual(tuple(out.shape),(2,12,5))


if __name__=='__main__':unittest.main()

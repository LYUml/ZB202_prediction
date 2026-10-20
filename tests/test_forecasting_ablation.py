"""Feature inclusion and sensor-isolation contracts for development experiments."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zb202.forecasting.data import SENSORS
from zb202.forecasting.ablation import feature_columns
from zb202.forecasting.tcn import IndependentTCN


class AblationTests(unittest.TestCase):
    def test_feature_groups_add_only_the_declared_measurements(self):
        names=sorted([s+'/co2_ppm' for s in SENSORS]+[s+'/temperature_c' for s in SENSORS]+
                     ['VS341_01/last_occupancy_event','WS301_01/last_magnet_event','WS523_01/active_power'])
        for group,suffix in [('environment','/temperature_c'),('occupancy','/last_occupancy_event'),
                             ('door','/last_magnet_event'),('power','/active_power')]:
            cols=feature_columns(names,group); n=len(names)
            value_cols=[j for j in cols if j<n]
            selected=[names[j] for j in value_cols]
            self.assertEqual(selected[:5],[s+'/co2_ppm' for s in SENSORS])
            self.assertTrue(all(name.endswith(suffix) for name in selected[5:]))
            self.assertEqual(cols[len(value_cols):2*len(value_cols)],[n+j for j in value_cols])
            self.assertEqual(cols[-4:],list(range(3*n,3*n+4)))
        with self.assertRaises(ValueError): feature_columns(names,'unknown')

    def test_local_network_cannot_read_other_sensor_inputs(self):
        torch.manual_seed(4)
        cfg={'channels':8,'kernel_size':3,'dilations':[1,2],'dropout':0.,'horizon_steps':12}
        model=IndependentTCN(19,5,cfg).eval()
        for net in model.models:
            torch.nn.init.normal_(net.head.weight,std=.1)
        x=torch.randn(2,48,19); current=torch.randn(2,5)
        altered=x.clone(); altered[:,:,[0,5,10]]+=100
        changed_current=current.clone(); changed_current[:,0]+=1000
        with torch.no_grad():
            first=model(x,current); second=model(altered,changed_current)
        torch.testing.assert_close(first[:,:,1:],second[:,:,1:],rtol=0,atol=0)
        self.assertFalse(torch.equal(first[:,:,0],second[:,:,0]))
        self.assertEqual(tuple(first.shape),(2,12,5))
        self.assertIsNot(model.models[0].head.weight,model.models[1].head.weight)

    def test_local_initialization_is_persistence_and_rejects_mixed_inputs(self):
        cfg={'channels':8,'kernel_size':3,'dilations':[1],'dropout':0.,'horizon_steps':12}
        model=IndependentTCN(19,5,cfg); current=torch.randn(2,5)
        torch.testing.assert_close(model(torch.randn(2,48,19),current),current[:,None].expand(2,12,5))
        with self.assertRaises(ValueError): IndependentTCN(100,5,cfg)



class CombinationTests(unittest.TestCase):
    def test_missing_calibration_labels_do_not_affect_weight(self):
        from zb202.forecasting.ensemble import choose_weight
        tcn=np.array([0.,999.]); ridge=np.array([10.,-999.])
        mask=np.array([True,False])
        first,_=choose_weight(tcn,ridge,np.array([5.,np.nan]),mask,[0.,.5,1.])
        second,_=choose_weight(tcn,ridge,np.array([5.,1e9]),mask,[0.,.5,1.])
        self.assertEqual(first,.5);self.assertEqual(first,second)

    def test_calibration_rejects_empty_truth_or_extrapolating_weights(self):
        from zb202.forecasting.ensemble import choose_weight
        x=np.zeros(2)
        with self.assertRaises(ValueError): choose_weight(x,x,x,np.zeros(2,dtype=bool),[0.,1.])
        with self.assertRaises(ValueError): choose_weight(x,x,x,np.ones(2,dtype=bool),[-.1,1.1])


class RefinementTests(unittest.TestCase):
    def test_epoch_cap_does_not_refit_models_already_early_stopped(self):
        from zb202.forecasting.refinement import hit_epoch_cap
        def entry(losses):
            return {'history':[{'epoch':i+1,'inner_validation_mae_ppm':v} for i,v in enumerate(losses)]}
        self.assertTrue(hit_epoch_cap(entry([3.,2.,1.]),3,2))
        self.assertFalse(hit_epoch_cap(entry([1.,2.,3.]),3,2))
        self.assertFalse(hit_epoch_cap(entry([1.,2.,3.]),5,2))
        self.assertTrue(hit_epoch_cap(entry([3.,1.,2.]),3,2))
        with self.assertRaises(ValueError): hit_epoch_cap(entry([]),3,2)


if __name__=='__main__': unittest.main()

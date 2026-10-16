"""Physical invariants and control behaviour, independent of calibration."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zb202.models.co2 import advance
from zb202.models.ahu import actuator, outdoor_airflow
from zb202.simulation import simulate

ROOT = Path(__file__).resolve().parents[1]


class PhysicsTests(unittest.TestCase):
    def test_unoccupied_room_decays_towards_outdoor(self):
        c = advance(1000, 420, 300, 0, 0.000005, 0.1, 600)
        self.assertGreater(c, 420)
        self.assertLess(c, 1000)
        self.assertEqual(advance(420, 420, 300, 0, 0, 0.1, 600), 420)

    def test_no_flow_mass_conservation(self):
        self.assertAlmostEqual(advance(420, 420, 300, 10, 0.000005, 0, 600), 520)

    def test_more_fresh_air_lowers_co2_above_outdoor(self):
        args = (1000, 420, 300, 8, 0.000005)
        self.assertLess(advance(*args, 0.2, 600), advance(*args, 0.05, 600))

    def test_actuator_delay_and_flow_limits(self):
        self.assertTrue(11 < actuator(11, 30, 120, 60) < 30)
        self.assertAlmostEqual(outdoor_airflow(100, 0.5, 0.02, 0.6, 1.5), 0.3)
        with self.assertRaises(ValueError):
            actuator(11, 130, 120, 60)

    def test_scenarios_have_consistent_intervals_and_response(self):
        configs = [json.loads((ROOT / p).read_text()) for p in ['configs/room.json', 'configs/ahu.json', 'configs/scenarios/demo.json']]
        low = simulate(*configs, configs[2]['strategies'][0])
        high = simulate(*configs, configs[2]['strategies'][1])
        self.assertEqual(len(low), 180)
        self.assertEqual(low[29]['occupants_assumed'], 0)
        self.assertEqual(low[30]['occupants_assumed'], 8)
        self.assertEqual(low[-1]['interval_end_s'], 10800)
        self.assertLess(high[119]['co2_end_ppm'], low[119]['co2_end_ppm'])
        for before, after in zip(low, low[1:]):
            self.assertEqual(before['co2_end_ppm'], after['co2_start_ppm'])


if __name__ == '__main__':
    unittest.main()

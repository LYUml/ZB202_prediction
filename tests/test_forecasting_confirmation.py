"""Follow-up evaluation cannot accidentally reopen an earlier label period."""
from pathlib import Path
import sys
import unittest
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zb202.forecasting.confirmation import evaluation_start


class ConfirmationPeriodTests(unittest.TestCase):
    def test_default_preserves_frozen_replay(self):
        self.assertEqual(evaluation_start(100., None), 100.)

    def test_timezone_offsets_define_same_cutoff(self):
        cutoff = datetime.fromisoformat('2026-10-20T09:22:56+00:00').timestamp()
        self.assertEqual(evaluation_start(100., '2026-10-20T17:22:56+08:00'), cutoff)
        self.assertEqual(evaluation_start(cutoff, '2026-10-20T09:22:56Z'), cutoff)

    def test_naive_and_earlier_cutoffs_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_start(100., '2026-10-20T09:22:56')
        with self.assertRaises(ValueError):
            evaluation_start(100., '1970-01-01T00:00:00Z')

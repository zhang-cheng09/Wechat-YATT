import importlib.util
import unittest
from pathlib import Path

from gpatch_v4.configs.training_config import RLTrainingConfig

_PATH = Path(__file__).resolve().parents[2] / "gpatch_v4" / "utils" / "fwd_only_exec.py"
_spec = importlib.util.spec_from_file_location("fwd_only_exec", _PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
split_microbatches_per_execution = _mod.split_microbatches_per_execution


class TestSplitMicrobatchesPerExecution(unittest.TestCase):
    def test_rl_default_preserves_unsplit_execution(self):
        training = RLTrainingConfig()

        self.assertIsNone(training.num_batches_per_execution)
        self.assertEqual(
            split_microbatches_per_execution(6, training.num_batches_per_execution),
            [6],
        )

    def test_one_caps_each_packed_mb(self):
        self.assertEqual(split_microbatches_per_execution(6, 1), [1, 1, 1, 1, 1, 1])

    def test_gas_larger_than_total_is_one_chunk(self):
        self.assertEqual(split_microbatches_per_execution(6, 8), [6])

    def test_splits_remainder(self):
        self.assertEqual(split_microbatches_per_execution(6, 4), [4, 2])

    def test_none_or_non_positive_means_no_split(self):
        self.assertEqual(split_microbatches_per_execution(6, None), [6])
        self.assertEqual(split_microbatches_per_execution(6, 0), [6])
        self.assertEqual(split_microbatches_per_execution(6, -1), [6])

    def test_empty(self):
        self.assertEqual(split_microbatches_per_execution(0, 1), [])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import math
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "zipcache" / "models" / "probe_selection.py"


def load_select_probe_ids():
    spec = importlib.util.spec_from_file_location("probe_selection", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load probe-selection helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_probe_ids


select_probe_ids = load_select_probe_ids()


class ProbeSelectionTests(unittest.TestCase):
    @staticmethod
    def official_probe_ids(q_len):
        # Mirrors the original ZipCache implementation, including its independent
        # int() rounding and random sampling with replacement.
        recent_start = int(q_len - 0.05 * q_len)
        random_count = int(0.05 * q_len)
        recent_ids = torch.arange(recent_start, q_len)
        random_ids = torch.randint(0, recent_start, (random_count,))
        return torch.cat((recent_ids, random_ids))

    def test_default_matches_original_rounding_and_sampling(self):
        for q_len in (2, 3, 7, 20, 127, 1024, 2048, 4096):
            with self.subTest(q_len=q_len):
                torch.manual_seed(20261029)
                expected = self.official_probe_ids(q_len)
                torch.manual_seed(20261029)
                actual = select_probe_ids(q_len)
                self.assertTrue(torch.equal(actual, expected))

    def test_probe_budget_is_constant_across_recent_shares(self):
        q_len = 1024
        shares = (0.0, 0.25, 0.5, 0.75, 1.0)
        counts = []
        for share in shares:
            torch.manual_seed(7)
            counts.append(
                select_probe_ids(q_len, probe_ratio=0.10, recent_share=share).numel()
            )
        expected_count = (
            q_len - int(q_len - 0.05 * q_len) + int(0.05 * q_len)
        )
        self.assertEqual(counts, [expected_count] * len(shares))

    def test_zero_recent_share_uses_only_random_probes(self):
        q_len = 1024
        torch.manual_seed(1)
        ids = select_probe_ids(q_len, recent_share=0.0)
        self.assertEqual(ids.numel(), self.official_probe_ids(q_len).numel())
        self.assertTrue(bool(torch.all((ids >= 0) & (ids < q_len))))

    def test_all_recent_share_returns_contiguous_recent_tail(self):
        q_len = 1024
        total = self.official_probe_ids(q_len).numel()
        ids = select_probe_ids(q_len, recent_share=1.0)
        expected = torch.arange(q_len - total, q_len)
        self.assertTrue(torch.equal(ids, expected))

    def test_quarter_and_three_quarter_shares_are_complements(self):
        q_len = 1024
        total = self.official_probe_ids(q_len).numel()
        quarter = select_probe_ids(q_len, recent_share=0.25)
        three_quarters = select_probe_ids(q_len, recent_share=0.75)
        recent_quarter = int(math.floor(total * 0.25 + 0.5))
        recent_three_quarters = int(math.floor(total * 0.75 + 0.5))
        self.assertEqual(recent_quarter, 26)
        self.assertEqual(recent_three_quarters, 77)
        self.assertEqual(recent_quarter + recent_three_quarters, total)
        self.assertEqual(quarter.numel(), total)
        self.assertEqual(three_quarters.numel(), total)

    def test_invalid_parameters_raise_value_error(self):
        for q_len in (0, -1):
            with self.subTest(q_len=q_len):
                with self.assertRaises(ValueError):
                    select_probe_ids(q_len)
        for ratio in (0.0, -0.1, 1.1):
            with self.subTest(probe_ratio=ratio):
                with self.assertRaises(ValueError):
                    select_probe_ids(1024, probe_ratio=ratio)
        for share in (-0.1, 1.1):
            with self.subTest(recent_share=share):
                with self.assertRaises(ValueError):
                    select_probe_ids(1024, recent_share=share)


if __name__ == "__main__":
    unittest.main()

"""Checks for generation comparison failures without downloading model weights."""

import contextlib
import io
import unittest

from run_gate import compare_completions
from tasks import T5GEMMA_WHITESPACE_SQUAD_IDS, _select_t5gemma_samples


class T5GemmaSampleSelectionTest(unittest.TestCase):
    def test_reference_whitespace_sample_is_replaced_without_reducing_limit(self):
        reference_whitespace_id = "5726d4a45951b619008f7f6c"
        self.assertIn(reference_whitespace_id, T5GEMMA_WHITESPACE_SQUAD_IDS)
        samples = [{"id": "first"}, {"id": reference_whitespace_id},
                   {"id": "second"}, {"id": "third"}]
        expected = [{"id": "first"}, {"id": "second"}]
        self.assertEqual(_select_t5gemma_samples(iter(samples), 2), expected)
        self.assertEqual(_select_t5gemma_samples(iter(samples), 2), expected)

    def test_unlimited_selection_preserves_remaining_order(self):
        samples = [{"id": "first"}, {"id": "second"}]
        self.assertEqual(_select_t5gemma_samples(iter(samples), None), samples)

    def test_zero_limit_does_not_select_a_prompt(self):
        self.assertEqual(_select_t5gemma_samples(iter([{"id": "first"}]), 0), [])


class CompletionComparisonTest(unittest.TestCase):
    def test_changes_are_reported_without_failing(self):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            changed = compare_completions(
                {"outputs": ["same", "old"]}, {"outputs": ["same", "new"]})
        self.assertEqual(changed, 1)
        self.assertIn("1/2 [informational]", log.getvalue())
        self.assertIn("prompt 2", log.getvalue())

    def test_identical_completions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare_completions(
                {"outputs": ["same"]}, {"outputs": ["same"]}), 0)

    def test_invalid_completions_fail(self):
        for baseline, candidate in [([], []), (["one"], []),
                                    (["one"], [" "]), ([""], ["one"]),
                                    (["one"], [None])]:
            with self.subTest(baseline=baseline, candidate=candidate):
                with self.assertRaises(ValueError):
                    compare_completions({"outputs": baseline}, {"outputs": candidate})


if __name__ == "__main__":
    unittest.main()

"""Checks for generation comparison failures without downloading model weights."""

import contextlib
import io
import unittest

from run_gate import compare_completions


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

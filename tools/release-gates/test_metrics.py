"""Validate measurements before applying release-gate thresholds."""

import contextlib
import io
import unittest
from unittest.mock import patch

import run_gate
from run_gate import judge


class MetricValidationTest(unittest.TestCase):
    def test_invalid_or_missing_metrics_fail_on_either_side(self):
        missing = object()
        valid = {"bleu": 30.0, "tokens_per_sec": 100.0}
        for label in ("baseline", "candidate"):
            for metric in valid:
                for value in (missing, None, "30", True, float("nan"), float("inf")):
                    with self.subTest(label=label, metric=metric, value=value):
                        invalid = dict(valid, **{metric: value})
                        if value is missing:
                            del invalid[metric]
                        base, cand = (invalid, valid) if label == "baseline" else (valid, invalid)
                        with self.assertRaisesRegex(ValueError, f"opus_bleu: {label} metric '{metric}'"):
                            judge("opus_bleu", base, cand)

    def test_wer_tolerance(self):
        for candidate, fails in ((0.20982142857142858, False), (0.22, True)):
            with self.subTest(candidate=candidate):
                results = judge("whisper_wer", {"wer": 0.203125, "rtf_inverse": 100},
                                {"wer": candidate, "rtf_inverse": 100})
                self.assertEqual(results["wer"].failed, fails)

    def test_performance_thresholds_are_relative_and_fail_gate(self):
        for task, metric, quality in (
            ("whisper_wer", "rtf_inverse", {"wer": 0.1}),
            ("opus_bleu", "tokens_per_sec", {"bleu": 30}),
            ("t5gemma_generation", "tokens_per_sec", {}),
        ):
            for baseline in (1, 100):
                for ratio, expected_failure in ((0.8, True), (0.95, False)):
                    with self.subTest(task=task, baseline=baseline, ratio=ratio):
                        results = judge(
                            task, dict(quality, **{metric: baseline}),
                            dict(quality, **{metric: baseline * ratio}))
                        self.assertEqual(results[metric].failed, expected_failure)
                        self.assertTrue(results[metric].performance)

    def test_confirmation_cannot_hide_quality_failures(self):
        for rounds, expected in (
            ([(False, {"tokens_per_sec"}), (False, set())], False),
            ([(False, {"tokens_per_sec"}), (False, {"tokens_per_sec"})], True),
            ([(True, {"tokens_per_sec"}), (False, set())], True),
            ([(False, {"tokens_per_sec"}), (True, set())], True),
            ([(True, set())], True),
            ([(False, set())], False),
        ):
            with self.subTest(rounds=rounds), \
                 patch.object(run_gate, "measure_round", side_effect=rounds) as measure, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_gate.evaluate_task("b", "c", "opus_bleu", 20, "cpu",
                        {"baseline": {}, "candidate": {}}, compute_type="auto"),
                                 expected)
                self.assertEqual(measure.call_count, len(rounds))
                self.assertTrue(all(call.kwargs["compute_type"] == "auto"
                                    for call in measure.call_args_list))

    def test_measurements_preserve_quality_and_ignore_performance_outliers(self):
        for metric, values, expected_quality in (
            ("tokens_per_sec", (100, 100, 1), False),
            ("bleu", (30, 28.9, 30), True),
        ):
            with self.subTest(metric=metric):
                values = iter(values)

                def result(py, *args, **kwargs):
                    metrics = {"bleu": 30, "tokens_per_sec": 100}
                    if py == "c":
                        metrics[metric] = next(values)
                    return metrics

                with patch.object(run_gate, "run_task", side_effect=result) as task, \
                     contextlib.redirect_stdout(io.StringIO()):
                    quality, performance = run_gate.measure_round("b", "c", "opus_bleu", 20, "cpu",
                        {"baseline": {}, "candidate": {}})
                self.assertEqual(quality, expected_quality)
                self.assertEqual(performance, set())
                self.assertEqual([call.args[0] for call in task.call_args_list],
                                 ["b", "c", "c", "b", "b", "c"])

    def test_confirmation_retains_late_quality_failure(self):
        candidates = iter([100, 100, 100, 100, 28])

        def result(py, *args, **kwargs):
            return {"bleu": 30 if py == "b" else next(candidates),
                    "tokens_per_sec": 100}

        with patch.object(run_gate, "run_task", side_effect=result), \
             contextlib.redirect_stdout(io.StringIO()):
            quality, performance = run_gate.measure_round(
                "b", "c", "opus_bleu", 20, "cpu",
                {"baseline": {}, "candidate": {}}, confirmation=True)
        self.assertTrue(quality)
        self.assertEqual(performance, set())


if __name__ == "__main__":
    unittest.main()

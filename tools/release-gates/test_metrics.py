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
                for value in (missing, None, "30", True, False, float("nan"),
                              float("inf"), float("-inf")):
                    with self.subTest(label=label, metric=metric, value=value):
                        invalid = dict(valid, **{metric: value})
                        if value is missing:
                            del invalid[metric]
                        base, cand = (invalid, valid) if label == "baseline" else (valid, invalid)
                        with self.assertRaisesRegex(ValueError, f"opus_bleu: {label} metric '{metric}'"):
                            judge("opus_bleu", base, cand)

    def test_zero_values_are_valid(self):
        results = judge("whisper_wer", {"wer": 0, "rtf_inverse": 0.0},
                             {"wer": 0.0, "rtf_inverse": 0})
        self.assertFalse(any(result.failed for result in results.values()))

    def test_wer_tolerance(self):
        for candidate, fails in ((0.20982142857142858, False), (0.22, True)):
            with self.subTest(candidate=candidate):
                results = judge("whisper_wer", {"wer": 0.203125, "rtf_inverse": 100},
                                {"wer": candidate, "rtf_inverse": 100})
                self.assertEqual(results["wer"].failed, fails)

    def test_quality_regression_fails(self):
        results = judge("opus_bleu", {"bleu": 30, "tokens_per_sec": 100},
                          {"bleu": 28.9, "tokens_per_sec": 100})
        self.assertTrue(results["bleu"].failed)

    def test_performance_thresholds_are_relative_and_fail_gate(self):
        for task, metric, quality in (
            ("whisper_wer", "rtf_inverse", {"wer": 0.1}),
            ("opus_bleu", "tokens_per_sec", {"bleu": 30}),
            ("t5gemma_generation", "tokens_per_sec", {}),
        ):
            for baseline in (1, 100, 1000):
                for ratio, expected_failure in ((0.8, True), (0.9, False),
                                                (0.95, False), (1.2, False)):
                    with self.subTest(task=task, baseline=baseline, ratio=ratio):
                        results = judge(
                            task, dict(quality, **{metric: baseline}),
                            dict(quality, **{metric: baseline * ratio}))
                        self.assertEqual(results[metric].failed, expected_failure)
                        self.assertTrue(results[metric].performance)

    def test_main_exit_status_for_performance_regressions(self):
        baseline = {
            "whisper_wer": {"wer": 0.1, "rtf_inverse": 100},
            "opus_bleu": {"bleu": 30, "tokens_per_sec": 100},
            "t5gemma_generation": {"tokens_per_sec": 100, "outputs": ["hello"]},
        }
        for degraded_task in (None, *baseline):
            with self.subTest(degraded_task=degraded_task):
                def result(py, task, limit, device):
                    metrics = dict(baseline[task])
                    if py == "candidate" and task == degraded_task:
                        metric = "rtf_inverse" if task == "whisper_wer" else "tokens_per_sec"
                        metrics[metric] = 80
                    return metrics
                with patch.object(run_gate, "latest_version", return_value="4.8.5"), \
                     patch.object(run_gate, "make_venv", side_effect=["baseline", "candidate"]), \
                     patch.object(run_gate, "run_task", side_effect=result), \
                     patch("sys.argv", ["run_gate.py"]), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaises(SystemExit) as exited:
                    run_gate.main()
                self.assertEqual(exited.exception.code, 1 if degraded_task else 0)

    def test_candidate_wheel_pattern_requires_one_match(self):
        for wheels in ([], ["candidate.whl"], ["a.whl", "b.whl"]):
            with self.subTest(wheels=wheels), \
                 patch("sys.argv", ["run_gate.py", "--candidate-wheel-pattern", "*.whl"]), \
                 patch.object(run_gate.glob, "glob", return_value=wheels), \
                 patch.object(run_gate, "latest_version", return_value="4.8.5") as latest, \
                 patch.object(run_gate, "make_venv") as make_venv, \
                 patch.object(run_gate, "evaluate_task", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as exited:
                run_gate.main()
            self.assertEqual(exited.exception.code, 0 if len(wheels) == 1 else 2)
            if len(wheels) == 1:
                self.assertEqual(make_venv.call_args.args[1],
                                 str(run_gate.pathlib.Path(wheels[0]).resolve()))
            else:
                latest.assert_not_called()
                make_venv.assert_not_called()

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
                self.assertEqual(run_gate.evaluate_task("b", "c", "opus_bleu", 20, "cpu"),
                                 expected)
                self.assertEqual(measure.call_count, len(rounds))

    def test_measurements_preserve_quality_and_ignore_performance_outliers(self):
        for metric, values, expected_quality in (
            ("tokens_per_sec", (100, 100, 1), False),
            ("bleu", (30, 28.9, 30), True),
        ):
            with self.subTest(metric=metric):
                values = iter(values)

                def result(py, *args):
                    metrics = {"bleu": 30, "tokens_per_sec": 100}
                    if py == "c":
                        metrics[metric] = next(values)
                    return metrics

                with patch.object(run_gate, "run_task", side_effect=result) as task, \
                     contextlib.redirect_stdout(io.StringIO()):
                    quality, performance = run_gate.measure_round("b", "c", "opus_bleu", 20, "cpu")
                self.assertEqual(quality, expected_quality)
                self.assertEqual(performance, set())
                self.assertEqual([call.args[0] for call in task.call_args_list],
                                 ["b", "c", "c", "b", "b", "c"])

    def test_threshold_equality_is_not_a_regression(self):
        results = judge("whisper_wer", {"wer": 0, "rtf_inverse": 100},
                             {"wer": 0.01, "rtf_inverse": 90})
        self.assertFalse(any(result.failed for result in results.values()))


if __name__ == "__main__":
    unittest.main()

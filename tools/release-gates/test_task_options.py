"""Check worker options, limits and compute type resolution without loading models."""

import contextlib
import io
import subprocess
import sys
import unittest
from unittest.mock import patch

import run_gate
import tasks


class TaskOptionsTest(unittest.TestCase):
    def test_translation_tasks_require_model_path(self):
        for task in tasks.CONVERTERS:
            with self.subTest(task=task), self.assertRaises(TypeError):
                tasks.TASKS[task]()
            result = subprocess.run(
                [sys.executable, str(run_gate.ROOT / "tasks.py"), "run", task],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--model-dir is required", result.stderr)

    def test_runner_forwards_sample_limits(self):
        for limit in (7, 0):
            with self.subTest(limit=limit), \
                    patch("sys.argv", ["run_gate.py", "--limit", str(limit), "--bleu-limit", "13"]), \
                    patch.object(run_gate, "latest_version", return_value="4.9.0"), \
                    patch.object(run_gate, "make_venv", side_effect=["b", "c"]), \
                    patch.object(run_gate, "convert_models", return_value={}), \
                    patch.object(run_gate, "select_compute_types", return_value=("default",)), \
                    patch.object(run_gate, "evaluate_task", return_value=False) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
                run_gate.main()
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual({call.args[2]: call.args[3] for call in evaluate.call_args_list},
                             {"whisper_wer": limit, "opus_bleu": 13,
                              "t5gemma_generation": limit})


class ComputeTypesTest(unittest.TestCase):
    def test_skip_auto_only_when_both_wheels_match(self):
        for baseline, candidate, expected in (
            (("float32", "float32"), ("float32", "float32"), ("default",)),
            (("float32", "int8"), ("float32", "float32"), ("default", "auto")),
            (("float32", "float32"), ("float32", "int8"), ("default", "auto")),
            (("float32", "float32"), ("int8", "int8"), ("default",)),
        ):
            with self.subTest(baseline=baseline, candidate=candidate), \
                    patch.object(run_gate, "call_worker", side_effect=[
                        dict(zip(("default", "auto"), pair)) for pair in (baseline, candidate)
                    ]) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_gate.select_compute_types(
                    "b", "c", "opus_bleu", "cpu",
                    {"baseline": {"opus_bleu": "bm"}, "candidate": {"opus_bleu": "cm"}}), expected)
                self.assertEqual([call.args[-1] for call in run.call_args_list], ["bm", "cm"])

    def test_invalid_resolution_fails(self):
        for invalid in (None, "", "auto", "default", 1):
            with self.subTest(invalid=invalid), patch.object(run_gate, "call_worker", return_value={
                "default": invalid, "auto": "int8"
            }), self.assertRaises(ValueError):
                run_gate.select_compute_types("b", "c", "opus_bleu", "cpu",
                                              {"baseline": {}, "candidate": {}})


"""Check MPS gate preflight and model device selection without Apple hardware."""

import contextlib
import io
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

import run_gate
import tasks


class MPSGateTest(unittest.TestCase):
    def test_preflight_checks_both_wheels_before_conversion(self):
        for failing_wheel in (None, "b", "c"):
            events = []

            def check(command):
                events.append(("check", command[0]))
                self.assertIn("get_mps_device_count", command[2])
                self.assertIn("get_supported_compute_types('mps')", command[2])
                if command[0] == failing_wheel:
                    raise subprocess.CalledProcessError(1, command)

            def convert(py, path):
                events.append(("convert", py))
                return {}

            with self.subTest(failing_wheel=failing_wheel), \
                    patch("sys.argv", ["run_gate.py", "--device", "mps"]), \
                    patch.object(run_gate, "latest_version", return_value="4.9.0"), \
                    patch.object(run_gate, "make_venv", side_effect=["b", "c"]), \
                    patch.object(run_gate.subprocess, "check_call", side_effect=check), \
                    patch.object(run_gate, "convert_models", side_effect=convert), \
                    patch.object(run_gate, "select_compute_types", return_value=("default",)), \
                    patch.object(run_gate, "evaluate_task", return_value=False) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()):
                if failing_wheel:
                    with self.assertRaises(subprocess.CalledProcessError):
                        run_gate.main()
                    self.assertFalse(any(event[0] == "convert" for event in events))
                    evaluate.assert_not_called()
                else:
                    with self.assertRaises(SystemExit) as exited:
                        run_gate.main()
                    self.assertEqual(exited.exception.code, 0)
                    self.assertEqual(events, [("check", "b"), ("check", "c"),
                                              ("convert", "b"), ("convert", "c")])
                    self.assertEqual(evaluate.call_count, len(run_gate.RULES))
                    self.assertTrue(all(call.args[4] == "mps" for call in evaluate.call_args_list))

    def test_all_model_loaders_forward_mps_and_requested_type(self):
        whisper = Mock()
        translator = Mock()
        with patch.dict(sys.modules, {
                "faster_whisper": types.SimpleNamespace(WhisperModel=whisper),
                "ctranslate2": types.SimpleNamespace(Translator=translator)}):
            for requested in ("default", "auto"):
                for task in tasks.TASKS:
                    with self.subTest(task=task, requested=requested):
                        tasks.load_task_model(task, "mps", requested, "model")
                        loader = whisper if task == "whisper_wer" else translator
                        self.assertEqual(loader.call_args.kwargs["device"], "mps")
                        self.assertEqual(loader.call_args.kwargs["compute_type"], requested)


if __name__ == "__main__":
    unittest.main()

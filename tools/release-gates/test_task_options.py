"""Check required model paths and workload limits without loading models."""

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
            for command in ("run", "resolve"):
                with self.subTest(task=task, command=command):
                    result = subprocess.run(
                        [sys.executable, str(run_gate.ROOT / "tasks.py"), command, task],
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("--model-dir is required", result.stderr)

    def test_full_dataset_limit_is_forwarded_to_worker(self):
        with patch.object(run_gate.subprocess, "check_output", return_value=b'{"wer": 0}\n') as run:
            run_gate.run_task("python", "whisper_wer", None, "cpu")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--limit") + 1], "0")

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
                             {"whisper_wer": limit or None, "opus_bleu": 13,
                              "t5gemma_generation": limit or None})

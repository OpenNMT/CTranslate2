"""Check model-specific compute type resolution without downloading weights."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_gate
import tasks


class ComputeTypesTest(unittest.TestCase):
    def test_skip_auto_only_when_both_wheels_match(self):
        for baseline, candidate, expected in (
            (("float32", "float32"), ("float32", "float32"), ("default",)),
            (("float32", "int8"), ("float32", "float32"), ("default", "auto")),
            (("float32", "float32"), ("float32", "int8"), ("default", "auto")),
            (("float32", "float32"), ("int8", "int8"), ("default",)),
        ):
            with self.subTest(baseline=baseline, candidate=candidate), \
                    patch.object(run_gate, "run_task", side_effect=[
                        dict(zip(("default", "auto"), pair)) for pair in (baseline, candidate)
                    ]) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_gate.select_compute_types(
                    "b", "c", "opus_bleu", "cpu",
                    {"baseline": {"opus_bleu": "bm"}, "candidate": {"opus_bleu": "cm"}}), expected)
                self.assertEqual([call.kwargs for call in run.call_args_list],
                                 [{"command": "resolve"}, {"command": "resolve"}])
                self.assertEqual([call.args[-1] for call in run.call_args_list], ["bm", "cm"])

    def test_invalid_resolution_fails(self):
        for invalid in (None, "", "auto", "default", 1):
            with self.subTest(invalid=invalid), patch.object(run_gate, "run_task", return_value={
                "default": invalid, "auto": "int8"
            }), self.assertRaises(ValueError):
                run_gate.select_compute_types("b", "c", "opus_bleu", "cpu",
                                              {"baseline": {}, "candidate": {}})

    def test_resolution_uses_loaded_models_for_each_request(self):
        for task in tasks.TASKS:
            models = [SimpleNamespace(compute_type="float32"), SimpleNamespace(compute_type="int8")]
            if task == "whisper_wer":
                models = [SimpleNamespace(model=model) for model in models]
            with self.subTest(task=task), patch.object(tasks, "load_task_model", side_effect=models) as load:
                self.assertEqual(tasks.resolve_compute_types(task, "cpu", "model"),
                                 {"default": "float32", "auto": "int8"})
                self.assertEqual([call.args for call in load.call_args_list],
                                 [(task, "cpu", kind, "model") for kind in ("default", "auto")])

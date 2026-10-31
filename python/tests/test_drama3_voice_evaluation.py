"""Guard WER normalization and prevent incomplete cloning studies from passing."""

import importlib.util
import json
import sys
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "voice_evaluation",
    Path(__file__).resolve().parents[2] / "tools/evaluate_drama3_voice_cloning.py",
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class VoiceEvaluationTest(unittest.TestCase):
    def test_known_edit_counts(self):
        self.assertEqual(
            evaluation.word_errors("one two three", "one four")["errors"], 2
        )
        self.assertEqual(evaluation.word_errors("one two", "one two three")["wer"], 0.5)
        self.assertEqual(evaluation.word_errors("one two", "")["wer"], 1)

    def test_normalization_preserves_accents(self):
        self.assertEqual(
            evaluation.word_errors("¡CAFÉ, mañana!", "café mañana")["errors"], 0
        )
        self.assertEqual(evaluation.word_errors("café", "cafe")["errors"], 1)

    def test_incomplete_matrix_cannot_pass_with_zero_wer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = {"cases": [1, 2, 3, 4], "seeds": [0, 17]}
            record = dict(
                id="partial",
                voice="a",
                text="same words",
                language="en",
                status="ok",
                finite=True,
                rms=0.1,
                finish_reason="eos",
                repeat_equal=True,
            )
            (output / "manifest.json").write_text(json.dumps(manifest))
            (output / "generation.json").write_text(json.dumps([record]))
            references = [
                dict(id=x, audio=x, text="same words", language="en")
                for x in ("a", "b")
            ]
            recognizer = SimpleNamespace(
                transcribe=lambda *a, **kw: ([SimpleNamespace(text="same words")], None)
            )
            fake = SimpleNamespace(WhisperModel=lambda *a, **kw: recognizer)
            args = SimpleNamespace(output=output, threads=1, asr_model=Path(directory))
            with patch.dict(sys.modules, {"faster_whisper": fake}), patch.object(
                evaluation, "read_audio", return_value=np.zeros(16000, dtype=np.float32)
            ):
                evaluation.asr(args, references)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["micro_wer"], 0)
            self.assertFalse(summary["content_pass"])

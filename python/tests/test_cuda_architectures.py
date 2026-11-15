import os
import subprocess
import tempfile
import unittest

from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "tools/get_cuda_architectures.sh"
LEGACY = "53-real;60-real;61-real;70-real;75-real;80-real;86"
BLACKWELL = LEGACY + ";100;120"


@unittest.skipUnless(os.name == "posix", "These helper tests require a POSIX shell")
class CudaArchitecturesTest(unittest.TestCase):
    def test_release_architectures(self):
        cases = [
            ("release 12.7", 0, "", LEGACY),
            (
                "nvcc\r\nCuda compilation tools, release 12.8, V12.8.93\r\n",
                0,
                "",
                BLACKWELL,
            ),
            ("release 12.10", 0, "", BLACKWELL),
            ("release 13.0", 0, "", BLACKWELL),
            ("", 2, "", None),
            ("unknown version", 0, "", None),
            ("", 2, "89-real", "89-real"),
        ]
        for output, exit_code, override, expected in cases:
            with self.subTest(output=output, override=override):
                with tempfile.TemporaryDirectory(prefix="cuda tools ") as directory:
                    compiler = Path(directory) / "nvcc"
                    compiler.write_text(
                        f"#!/bin/sh\nprintf '%s' '{output}'\nexit {exit_code}\n"
                    )
                    compiler.chmod(0o755)
                    result = subprocess.run(
                        ["bash", str(HELPER), str(compiler)],
                        capture_output=True,
                        text=True,
                        env=dict(os.environ, CMAKE_CUDA_ARCHITECTURES=override),
                    )
                    self.assertEqual(
                        result.returncode == 0, expected is not None, result.stderr
                    )
                    self.assertEqual(result.stdout.strip(), expected or "")

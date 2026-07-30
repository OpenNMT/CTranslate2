import subprocess
import sys

import pytest

from ctranslate2.extensions import _batch_iterator as batch_iterator


@pytest.mark.parametrize(
    "batch_size,batch_type,lengths,expected_batch_sizes",
    [
        (2, "examples", [2, 3, 4, 1, 1], [2, 2, 1]),
        (6, "tokens", [2, 3, 1, 4, 1, 2], [2, 1, 1, 2]),
    ],
)
def test_batch_iterator(batch_size, batch_type, lengths, expected_batch_sizes):
    iterable = (["a"] * length for length in lengths)

    batches = batch_iterator(iterable, batch_size, batch_type)
    batch_sizes = [len(batch[0]) for batch in batches]

    assert batch_sizes == expected_batch_sizes


@pytest.mark.parametrize("module_name", ["torch", "transformers"])
def test_import_does_not_load_conversion_dependencies(module_name):
    # The converters and specs submodules are only needed to convert models, so importing
    # the package for inference should not pull their heavy dependencies into the process.
    # Run in a subprocess because the test session itself imports them.
    code = "import sys; import ctranslate2; print(%r in sys.modules)" % module_name
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "False"

import importlib.util
import subprocess

from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "tools/benchmark/llm/perplexity.py"
_spec = importlib.util.spec_from_file_location("llm_perplexity", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("size", [2, 3, 7, 32, 33, 64, 65, 4097])
@pytest.mark.parametrize("context,stride", [(2, 1), (8, 1), (8, 4), (8, 7), (512, 256)])
def test_corpus_windows_score_every_target_once(size, context, stride):
    targets = []
    for start, end, skip in _module.windows(size, context, stride):
        assert 2 <= end - start <= context
        targets.extend(range(start + 1 + skip, end))
    assert targets == list(range(1, size))


@pytest.mark.parametrize(
    "size,context,stride", [(1, 8, 4), (8, 1, 1), (8, 8, 0), (8, 8, 8)]
)
def test_corpus_windows_reject_invalid_settings(size, context, stride):
    with pytest.raises(ValueError):
        list(_module.windows(size, context, stride))


def test_benchmark_metadata_finds_repository_outside_working_directory(
    tmp_path, monkeypatch
):
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        pytest.skip("Source archive has no Git metadata")
    spec = importlib.util.spec_from_file_location(
        "llm_benchmark", repository / "tools/benchmark/llm/benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    monkeypatch.chdir(tmp_path)
    result = benchmark.metadata()
    assert result["commit"] == expected
    assert isinstance(result["dirty"], bool)

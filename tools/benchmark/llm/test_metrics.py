import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "llm_benchmark", Path(__file__).with_name("benchmark.py")
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_ttft_is_not_counted_as_a_decode_interval():
    report = benchmark.summarize_run(10, [12, 12.25, 12.75], 13, 128, [1, 2, 3])
    assert report["ttft_s"] == 2
    assert report["decode_tokens_per_s"] == pytest.approx(2 / 0.75)
    assert report["per_token_latency_s"] == [0.25, 0.5]
    assert report["total_s"] == 3


def test_incomplete_generation_cannot_produce_misleading_timing():
    with pytest.raises(RuntimeError):
        benchmark.summarize_run(10, [12], 13, 128, [1, 2])

"""Opt-in real decoder-only regressions; model downloads are kept out of pytest."""

import os
import subprocess
import sys

from pathlib import Path

import pytest
import test_utils


@test_utils.require_mps
@pytest.mark.parametrize("length", [33, 2048, 2051])
@pytest.mark.parametrize("flash_attention", [False, True])
def test_real_decoder_repeated_prefill_logits_are_finite(
    tmp_path, length, flash_attention
):
    model = os.environ.get("CT2_MPS_LLM_MODEL")
    source = os.environ.get("CT2_MPS_LLM_SOURCE")
    if not model or not source:
        pytest.skip(
            "Set CT2_MPS_LLM_MODEL and CT2_MPS_LLM_SOURCE to local pinned models"
        )
    repo = Path(__file__).resolve().parents[2]
    # The second request exercises reused storage and, at 2048, a rotary cache
    # grown by the preceding incremental decoding. Validate every vocabulary
    # logit: token 0 is also a legitimate token and cannot detect NaNs alone.
    environment = os.environ.copy()
    # This regression must exercise the default path. An inherited experimental
    # switch must not make a broken default implementation appear to pass.
    for name in (
        "CT2_MPS_USE_SIMDGROUP_GEMM",
        "CT2_MPS_USE_FUSED_ATTENTION",
        "CT2_MPS_DISABLE_FLASH_DECODE",
        "CT2_MPS_DISABLE_FLASH_PREFILL",
        "CT2_MPS_USE_FLASH_CROSS_ATTENTION",
        "CT2_MPS_DISABLE_COOPERATIVE_QK",
        "CT2_MPS_DISABLE_VECTOR_CACHE",
        "CT2_MPS_DISABLE_VECTOR_MOE",
        "CT2_MPS_DISABLE_SMALL_MOE_PLAN",
        "CT2_MPS_DISABLE_SHORT_MOE_GEMM",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "tools/benchmark/llm/benchmark.py"),
            "--model",
            model,
            "--tokenizer",
            source,
            "--device",
            "mps",
            "--compute-type",
            "float16",
            "--lengths",
            str(length),
            "--output-tokens",
            "4",
            "--repeats",
            "2",
            "--validate-logits",
            "--output",
            str(tmp_path / "generation.json"),
            *(["--flash-attention"] if flash_attention else []),
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@test_utils.require_mps
@pytest.mark.parametrize("capture_logits", [False, True])
def test_flash_mixed_length_batch_greedy_cpu_parity(capture_logits):
    """FP32 vocabulary logits must preserve the formerly tied greedy decision."""
    import json

    model = os.environ.get("CT2_MPS_LLM_MODEL")
    source = os.environ.get("CT2_MPS_LLM_SOURCE")
    if not model or not source:
        pytest.skip(
            "Set CT2_MPS_LLM_MODEL and CT2_MPS_LLM_SOURCE to local pinned models"
        )
    repo = Path(__file__).resolve().parents[2]
    code = """
import json, sys
import ctranslate2
from transformers import AutoTokenizer
sys.path.insert(0, 'tools/benchmark/llm')
from benchmark import make_prompts
model, source, device, capture = sys.argv[1:]
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
prompts = make_prompts(tokenizer, [33, 65])
generator = ctranslate2.Generator(model, device=device,
    compute_type='float32' if device == 'cpu' else 'float16',
    flash_attention=device == 'mps', inter_threads=1, intra_threads=4)
results = generator.generate_batch([tokenizer.convert_ids_to_tokens(ids) for ids in prompts],
    include_prompt_in_result=False, beam_size=1, sampling_topk=1,
    max_length=8, min_length=8, end_token=[],
    return_logits_vocab=capture == "1", return_scores=capture == "1")
print(json.dumps([result.sequences_ids[0] for result in results]))
"""
    outputs = {}
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("CT2_MPS_USE_") or name in (
            "CT2_MPS_DISABLE_FLASH_DECODE",
            "CT2_MPS_DISABLE_FLASH_PREFILL",
            "CT2_MPS_DISABLE_COOPERATIVE_QK",
            "CT2_MPS_DISABLE_VECTOR_CACHE",
            "CT2_MPS_DISABLE_VECTOR_MOE",
            "CT2_MPS_DISABLE_SMALL_MOE_PLAN",
            "CT2_MPS_DISABLE_SHORT_MOE_GEMM",
        ):
            environment.pop(name)
    for device in ("cpu", "mps"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                model,
                source,
                device,
                str(int(capture_logits)),
            ],
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs[device] = json.loads(result.stdout.strip().splitlines()[-1])
    assert outputs["mps"] == outputs["cpu"]

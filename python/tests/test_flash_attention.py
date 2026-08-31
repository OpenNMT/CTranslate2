"""Correctness tests for the native HIP Flash Attention path.

These tests exercise the FP16 and BF16 WMMA kernels, the decode kernel, and
the KV-cache write path against the standard MultiHeadAttention oracle.
A model snapshot (Systran/faster-whisper-medium) is required; the test is
skipped if it isn't already on disk so it can run in environments without
network access.

The tests are written to be CI-friendly: only correctness is asserted; the
memory footprint of the materialised attention score buffer that the Flash
path avoids is printed for context, not asserted.
"""

import os
import sys

import numpy as np
import pytest

# Local-build dev-loop helper: when the test is run against a non-installed
# ctranslate2 source tree (i.e. python/ctranslate2 imports a freshly-built
# DLL but the ROCm SDK is only available via the pip-installed
# _rocm_sdk_core/_rocm_sdk_libraries_custom wheels), Python 3.8+ no longer
# honours PATH for DLL search and the load fails before we get to import.
# Find the SDK directories among site-packages and add them up-front.  In a
# normal CI setup this is a no-op (the SDK is already discoverable).
if sys.platform == "win32":
    import site

    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        for sub in ("_rocm_sdk_core/bin", "_rocm_sdk_libraries_custom/bin"):
            cand = os.path.join(site_dir, *sub.split("/"))
            if os.path.isdir(cand):
                try:
                    os.add_dll_directory(cand)
                except (FileNotFoundError, OSError):
                    pass

import test_utils

import ctranslate2


# ----------------------------------------------------------------------------
# Model discovery — uses an existing CT2 Whisper snapshot if available.
# ----------------------------------------------------------------------------
def _find_whisper_medium():
    """Return the path to a converted faster-whisper-medium snapshot, or None."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(
        hf_cache,
        "models--Systran--faster-whisper-medium",
        "snapshots",
    )
    if not os.path.isdir(model_dir):
        return None
    # snapshots/<sha>/{model.bin, config.json, tokenizer.json, ...}
    for sha in os.listdir(model_dir):
        full = os.path.join(model_dir, sha)
        if os.path.isfile(os.path.join(full, "model.bin")):
            return full
    return None


_MODEL_PATH = _find_whisper_medium()

require_model = pytest.mark.skipif(
    _MODEL_PATH is None,
    reason="faster-whisper-medium snapshot not cached locally",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _mel(seed=0):
    """Synthetic 30-second mel spectrogram, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 80, 3000)).astype(np.float32)


def _encoder_output(model, mel):
    """Run the encoder and return the result as a CPU FP32 numpy array."""
    features = ctranslate2.StorageView.from_array(mel)
    out = model.encode(features, to_cpu=True)
    # BF16 outputs aren't directly numpy-castable.
    if out.dtype != ctranslate2.DataType.float32:
        out = out.to(ctranslate2.DataType.float32)
    return np.array(out)


def _score_buffer_bytes(seqlen_q, seqlen_k, num_heads=16, batch=1):
    """Bytes the standard attention path would allocate for the
    [B, H, Sq, Sk] FP32 score matrix.  This is what Flash Attention's
    online-softmax design avoids materialising in HBM."""
    return batch * num_heads * seqlen_q * seqlen_k * 4


# ----------------------------------------------------------------------------
# Encoder correctness — Flash=ON must match Flash=OFF within a small tolerance.
# ----------------------------------------------------------------------------
@test_utils.require_cuda
@require_model
@pytest.mark.parametrize(
    "compute_type,abs_tol,rel_tol",
    [
        # FP16 path: 11-bit mantissa, accumulators are FP32.  The diff is
        # dominated by re-ordering of summations in WMMA vs. rocBLAS-GEMM.
        ("float16", 0.5, 5e-3),
        # BF16 path: only 8-bit mantissa, larger accumulated rounding.
        ("bfloat16", 2.0, 3e-2),
    ],
)
def test_flash_attention_encoder_matches_standard(compute_type, abs_tol, rel_tol):
    """The Flash Attention encoder output must match the standard path
    within float16/bfloat16 rounding noise."""
    mel = _mel(seed=0)

    m_off = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type=compute_type,
        flash_attention=False,
    )
    m_on = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type=compute_type,
        flash_attention=True,
    )

    out_off = _encoder_output(m_off, mel)
    out_on = _encoder_output(m_on, mel)

    assert out_off.shape == out_on.shape
    diff = np.abs(out_off - out_on)
    max_diff = float(diff.max())
    max_abs = float(np.abs(out_off).max()) + 1e-8
    rel_diff = max_diff / max_abs

    # Informative — what Flash Attention's online softmax saves us in HBM
    # per layer (Whisper-medium encoder: Sq = Sk = 1500, 16 heads).
    saved_bytes = _score_buffer_bytes(1500, 1500, num_heads=16, batch=1)
    print(
        f"\n[{compute_type}] encoder max_abs_diff={max_diff:.4f}, "
        f"rel_diff={rel_diff*100:.3f}%; "
        f"per-layer score-buffer avoided = {saved_bytes/1024/1024:.1f} MiB"
    )

    assert (
        max_diff <= abs_tol
    ), f"{compute_type} encoder max diff {max_diff:.4f} exceeds {abs_tol}"
    assert rel_diff <= rel_tol, (
        f"{compute_type} encoder rel diff {rel_diff*100:.3f}% exceeds "
        f"{rel_tol*100:.3f}%"
    )


# ----------------------------------------------------------------------------
# generate() correctness — exercises decode-kernel + KV-cache write path.
# FP16 should produce token-identical output to the standard path; BF16 may
# differ in the last few tokens due to the smaller mantissa.
# ----------------------------------------------------------------------------
PROMPTS = [[50258, 50259, 50360]]


@test_utils.require_cuda
@require_model
@pytest.mark.parametrize("seed", [42, 123, 777, 999, 1234])
def test_flash_attention_generate_fp16_token_match(seed):
    """FP16 Flash=ON must produce identical token sequences to Flash=OFF
    across multiple random inputs.

    This is also the regression test for the softmax-reduction block-size
    bug: that bug caused the very first generated token to always be 50411
    regardless of the input.  Five distinct seeds with distinct expected
    tokens makes silent regression effectively impossible.
    """
    mel = _mel(seed=seed)

    m_off = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type="float16",
        flash_attention=False,
    )
    m_on = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type="float16",
        flash_attention=True,
    )

    feat_off = ctranslate2.StorageView.from_array(mel)
    feat_on = ctranslate2.StorageView.from_array(mel)

    r_off = m_off.generate(feat_off, PROMPTS, beam_size=1, max_length=20)
    r_on = m_on.generate(feat_on, PROMPTS, beam_size=1, max_length=20)

    tok_off = r_off[0].sequences_ids[0]
    tok_on = r_on[0].sequences_ids[0]
    assert (
        tok_off == tok_on
    ), f"seed={seed}: Flash=ON produced {tok_on} but oracle is {tok_off}"


# ----------------------------------------------------------------------------
# Regression test for the softmax-reduction-block-size bug.
#
# Symptom (before the fix in commit d9016a58): generate() with Flash=ON would
# emit token 50411 as the first generated token regardless of the audio input,
# because the per-row tree reduction in hip_attn_softmax_kernel ran with
# block = min(seqlen_k, 256), and for seqlen_k = 3 (the Whisper prompt
# prefill with three tokens) the reduction silently dropped the third score.
#
# This test reproduces the exact pre-fix configuration and asserts that the
# first generated token actually depends on the audio input.
# ----------------------------------------------------------------------------
@test_utils.require_cuda
@require_model
def test_softmax_block_size_regression():
    """Different audio inputs must produce different first generated tokens
    when Flash Attention is enabled.  Guards against a re-emergence of the
    softmax reduction block-size bug."""
    m_on = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type="float16",
        flash_attention=True,
    )

    first_tokens = set()
    for seed in [42, 123, 777, 999, 1234]:
        mel = _mel(seed=seed)
        feat = ctranslate2.StorageView.from_array(mel)
        # max_length must exceed the 3 prompt tokens for the generation step
        # to actually emit a token; we only inspect index 0.
        r = m_on.generate(feat, PROMPTS, beam_size=1, max_length=5)
        seq = r[0].sequences_ids[0]
        assert len(seq) >= 1, f"seed={seed}: no token generated"
        first_tokens.add(seq[0])

    assert len(first_tokens) > 1, (
        "Flash Attention generated the same first token for five distinct "
        f"random inputs ({first_tokens}) — softmax-reduction block-size bug "
        "may have re-surfaced (see commit d9016a58)."
    )


# ----------------------------------------------------------------------------
# Batch > 1 — exercises the WMMA path with a non-trivial batch dimension.
# Whisper-medium encoder runs at Sq = Sk = 1500, so this also covers the
# largest tile-count case.  Both flash and standard paths must agree per
# batch element.
# ----------------------------------------------------------------------------
@test_utils.require_cuda
@require_model
@pytest.mark.parametrize("batch_size", [2, 4])
def test_flash_attention_encoder_batched(batch_size):
    """Encoder correctness for batch_size > 1."""
    mels = np.stack(
        [
            np.random.default_rng(seed).standard_normal((80, 3000)).astype(np.float32)
            for seed in range(batch_size)
        ],
        axis=0,
    )

    m_off = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type="float16",
        flash_attention=False,
    )
    m_on = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type="float16",
        flash_attention=True,
    )

    out_off = _encoder_output(m_off, mels)
    out_on = _encoder_output(m_on, mels)
    assert out_off.shape == out_on.shape == (batch_size, 1500, 1024)

    diff = np.abs(out_off - out_on)
    max_diff = float(diff.max())
    max_abs = float(np.abs(out_off).max()) + 1e-8
    rel = max_diff / max_abs
    print(
        f"\n[B={batch_size}] encoder max_abs_diff={max_diff:.4f}, "
        f"rel_diff={rel*100:.3f}%"
    )
    assert max_diff <= 0.5, f"B={batch_size} max diff {max_diff:.4f} > 0.5"


# ----------------------------------------------------------------------------
# Prompt-prefill correctness across a few prefill lengths.
# This exercises the seqlen_q path that's neither pure decode (seqlen_q=1)
# nor the WMMA fast path (seqlen_q >= 16), forcing the scalar-tiled and
# 3-pass fallback kernels to also get a correctness pass.
# ----------------------------------------------------------------------------
@test_utils.require_cuda
@require_model
@pytest.mark.parametrize("n_prompt", [1, 3, 5, 8])
def test_flash_attention_variable_prompt_length(n_prompt):
    """generate() must agree between Flash=ON and OFF for varying numbers
    of prompt tokens — covers seqlen_q = 1, 3, 5, 8 in the prefill step,
    which routes through the decode-kernel (1), 3-pass fallback (3, 5, 8)
    pieces of the dispatcher."""
    mel = _mel(seed=0)
    base = [50258, 50259, 50360, 50364, 1029, 290, 264, 7184]
    prompt = [base[:n_prompt]]

    m_off = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type="float16",
        flash_attention=False,
    )
    m_on = ctranslate2.models.Whisper(
        _MODEL_PATH,
        device="cuda",
        compute_type="float16",
        flash_attention=True,
    )

    r_off = m_off.generate(
        ctranslate2.StorageView.from_array(mel),
        prompt,
        beam_size=1,
        max_length=n_prompt + 5,
    )
    r_on = m_on.generate(
        ctranslate2.StorageView.from_array(mel),
        prompt,
        beam_size=1,
        max_length=n_prompt + 5,
    )
    tok_off = r_off[0].sequences_ids[0]
    tok_on = r_on[0].sequences_ids[0]
    assert (
        tok_off == tok_on
    ), f"n_prompt={n_prompt}: Flash=ON {tok_on} vs Flash=OFF {tok_off}"


# ----------------------------------------------------------------------------
# Informational test: what does Flash Attention save in peak HBM for
# attention score buffers?  Reported, not asserted — it's hardware-independent
# and serves as documentation of the algorithmic benefit.
# ----------------------------------------------------------------------------
def test_flash_attention_score_buffer_savings():
    """Print what Flash Attention's online softmax avoids materialising in
    HBM at typical Whisper / LLM shapes.  Useful context when reading the
    timing numbers."""
    cases = [
        ("Whisper-medium encoder layer", 1, 16, 1500, 1500),
        ("Whisper-medium decoder self-attn (max_length=200)", 1, 16, 1, 200),
        ("Whisper-medium decoder cross-attn", 1, 16, 1, 1500),
        ("Hypothetical LLM @ 4k context", 1, 32, 4096, 4096),
        ("Hypothetical LLM @ 32k context", 1, 32, 32768, 32768),
    ]
    print(
        "\nScore-matrix HBM footprint that Flash Attention's online softmax "
        "avoids materialising (per attention call):"
    )
    print(f"  {'shape':45s}  {'bytes':>10s}")
    for name, b, h, sq, sk in cases:
        n = _score_buffer_bytes(sq, sk, num_heads=h, batch=b)
        if n < 1024 * 1024:
            n_human = f"{n/1024:.1f} KiB"
        elif n < 1024 * 1024 * 1024:
            n_human = f"{n/1024/1024:.1f} MiB"
        else:
            n_human = f"{n/1024/1024/1024:.2f} GiB"
        print(f"  {name:45s}  {n_human:>10s}")

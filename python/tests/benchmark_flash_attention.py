"""Standalone benchmark script for the native HIP Flash Attention path.

Measures both speed and HBM footprint of `flash_attention=True` vs the
standard MultiHeadAttention oracle, on the encoder pass and on `generate`
at a few different max_length values.

This is *not* a pytest — it's run manually (`python benchmark_flash_attention.py`)
because the numbers are timing-sensitive and we don't want regression
flapping in CI.  pytest tests for correctness live in test_flash_attention.py.

HBM is measured via the HIP runtime's `hipMemGetInfo`, called through
ctypes.  On Windows the runtime DLL is `amdhip64_*.dll`; on Linux it's
`libamdhip64.so`.

Usage:
    python benchmark_flash_attention.py [--model PATH] [--runs N]
"""

import argparse
import ctypes
import os
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------
# Local-build DLL loader (mirrors test_flash_attention.py).
# ----------------------------------------------------------------------------
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

import ctranslate2  # noqa: E402


# ----------------------------------------------------------------------------
# HIP runtime memory query via ctypes.
# Returns (free_bytes, total_bytes) for the currently-active device.
# ----------------------------------------------------------------------------
def _load_hip_runtime():
    if sys.platform == "win32":
        # The DLL name on Windows has a major-version suffix that varies.
        site_dir = next(
            d for d in (
                os.path.join(s, "_rocm_sdk_core", "bin")
                for s in (__import__("site").getsitepackages()
                          + [__import__("site").getusersitepackages()])
            ) if os.path.isdir(d)
        )
        candidates = sorted(
            f for f in os.listdir(site_dir)
            if f.startswith("amdhip64") and f.endswith(".dll")
        )
        if not candidates:
            raise FileNotFoundError("amdhip64_*.dll not found in ROCm SDK bin")
        return ctypes.CDLL(os.path.join(site_dir, candidates[-1]))
    else:
        return ctypes.CDLL("libamdhip64.so")


_hip = _load_hip_runtime()
_hip.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                               ctypes.POINTER(ctypes.c_size_t)]
_hip.hipMemGetInfo.restype = ctypes.c_int
_hip.hipDeviceSynchronize.argtypes = []
_hip.hipDeviceSynchronize.restype = ctypes.c_int


def hbm_free_total():
    """(free_bytes, total_bytes) on the active HIP device."""
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    rc = _hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if rc != 0:
        raise RuntimeError(f"hipMemGetInfo returned {rc}")
    return free.value, total.value


def gpu_sync():
    _hip.hipDeviceSynchronize()


# ----------------------------------------------------------------------------
# Bench helpers.
# ----------------------------------------------------------------------------
def _mel(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 80, 3000)).astype(np.float32)


def _bench(fn, runs=20, warmup=3):
    for _ in range(warmup):
        fn()
    gpu_sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    gpu_sync()
    return (time.perf_counter() - t0) / runs * 1000  # ms per call


def measure_hbm_delta(loader_fn, work_fn):
    """Build a model with `loader_fn`, baseline its persistent HBM, run
    `work_fn(model)` once (so the allocator pool grows), then measure the
    additional HBM held.  Returns (model_bytes, work_bytes_added)."""
    gpu_sync()
    free_empty, _ = hbm_free_total()

    model = loader_fn()
    gpu_sync()
    free_after_load, _ = hbm_free_total()
    model_bytes = free_empty - free_after_load

    # Warm up enough to make the allocator pool reach its working-set peak.
    for _ in range(3):
        work_fn(model)
    gpu_sync()
    free_after_work, _ = hbm_free_total()
    work_bytes = free_after_load - free_after_work
    return model, model_bytes, work_bytes


def fmt_bytes(n):
    n = abs(n)
    if n < 1024 * 1024:
        return f"{n/1024:.1f} KiB"
    if n < 1024 * 1024 * 1024:
        return f"{n/1024/1024:.1f} MiB"
    return f"{n/1024/1024/1024:.2f} GiB"


# ----------------------------------------------------------------------------
# Main benchmark.
# ----------------------------------------------------------------------------
PROMPTS = [[50258, 50259, 50360]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a converted faster-whisper-medium snapshot. "
             "If omitted, the HuggingFace cache is searched.",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--compute-type", default="float16", choices=["float16", "bfloat16"],
    )
    args = parser.parse_args()

    if args.model is None:
        snaps = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--Systran--faster-whisper-medium/snapshots"
        )
        if not os.path.isdir(snaps):
            sys.exit("No --model given and faster-whisper-medium not cached.")
        shas = [d for d in os.listdir(snaps)
                if os.path.isfile(os.path.join(snaps, d, "model.bin"))]
        if not shas:
            sys.exit("No converted snapshot found in HF cache.")
        args.model = os.path.join(snaps, shas[0])

    print(f"Model: {args.model}")
    print(f"Compute type: {args.compute_type}, runs/case: {args.runs}\n")

    mel = _mel(seed=0)

    # ------------------------------------------------------------------
    # HBM measurement: load both variants, measure persistent + working set.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("HBM footprint")
    print("=" * 70)
    for flash in (False, True):
        label = "Flash=ON " if flash else "Flash=OFF"

        def load():
            return ctranslate2.models.Whisper(
                args.model, device="cuda",
                compute_type=args.compute_type, flash_attention=flash,
            )

        def work(m):
            r = m.generate(
                ctranslate2.StorageView.from_array(mel),
                PROMPTS, beam_size=1, max_length=100,
            )
            return r

        model, model_b, work_b = measure_hbm_delta(load, work)
        print(
            f"  {label}:  model = {fmt_bytes(model_b)},  "
            f"working set (generate max_length=100) = +{fmt_bytes(work_b)}"
        )
        del model
        gpu_sync()

    # Theoretical Flash Attention savings on the encoder score buffer.
    sq = sk = 1500
    h = 16
    saved_per_layer = sq * sk * h * 4
    print()
    print(
        f"  Score-matrix that the standard path materialises and Flash "
        f"avoids, per encoder layer:"
    )
    print(f"    {sq}x{sk}x{h} heads x FP32 = {fmt_bytes(saved_per_layer)} per layer")
    print(f"    24 layers => up to {fmt_bytes(24 * saved_per_layer)} of HBM "
          f"traffic per encoder pass")
    print()

    # ------------------------------------------------------------------
    # Performance: encoder-only, then generate() at several lengths.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Performance")
    print("=" * 70)

    m_off = ctranslate2.models.Whisper(
        args.model, device="cuda",
        compute_type=args.compute_type, flash_attention=False,
    )
    m_on = ctranslate2.models.Whisper(
        args.model, device="cuda",
        compute_type=args.compute_type, flash_attention=True,
    )

    def enc_fn(m):
        return lambda: m.encode(ctranslate2.StorageView.from_array(mel),
                                to_cpu=False)

    for batch in (1, 4):
        mel_b = np.stack([mel[0]] * batch, axis=0)

        def enc_b(m):
            return lambda: m.encode(ctranslate2.StorageView.from_array(mel_b),
                                    to_cpu=False)

        off_ms = _bench(enc_b(m_off), runs=args.runs)
        on_ms  = _bench(enc_b(m_on),  runs=args.runs)
        print(
            f"  encoder (B={batch}, Sq=Sk=1500):  "
            f"OFF={off_ms:6.2f} ms  ON={on_ms:6.2f} ms  speedup={off_ms/on_ms:.2f}x"
        )

    print()
    for max_len in (30, 100, 200, 448):
        def gen_b(m, ml=max_len):
            return lambda: m.generate(
                ctranslate2.StorageView.from_array(mel),
                PROMPTS, beam_size=1, max_length=ml,
            )
        off_ms = _bench(gen_b(m_off), runs=max(args.runs // 4, 3))
        on_ms  = _bench(gen_b(m_on),  runs=max(args.runs // 4, 3))
        print(
            f"  generate (max_length={max_len:3d}):  "
            f"OFF={off_ms:7.1f} ms  ON={on_ms:7.1f} ms  speedup={off_ms/on_ms:.2f}x"
        )


if __name__ == "__main__":
    main()

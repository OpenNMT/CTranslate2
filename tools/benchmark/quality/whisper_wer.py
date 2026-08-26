#!/usr/bin/env python3
"""Compare Whisper quality and speed on CPU and MPS.

This follows faster-whisper's WER benchmark setup: LibriSpeech clean
validation, English text normalization, and corpus WER. Model loading and one
warm-up utterance are reported separately from measured inference.
"""

import argparse
import gc
import hashlib
import io
import json
import platform
import time
import urllib.request
from pathlib import Path

import ctranslate2
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel, decode_audio
from jiwer import wer
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


NORMALIZER_REVISION = "ed9a06cd89a93e47838f564998a6c09b655d7f43"
NORMALIZER_SHA256 = "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"
NORMALIZER_URL = (
    "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/"
    f"{NORMALIZER_REVISION}/benchmark/normalizer.json"
)


def load_normalizer(path):
    if path is None:
        path = (
            Path.home()
            / ".cache"
            / "ctranslate2"
            / "benchmarks"
            / f"faster-whisper-normalizer-{NORMALIZER_REVISION}.json"
        )
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(NORMALIZER_URL, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != NORMALIZER_SHA256:
                raise RuntimeError("Downloaded normalizer has an unexpected checksum")
            path.write_bytes(data)

    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != NORMALIZER_SHA256:
        raise RuntimeError(f"Normalizer checksum does not match: {path}")
    return EnglishTextNormalizer(json.loads(data))


def load_audio(sample):
    encoded = sample["audio"]
    if encoded.get("bytes") is not None:
        source = io.BytesIO(encoded["bytes"])
    elif encoded.get("path") is not None:
        source = encoded["path"]
    else:
        raise RuntimeError("Dataset row contains neither audio bytes nor a path")
    return decode_audio(source, sampling_rate=16000)


def iter_dataset(args):
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    # Let PyAV (already required by faster-whisper) decode the FLAC bytes. This
    # avoids an additional TorchCodec dependency in recent datasets releases.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    for index, sample in enumerate(dataset):
        if args.max_samples is not None and index >= args.max_samples:
            break
        yield sample


def transcribe(model, audio, beam_size):
    segments, _ = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=beam_size,
        temperature=0,
        vad_filter=False,
        word_timestamps=False,
    )
    return "".join(segment.text for segment in segments)


def benchmark_device(args, device, compute_type, normalizer):
    if device == "mps":
        if not hasattr(ctranslate2, "get_mps_device_count"):
            raise RuntimeError("CTranslate2 was not built with MPS support")
        if ctranslate2.get_mps_device_count() < 1:
            raise RuntimeError("No Apple MPS device is available")

    load_start = time.perf_counter()
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        cpu_threads=args.cpu_threads,
    )
    load_seconds = time.perf_counter() - load_start

    warmup_sample = next(iter(iter_dataset(args)), None)
    if warmup_sample is None:
        raise RuntimeError("The selected dataset is empty")
    transcribe(model, load_audio(warmup_sample), args.beam_size)

    hypotheses = []
    references = []
    sample_ids = []
    audio_seconds = 0.0
    inference_seconds = 0.0
    evaluation_start = time.perf_counter()
    samples = tqdm(
        iter_dataset(args),
        total=args.max_samples,
        desc=f"{device.upper()} WER",
        unit="utterance",
    )
    for index, sample in enumerate(samples):
        audio = load_audio(sample)
        inference_start = time.perf_counter()
        hypothesis = transcribe(model, audio, args.beam_size)
        inference_seconds += time.perf_counter() - inference_start
        hypotheses.append(normalizer(hypothesis))
        references.append(normalizer(sample["text"]))
        sample_ids.append(str(sample.get("id", index)))
        audio_seconds += len(audio) / 16000
    evaluation_wall_seconds = time.perf_counter() - evaluation_start

    result = {
        "device": device,
        "compute_type": compute_type,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "audio_seconds": audio_seconds,
        "real_time_factor": inference_seconds / audio_seconds,
        "audio_speedup": audio_seconds / inference_seconds,
        "samples": len(hypotheses),
        "wer_percent": 100 * wer(reference=references, hypothesis=hypotheses),
        "sample_ids": sample_ids,
        "hypotheses": hypotheses,
        "references": references,
    }
    del model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="Systran/faster-whisper-small.en")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--cpu-compute-type", default="float32")
    parser.add_argument("--mps-compute-type", default="float16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--normalizer-json",
        type=Path,
        help="Pinned faster-whisper normalizer; downloaded and verified by default",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit rows for a smoke run; omit for publishable WER",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    normalizer = load_normalizer(args.normalizer_json)
    cpu = benchmark_device(args, "cpu", args.cpu_compute_type, normalizer)
    mps = benchmark_device(args, "mps", args.mps_compute_type, normalizer)

    if cpu["sample_ids"] != mps["sample_ids"]:
        raise RuntimeError("CPU and MPS did not evaluate the same dataset rows")
    if cpu["references"] != mps["references"]:
        raise RuntimeError("CPU and MPS references differ")

    agreement = sum(
        cpu_hypothesis == mps_hypothesis
        for cpu_hypothesis, mps_hypothesis in zip(cpu["hypotheses"], mps["hypotheses"])
    )
    summary = {
        "benchmark": "Whisper LibriSpeech clean WER",
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "beam_size": args.beam_size,
        "normalizer_revision": NORMALIZER_REVISION,
        "normalizer_sha256": NORMALIZER_SHA256,
        "max_samples": args.max_samples,
        "ctranslate2_version": ctranslate2.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "mps": mps,
        "mps_inference_speedup": cpu["inference_seconds"] / mps["inference_seconds"],
        "wer_delta_points": mps["wer_percent"] - cpu["wer_percent"],
        "exact_transcript_agreement_percent": 100 * agreement / cpu["samples"],
    }

    # The per-sample text is useful for investigating quality differences but
    # makes console output unreadable, so it is only retained in the JSON file.
    console_summary = json.loads(json.dumps(summary))
    for device in ("cpu", "mps"):
        del console_summary[device]["sample_ids"]
        del console_summary[device]["hypotheses"]
        del console_summary[device]["references"]
    print(json.dumps(console_summary, indent=2, sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

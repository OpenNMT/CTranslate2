#!/usr/bin/env python3
"""Compare CTranslate2 translation quality and speed on CPU and MPS."""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import ctranslate2
import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer


def get_wmt14_files():
    if not hasattr(sacrebleu, "get_source_file"):
        raise RuntimeError(
            "This benchmark requires the SacreBLEU compatibility dataset API"
        )
    source = sacrebleu.get_source_file("wmt14", langpair="en-de")
    reference = sacrebleu.get_reference_files("wmt14", langpair="en-de")[0]
    return Path(source), Path(reference)


def read_lines(path, max_samples):
    with path.open(encoding="utf-8") as stream:
        lines = [line.rstrip("\n") for line in stream]
    return lines if max_samples is None else lines[:max_samples]


def encode_source(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    return tokenizer.convert_ids_to_tokens(token_ids)


def decode_target(tokenizer, tokens):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def benchmark_device(args, device, compute_type, source_tokens, tokenizer):
    if device == "mps":
        if not hasattr(ctranslate2, "get_mps_device_count"):
            raise RuntimeError("CTranslate2 was not built with MPS support")
        if ctranslate2.get_mps_device_count() < 1:
            raise RuntimeError("No Apple MPS device is available")

    load_start = time.perf_counter()
    translator = ctranslate2.Translator(
        str(args.model),
        device=device,
        compute_type=compute_type,
        intra_threads=args.cpu_threads,
    )
    load_seconds = time.perf_counter() - load_start

    translate_options = {
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
    }
    translator.translate_batch(source_tokens[:1], **translate_options)

    hypotheses = []
    output_tokens = 0
    inference_start = time.perf_counter()
    offsets = range(0, len(source_tokens), args.batch_size)
    for offset in tqdm(
        offsets,
        desc=f"{device.upper()} BLEU",
        unit="batch",
    ):
        batch = source_tokens[offset : offset + args.batch_size]
        results = translator.translate_batch(batch, **translate_options)
        for result in results:
            tokens = result.hypotheses[0]
            output_tokens += len(tokens)
            hypotheses.append(decode_target(tokenizer, tokens))
    inference_seconds = time.perf_counter() - inference_start

    result = {
        "device": device,
        "compute_type": compute_type,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "sentences_per_second": len(hypotheses) / inference_seconds,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / inference_seconds,
        "hypotheses": hypotheses,
    }
    del translator
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--tokenizer",
        default="Helsinki-NLP/opus-mt-en-de",
        help="Tokenizer path or Hugging Face model name",
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-compute-type", default="float32")
    parser.add_argument("--mps-compute-type", default="float16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit sentences for a smoke run; omit for publishable BLEU",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if (args.source is None) != (args.reference is None):
        parser.error("--source and --reference must be specified together")
    source_path, reference_path = (
        (args.source, args.reference) if args.source is not None else get_wmt14_files()
    )

    sources = read_lines(source_path, args.max_samples)
    references = read_lines(reference_path, args.max_samples)
    if len(sources) != len(references):
        raise RuntimeError("Source and reference files have different line counts")
    if not sources:
        raise RuntimeError("The selected translation dataset is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source_tokens = [encode_source(tokenizer, source) for source in sources]
    cpu = benchmark_device(args, "cpu", args.cpu_compute_type, source_tokens, tokenizer)
    mps = benchmark_device(args, "mps", args.mps_compute_type, source_tokens, tokenizer)

    bleu_metric = sacrebleu.metrics.BLEU(force=True)
    cpu["bleu"] = bleu_metric.corpus_score(cpu["hypotheses"], [references]).score
    mps["bleu"] = bleu_metric.corpus_score(mps["hypotheses"], [references]).score
    agreement = sum(
        cpu_hypothesis == mps_hypothesis
        for cpu_hypothesis, mps_hypothesis in zip(cpu["hypotheses"], mps["hypotheses"])
    )
    summary = {
        "benchmark": "WMT14 English-German BLEU",
        "model": str(args.model),
        "tokenizer": args.tokenizer,
        "source": str(source_path),
        "reference": str(reference_path),
        "sentences": len(sources),
        "max_samples": args.max_samples,
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "batch_size": args.batch_size,
        "ctranslate2_version": ctranslate2.__version__,
        "sacrebleu_signature": str(bleu_metric.get_signature()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "mps": mps,
        "mps_inference_speedup": cpu["inference_seconds"] / mps["inference_seconds"],
        "bleu_delta": mps["bleu"] - cpu["bleu"],
        "exact_translation_agreement_percent": 100 * agreement / len(sources),
    }

    console_summary = json.loads(json.dumps(summary))
    del console_summary["cpu"]["hypotheses"]
    del console_summary["mps"]["hypotheses"]
    print(json.dumps(console_summary, indent=2, sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

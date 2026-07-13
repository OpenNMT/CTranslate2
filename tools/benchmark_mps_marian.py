#!/usr/bin/env python3
"""Reproducible single-process Marian/M2M100 CTranslate2 benchmark.

Run CPU and MPS in separate invocations so model loading and allocator state do
not influence the other device.  Kernel/profile output is written by the C++
backend to stderr when CT2_MPS_PROFILE or CT2_MPS_LOG_GEMM is enabled.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import ctranslate2
from transformers import M2M100Tokenizer


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def source_tokens(tokenizer, requested_length):
    phrase = "Da yaw azmoina jumla da. Sta num sa de?"
    ids = tokenizer(phrase, add_special_tokens=True)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if len(tokens) >= requested_length:
        return tokens[:requested_length]
    content = [token for token in tokens if token not in tokenizer.all_special_tokens]
    if not content:
        raise RuntimeError("tokenizer produced no content tokens")
    while len(tokens) < requested_length:
        tokens.insert(-1 if tokens else 0, content[(len(tokens) - 1) % len(content)])
    return tokens[:requested_length]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--source-lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-output-length", type=int, default=100)
    parser.add_argument("--profile-only", action="store_true",
                        help="run one warm inference for compact C++ profile capture")
    args = parser.parse_args()

    compute_type = args.compute_type or ("float16" if args.device == "mps" else "default")
    tokenizer = M2M100Tokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    tokenizer.src_lang = "ps"
    target_prefix = [[tokenizer.convert_ids_to_tokens([tokenizer.get_lang_id("en")])[0]]]
    translator = ctranslate2.Translator(
        str(args.model), device=args.device, compute_type=compute_type)

    cases = []
    for length in args.source_lengths:
        tokens = source_tokens(tokenizer, length)

        def translate():
            return translator.translate_batch(
                [tokens],
                target_prefix=target_prefix,
                beam_size=args.beam_size,
                max_decoding_length=args.max_output_length,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )[0]

        warmups = 1 if args.profile_only else args.warmups
        runs = 1 if args.profile_only else args.runs
        for _ in range(warmups):
            translate()

        latencies = []
        output_lengths = []
        for _ in range(runs):
            start = time.perf_counter_ns()
            result = translate()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            latencies.append(elapsed)
            output_lengths.append(len(result.hypotheses[0]))

        median_ms = statistics.median(latencies)
        mean_output = statistics.mean(output_lengths)
        cases.append({
            "source_length": length,
            "output_tokens_mean": mean_output,
            "latency_ms_median": median_ms,
            "latency_ms_p90": percentile(latencies, 0.90),
            "latency_ms_mean": statistics.mean(latencies),
            "latency_ms_stdev": statistics.pstdev(latencies),
            "output_tokens_per_second": mean_output / (median_ms / 1000.0),
        })

    print(json.dumps({
        "device": args.device,
        "compute_type": compute_type,
        "beam_size": args.beam_size,
        "warmups": 1 if args.profile_only else args.warmups,
        "runs": 1 if args.profile_only else args.runs,
        "max_output_length": args.max_output_length,
        "cases": cases,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

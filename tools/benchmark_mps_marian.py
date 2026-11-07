#!/usr/bin/env python3
"""Reproducible single-process Marian CTranslate2 benchmark.

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
from transformers import MarianTokenizer


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def source_tokens(tokenizer, requested_length):
    phrase = (
        "Paroon che za bazar ta talay wam, halta domra rush wo che pa motor ke "
        "ma taqreeban yaw ghanta pa lar ke ter kro, kho bia hum zama tol zaroori "
        "saman wakhisto."
    )
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(phrase, add_special_tokens=True)
    )
    content = [token for token in tokens if token not in tokenizer.all_special_tokens]
    if not content:
        raise RuntimeError("tokenizer produced no content tokens")
    eos = tokenizer.eos_token
    requested_content = max(0, requested_length - (1 if eos else 0))
    generated = [content[index % len(content)] for index in range(requested_content)]
    if eos:
        generated.append(eos)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument(
        "--source-lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-output-length", type=int, default=100)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="run one warm inference for compact C++ profile capture",
    )
    args = parser.parse_args()

    compute_type = args.compute_type or (
        "float16" if args.device == "mps" else "default"
    )
    tokenizer = MarianTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    translator = ctranslate2.Translator(
        str(args.model), device=args.device, compute_type=compute_type
    )

    cases = []
    for length in args.source_lengths:
        tokens = source_tokens(tokenizer, length)

        def translate():
            return translator.translate_batch(
                [tokens],
                beam_size=args.beam_size,
                max_decoding_length=args.max_output_length,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                return_scores=True,
            )[0]

        warmups = 1 if args.profile_only else args.warmups
        runs = 1 if args.profile_only else args.runs
        for _ in range(warmups):
            translate()

        latencies = []
        output_lengths = []
        hypotheses = []
        for _ in range(runs):
            start = time.perf_counter_ns()
            result = translate()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            latencies.append(elapsed)
            output_lengths.append(len(result.hypotheses[0]))
            hypotheses.append(list(result.hypotheses[0]))

        median_ms = statistics.median(latencies)
        mean_output = statistics.mean(output_lengths)
        cases.append(
            {
                "source_length": length,
                "output_tokens_mean": mean_output,
                "latency_ms_median": median_ms,
                "latency_ms_p90": percentile(latencies, 0.90),
                "latency_ms_mean": statistics.mean(latencies),
                "latency_ms_stdev": statistics.pstdev(latencies),
                "output_tokens_per_second": mean_output / (median_ms / 1000.0),
                "outputs_consistent": len({tuple(output) for output in hypotheses})
                == 1,
                "hypothesis": hypotheses[0],
            }
        )

    print(
        json.dumps(
            {
                "device": args.device,
                "compute_type": compute_type,
                "beam_size": args.beam_size,
                "warmups": 1 if args.profile_only else args.warmups,
                "runs": 1 if args.profile_only else args.runs,
                "max_output_length": args.max_output_length,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

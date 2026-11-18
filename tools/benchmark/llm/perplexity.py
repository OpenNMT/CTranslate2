#!/usr/bin/env python3
"""Score a pinned token corpus with overlapping, bounded-memory windows."""

import argparse
import hashlib
import json
import math
import time

from pathlib import Path


def windows(size, context, stride):
    """Yield (start, end, skip): score each target after token zero once."""
    if size < 2 or context < 2 or not 1 <= stride < context:
        raise ValueError("Require two tokens, context >= 2 and 1 <= stride < context")
    previous_end = 1
    for start in range(0, size - 1, stride):
        end = min(start + context, size)
        yield start, end, max(0, previous_end - start - 1)
        previous_end = end
        if end == size:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--compute-type")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--kv-cache-type", default="auto")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.max_tokens < 0:
        parser.error("max-tokens must be nonnegative")

    from benchmark import metadata
    from transformers import AutoTokenizer

    import ctranslate2

    manifest = json.loads(args.corpus.read_text())
    tokenizer_hash = hashlib.sha256(
        (args.tokenizer / "tokenizer.json").read_bytes()
    ).hexdigest()
    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Corpus tokenizer_sha256 does not match tokenizer.json")
    all_ids = manifest["tokens"]
    ids = all_ids[: args.max_tokens] if args.max_tokens else all_ids
    plan = list(windows(len(ids), args.context, args.stride))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if tokenizer.convert_tokens_to_ids(tokens) != ids:
        raise ValueError("Corpus token IDs do not round-trip through this tokenizer")
    model = ctranslate2.Generator(
        str(args.model),
        device=args.device,
        compute_type=args.compute_type
        or ("float16" if args.device == "mps" else "float32"),
        flash_attention=args.device == "mps",
        kv_cache_type=args.kv_cache_type,
        inter_threads=1,
        intra_threads=4,
    )
    report = {
        **metadata(),
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "corpus": {key: value for key, value in manifest.items() if key != "tokens"},
        "corpus_file_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "compute_type": model.compute_type,
        "kv_cache_type": model.kv_cache_type,
        "context": args.context,
        "stride": args.stride,
        "available_tokens": len(all_ids),
        "evaluated_tokens": len(ids),
        "full_corpus": len(ids) == len(all_ids),
        "complete": False,
        "windows": [],
        "notes": [
            "Teacher-forced score_batch; every target after token zero is scored once.",
            "Context resets at each window; overlap is context, not additional targets.",
            "This quality run is not a throughput benchmark.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_targets = 0
    for index, (start, end, skip) in enumerate(plan):
        score = model.score_batch([tokens[start:end]], max_input_length=0)[0]
        if len(score.log_probs) != end - start - 1:
            raise ValueError("Scoring changed the sequence length or special tokens")
        probabilities = score.log_probs[skip:]
        if not all(math.isfinite(value) for value in probabilities):
            raise ValueError("Nonfinite corpus log probabilities")
        total_targets += len(probabilities)
        report["windows"].append(
            {
                "start": start,
                "end": end,
                "targets": len(probabilities),
                "nll_sum": -math.fsum(probabilities),
            }
        )
        report["scored_targets"] = total_targets
        report["nll"] = (
            math.fsum(w["nll_sum"] for w in report["windows"]) / total_targets
        )
        report["perplexity"] = math.exp(report["nll"])
        report["elapsed_s"] = time.perf_counter() - started
        if index + 1 == len(plan):
            if total_targets != len(ids) - 1:
                raise ValueError("Corpus targets were omitted or scored twice")
            report["complete"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if index % 16 == 0 or report["complete"]:
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "scored_targets",
                            "perplexity",
                            "elapsed_s",
                            "complete",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

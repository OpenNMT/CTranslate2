#!/usr/bin/env python3
"""Real decoder-only model timing. Run each device in a separate process."""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time

from pathlib import Path


def make_prompts(tokenizer, lengths):
    # Persist IDs: comparisons must consume identical tokens, not independently
    # tokenized text or a backend's default chat template/BOS policy.
    context = tokenizer.encode(
        "The observatory recorded the temperature and wind every morning. "
        "The researchers compared these measurements across several seasons. ",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        "\nExplain how repeated measurements help identify reliable trends.\n",
        add_special_tokens=False,
    )
    prompts = []
    for length in lengths:
        if length <= 0:
            raise ValueError("Prompt lengths must be positive")
        suffix = question[-min(length, len(question)) :]
        needed = length - len(suffix)
        ids = (context * ((needed + len(context) - 1) // len(context)))[:needed]
        prompts.append(ids + suffix)
    return prompts


def summarize_run(start, callbacks, end, prompt_length, output_ids):
    if len(callbacks) != len(output_ids) or not callbacks:
        raise RuntimeError("Expected one callback for every generated token")
    gaps = [b - a for a, b in zip(callbacks, callbacks[1:])]
    ttft = callbacks[0] - start
    decode_time = callbacks[-1] - callbacks[0]
    return {
        "prompt_tokens": prompt_length,
        "output_tokens": len(output_ids),
        "ttft_s": ttft,
        # TTFT includes prefill + the first decode + sampling + callback delivery.
        # This is intentionally NOT called isolated prefill throughput.
        "prompt_tokens_per_ttft_s": prompt_length / ttft,
        "decode_tokens_per_s": len(gaps) / decode_time if gaps else None,
        "per_token_latency_s": gaps,
        "median_per_token_s": statistics.median(gaps) if gaps else None,
        "total_s": end - start,
        "output_ids": output_ids,
    }


def metadata():
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repository)
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    library = None
    if platform.system() == "Darwin":
        import ctypes

        process = ctypes.CDLL(None)
        process._dyld_image_count.restype = ctypes.c_uint32
        process._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        process._dyld_get_image_name.restype = ctypes.c_char_p
        for index in range(process._dyld_image_count()):
            path = Path(os.fsdecode(process._dyld_get_image_name(index)))
            if path.name.startswith("libctranslate2.") and path.is_file():
                library = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                break
    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "commit": commit,
        "dirty": dirty,
        "loaded_library": library,
        "runtime_controls": {
            k: v for k, v in os.environ.items() if k.startswith("CT2_MPS_")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument(
        "--kv-cache-type",
        choices=["auto", "float16", "int8", "int5", "int4"],
        default="auto",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 512, 2048])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topp", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument(
        "--validate-logits",
        action="store_true",
        help="Capture/check every generated logit; timings include extra GPU copies",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output_tokens < 2 or args.repeats < 1:
        parser.error("Use at least two output tokens and one repeat")

    from transformers import AutoTokenizer

    import ctranslate2

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = (
        json.loads(args.prompts.read_text())
        if args.prompts
        else make_prompts(tokenizer, args.lengths)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.output.with_suffix(".prompts.json")
    manifest.write_text(json.dumps(prompts) + "\n")
    report = {
        **metadata(),
        "model": str(args.model.resolve()),
        "tokenizer": args.tokenizer,
        "ct2_version": ctranslate2.__version__,
        "ct2_module": ctranslate2.__file__,
        "device": args.device,
        "requested_compute_type": args.compute_type,
        "flash_attention": args.flash_attention,
        "requested_kv_cache_type": args.kv_cache_type,
        "logits_validation": args.validate_logits,
        "prompt_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sampling": {
            "seed": args.seed,
            "topk": args.topk,
            "topp": args.topp,
            "temperature": args.temperature,
        },
        "runs": [],
        "notes": [
            "Model load and tokenizer excluded from generation timing.",
            "Callbacks add Python overhead; use identical settings for comparisons.",
            "First request is cold; later first-per-shape requests can compile pipelines.",
            "RSS is process resident memory, not total peak unified/GPU memory.",
            "KV bytes are an estimate from model geometry, not allocator telemetry.",
        ],
    }
    started = time.perf_counter()
    generator = ctranslate2.Generator(
        str(args.model),
        device=args.device,
        compute_type=args.compute_type,
        inter_threads=1,
        intra_threads=args.threads,
        flash_attention=args.flash_attention,
        kv_cache_type=args.kv_cache_type,
    )
    report["load_s"] = time.perf_counter() - started
    report["resolved_compute_type"] = generator.compute_type
    report["resolved_kv_cache_type"] = generator.kv_cache_type
    config = json.loads((Path(args.tokenizer) / "config.json").read_text())
    head_dim = config.get(
        "head_dim", config["hidden_size"] // config["num_attention_heads"]
    )
    # The generic CT2 attention layer currently replicates GQA heads in its cache.
    report["logical_kv_bytes_per_token_fp16"] = (
        2
        * config["num_hidden_layers"]
        * config.get("num_key_value_heads", config["num_attention_heads"])
        * head_dim
        * 2
    )
    cache_bits = {"int8": 8, "int5": 5, "int4": 4}.get(generator.kv_cache_type, 0)
    row_bytes = (
        ((head_dim + 31) // 32) * (4 + 4 * cache_bits)
        if cache_bits
        else (4 if generator.kv_cache_type == "float32" else 2) * head_dim
    )
    report["logical_kv_bytes_per_token"] = (
        2
        * config["num_hidden_layers"]
        * config.get("num_key_value_heads", config["num_attention_heads"])
        * row_bytes
    )
    report["kv_cache_bytes_per_token_estimate"] = (
        2
        * config["num_hidden_layers"]
        * row_bytes
        * (
            config.get("num_key_value_heads", config["num_attention_heads"])
            if args.flash_attention
            else config["num_attention_heads"]
        )
    )
    report["replicated_kv_bytes_per_token_fp16"] = (
        2 * config["num_hidden_layers"] * config["num_attention_heads"] * head_dim * 2
    )
    report["compact_flash_cache_capacity_multiple"] = (
        512 if args.flash_attention else None
    )
    for prompt in prompts:
        tokens = tokenizer.convert_ids_to_tokens(prompt)
        first_ids = None
        for repeat in range(args.repeats + 1):
            callbacks = []

            def callback(step):
                callbacks.append(time.perf_counter())
                return False

            ctranslate2.set_random_seed(args.seed)
            started = time.perf_counter()
            result = generator.generate_batch(
                [tokens],
                include_prompt_in_result=False,
                beam_size=1,
                max_length=args.output_tokens,
                min_length=args.output_tokens,
                end_token=[],
                sampling_topk=args.topk,
                sampling_topp=args.topp,
                sampling_temperature=args.temperature,
                callback=callback,
                return_logits_vocab=args.validate_logits,
            )[0]
            ended = time.perf_counter()
            run = summarize_run(
                started, callbacks, ended, len(prompt), result.sequences_ids[0]
            )
            if first_ids is None:
                first_ids = run["output_ids"]
            run["matches_first_greedy"] = (
                run["output_ids"] == first_ids if args.topk == 1 else None
            )
            if args.validate_logits:
                import numpy as np

                from validate import host_array

                logits = np.stack([host_array(x).reshape(-1) for x in result.logits[0]])
                run["finite_logit_fraction"] = float(np.isfinite(logits).mean())
                np.savez(
                    args.output.with_name(
                        args.output.stem + f"-{len(prompt)}-{repeat}.npz"
                    ),
                    logits=logits,
                )
            run.update(
                repeat=repeat,
                phase="first_request_for_shape" if repeat == 0 else "warm",
            )
            run["text"] = tokenizer.decode(run["output_ids"])
            report["runs"].append(run)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["peak_process_rss_bytes"] = (
                rss if platform.system() == "Darwin" else rss * 1024
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: run[k]
                        for k in [
                            "prompt_tokens",
                            "phase",
                            "ttft_s",
                            "decode_tokens_per_s",
                        ]
                    }
                ),
                flush=True,
            )
            if (
                run["matches_first_greedy"] is False
                or run.get("finite_logit_fraction", 1) < 1
            ):
                raise RuntimeError(
                    "Nonfinite logits or cold/warm token disagreement; failing run saved"
                )


if __name__ == "__main__":
    main()

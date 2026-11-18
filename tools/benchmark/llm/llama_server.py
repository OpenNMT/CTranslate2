#!/usr/bin/env python3
"""Replay the exact CT2 prompt IDs against a local llama.cpp server."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from benchmark import summarize_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topp", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    args = parser.parse_args()
    report = {
        "backend": "llama.cpp/server",
        "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "sampling": {
            "seed": args.seed,
            "topk": args.topk,
            "topp": args.topp,
            "temperature": args.temperature,
        },
        "runs": [],
        "notes": [
            "TTFT includes localhost HTTP/SSE delivery.",
            "Server must use the identical source model, F16 weights, F16 KV, one slot, and all layers on Metal.",
            "Prompt cache disabled; server prompt throughput excludes first-token decode.",
        ],
    }
    with urllib.request.urlopen(args.url + "/props", timeout=30) as response:
        report["server_props"] = json.load(response)
    for prompt in json.loads(args.prompts.read_text()):
        for repeat in range(args.repeats + 1):
            payload = {
                "prompt": prompt,
                "n_predict": args.output_tokens,
                "stream": True,
                "return_tokens": True,
                "cache_prompt": False,
                "ignore_eos": True,
                "seed": args.seed,
                "top_k": args.topk,
                "top_p": args.topp,
                "temperature": args.temperature,
                "min_p": 0,
                "repeat_penalty": 1,
                "typical_p": 1,
            }
            request = urllib.request.Request(
                args.url + "/completion",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = time.perf_counter()
            ids, callbacks, final = [], [], None
            with urllib.request.urlopen(request, timeout=1800) as response:
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    packet = json.loads(raw[6:])
                    now = time.perf_counter()
                    if "error" in packet:
                        raise RuntimeError(packet["error"])
                    tokens = packet.get("tokens", [])
                    # A final packet can repeat the complete token list in some
                    # server versions. Streaming deltas must be one token each.
                    if not packet.get("stop"):
                        if len(tokens) != 1:
                            raise RuntimeError("Expected one token per SSE delta")
                        ids.extend(tokens)
                        callbacks.append(now)
                    else:
                        final = packet
            if len(ids) != args.output_tokens or final is None:
                raise RuntimeError("Incomplete llama.cpp generation")
            # Detect implicit BOS insertion or accidental prompt-cache reuse.
            if final.get("tokens_evaluated") != len(prompt):
                raise RuntimeError("llama.cpp evaluated a different prompt length")
            run = summarize_run(
                started, callbacks, time.perf_counter(), len(prompt), ids
            )
            run.update(
                phase="first_request_for_shape" if repeat == 0 else "warm",
                repeat=repeat,
                server_timings=final.get("timings"),
            )
            report["runs"].append(run)
            args.output.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()

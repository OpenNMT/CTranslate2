#!/usr/bin/env python3
"""Export real-model logits/reference tokens, or compare two isolated runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def host_array(storage):
    from ctranslate2 import Device

    return np.array(storage.to_device(Device.cpu), copy=True).astype(np.float32)


def log_softmax(values):
    values = values.astype(np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def compare(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite logits")
    difference = candidate.astype(np.float64) - reference
    p = np.exp(log_softmax(reference))
    kl = np.sum(p * (log_softmax(reference) - log_softmax(candidate)), axis=-1)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_kl": float(np.mean(kl)),
        "argmax_agreement": float(
            np.mean(reference.argmax(-1) == candidate.argmax(-1))
        ),
    }


def export(args):
    from transformers import AutoTokenizer
    from benchmark import make_prompts, metadata

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = make_prompts(tokenizer, args.lengths)
    arrays = {}
    report = {"backend": args.backend, "device": args.device, "cases": []}
    if args.backend == "ct2":
        import ctranslate2

        model = ctranslate2.Generator(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            inter_threads=1,
            intra_threads=4,
            flash_attention=args.flash_attention,
            kv_cache_type=args.kv_cache_type,
        )
        forward = lambda ids: host_array(model.forward_batch([ids]))[0]
        report.update(
            ct2_module=ctranslate2.__file__,
            compute_type=model.compute_type,
            kv_cache_type=model.kv_cache_type,
        )
        report.update(metadata())
    else:
        import torch
        from transformers import AutoModelForCausalLM

        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            local_files_only=True,
            attn_implementation="eager",
        ).eval()

        def forward(ids):
            with torch.inference_mode():
                return model(torch.tensor([ids]), use_cache=False).logits[0].numpy()

    validation_failures = []
    for i, ids in enumerate(prompts):
        arrays[f"ids_{i}"] = np.array(ids, dtype=np.int32)
        case = {"length": len(ids)}
        validation_failure = None
        if not args.cached_only:
            full = forward(ids)
            arrays[f"logits_{i}"] = full
            # Bound the FP64 softmax workspace for large vocabularies/contexts.
            nll = 0.0
            for begin in range(0, len(ids) - 1, 32):
                end = min(begin + 32, len(ids) - 1)
                nll -= float(
                    log_softmax(full[begin:end])[
                        np.arange(end - begin), ids[begin + 1 : end + 1]
                    ].sum()
                )
            nll /= len(ids) - 1
            case.update(nll=nll, perplexity=float(np.exp(nll)))
        if args.backend == "ct2":
            tokens = tokenizer.convert_ids_to_tokens(ids)
            result = model.generate_batch(
                [tokens],
                include_prompt_in_result=False,
                max_length=args.output_tokens,
                min_length=args.output_tokens,
                end_token=[],
                beam_size=1,
                sampling_topk=1,
                return_logits_vocab=True,
            )[0]
            generated = result.sequences_ids[0]
            cached = np.stack([host_array(x).reshape(-1) for x in result.logits[0]])
            arrays[f"cached_{i}"] = cached
            arrays[f"generated_{i}"] = np.array(generated, dtype=np.int32)
            case.update(
                generated_ids=generated, finite_logits=bool(np.isfinite(cached).all())
            )
            if not case["finite_logits"]:
                raise AssertionError("Nonfinite generated logits")
            if args.cached_only:
                # Compare this with reference_moe.py --last-logits-only. The
                # independent CPU reference still checks cached vs full sequence.
                arrays[f"last_logits_{i}"] = cached[:1]
            else:
                all_logits = forward(ids + generated[:-1])
                expected = all_logits[len(ids) - 1 :]
                cache_check = compare(expected, cached)
                arrays[f"full_cache_{i}"] = expected
                case.update(
                    cache_vs_full=cache_check,
                    full_sequence_next_ids=expected.argmax(-1).tolist(),
                )
                if cache_check["max_abs"] > args.cache_atol:
                    validation_failure = f"KV cache/full sequence disagreement: {case}"
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output.with_suffix(".npz"), **arrays)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if validation_failure:
            validation_failures.append(validation_failure)
    if validation_failures:
        raise AssertionError("\n".join(validation_failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    dump = sub.add_parser("export")
    dump.add_argument("--backend", choices=["ct2", "transformers"], default="ct2")
    dump.add_argument("--model", required=True)
    dump.add_argument("--tokenizer", required=True)
    dump.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    dump.add_argument("--compute-type", default="float32")
    dump.add_argument("--flash-attention", action="store_true")
    dump.add_argument(
        "--kv-cache-type",
        choices=["auto", "float16", "int8", "int5", "int4"],
        default="auto",
    )
    dump.add_argument("--lengths", nargs="+", type=int, default=[7, 33, 65])
    dump.add_argument("--output-tokens", type=int, default=4)
    dump.add_argument("--cache-atol", type=float, default=0.2)
    dump.add_argument(
        "--cached-only",
        action="store_true",
        help="Export generated logits only, for a memory-bounded independent reference",
    )
    dump.add_argument("--output", required=True, type=Path)
    diff = sub.add_parser("compare")
    diff.add_argument("reference", type=Path)
    diff.add_argument("candidate", type=Path)
    diff.add_argument("--max-abs", type=float, default=0.2)
    diff.add_argument("--rmse", type=float, default=0.025)
    diff.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        if args.cached_only and args.backend != "ct2":
            parser.error("--cached-only requires the CT2 backend")
        export(args)
        return
    results = {}
    failed = False
    with np.load(args.reference) as ref, np.load(args.candidate) as cand:
        if not set(ref.files).issubset(cand.files):
            raise ValueError("Candidate is missing reference cases")
        for key in ref.files:
            if key.startswith("ids_") or key.startswith("generated_"):
                equal = np.array_equal(ref[key], cand[key])
                results[key] = {"exact_match": equal}
                failed |= not equal
            else:
                results[key] = compare(ref[key], cand[key])
                failed |= (
                    results[key]["max_abs"] > args.max_abs
                    or results[key]["rmse"] > args.rmse
                )
    result = json.dumps(results, indent=2) + "\n"
    print(result)
    if args.output:
        args.output.write_text(result)
    if failed:
        raise SystemExit("Logit tolerance or greedy token comparison failed")


if __name__ == "__main__":
    main()

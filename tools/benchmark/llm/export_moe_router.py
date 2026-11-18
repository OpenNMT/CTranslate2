#!/usr/bin/env python3
"""Export actual first-layer Qwen3 MoE router inputs for native Metal validation.

This runs embeddings, attention and both norms with source weights on CPU. It
stops before expert execution; it is not end-to-end CT2 MoE model support.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)

from benchmark import make_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 128, 512, 2048])
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    if not args.lengths or any(n <= 0 for n in args.lengths):
        parser.error("Lengths must be positive")
    torch.set_num_threads(4)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True)
    if (
        config.model_type != "qwen3_moe"
        or not config.norm_topk_prob
        or config.decoder_sparse_step != 1
        or 0 in config.mlp_only_layers
    ):
        parser.error("Requires Qwen3 MoE with a sparse first layer and normalized TopK")
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    prompts = make_prompts(tokenizer, args.lengths)
    attention = Qwen3MoeAttention(config, layer_idx=0).eval()
    before = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    after = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    rotary = Qwen3MoeRotaryEmbedding(config).eval()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "snapshot": str(args.snapshot.resolve()),
        "revision": args.snapshot.name,
        "scope": "actual first-layer router; CPU FP32 source attention/norms; no expert execution",
        "cases": [],
    }
    # This pinned small model has one safetensors file. Never load the complete
    # 2.4B model merely to validate its first router.
    with safe_open(
        args.snapshot / "model.safetensors", framework="pt", device="cpu"
    ) as f:
        prefix = "model.layers.0."
        attention.load_state_dict(
            {
                name: f.get_tensor(prefix + "self_attn." + name).float()
                for name in attention.state_dict()
            }
        )
        before.weight.data.copy_(
            f.get_tensor(prefix + "input_layernorm.weight").float()
        )
        after.weight.data.copy_(
            f.get_tensor(prefix + "post_attention_layernorm.weight").float()
        )
        router = f.get_tensor(prefix + "mlp.gate.weight").float()
        embedding = f.get_slice("model.embed_tokens.weight")
        rows = {i: embedding[i : i + 1].float() for i in set(sum(prompts, []))}
        for ids in prompts:
            with torch.inference_mode():
                hidden = torch.cat([rows[i] for i in ids]).unsqueeze(0)
                positions = torch.arange(len(ids)).unsqueeze(0)
                mask = torch.full(
                    (len(ids), len(ids)), torch.finfo(torch.float32).min
                ).triu_(1)
                attended, _ = attention(
                    before(hidden), rotary(hidden, positions), mask[None, None]
                )
                normalized = after(hidden + attended)
                logits = torch.nn.functional.linear(normalized, router).reshape(
                    -1, config.num_experts
                )
                probabilities = torch.softmax(logits, dim=-1)
                weights, selected = torch.topk(
                    probabilities, config.num_experts_per_tok, dim=-1
                )
                weights /= weights.sum(dim=-1, keepdim=True)
                # CT2's documented tie policy is lowest expert index. Report
                # an incompatible source tie rather than silently changing it.
                stable = torch.argsort(logits, descending=True, stable=True)[
                    :, : config.num_experts_per_tok
                ]
                if not torch.equal(stable, selected):
                    raise RuntimeError(
                        "Source TopK tie ordering differs from CT2's deterministic policy"
                    )
            path = args.output_prefix.with_name(
                args.output_prefix.name + f"-{len(ids)}.bin"
            )
            data = (
                b"CT2MOE1\0"
                + struct.pack(
                    "<III", len(ids), config.num_experts, config.num_experts_per_tok
                )
                + logits.numpy().astype("<f4").tobytes()
                + selected.numpy().astype("<i4").tobytes()
                + weights.numpy().astype("<f4").tobytes()
            )
            path.write_bytes(data)
            manifest["cases"].append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "token_ids": ids,
                    "expert_counts": np.bincount(
                        selected.numpy().ravel(), minlength=config.num_experts
                    ).tolist(),
                }
            )
            print(json.dumps({"path": str(path), "rows": len(ids)}), flush=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

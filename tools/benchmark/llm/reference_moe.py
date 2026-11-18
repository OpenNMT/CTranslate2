#!/usr/bin/env python3
"""Memory-bounded CPU FP32 Qwen3 MoE reference using source Transformers layers.

Weights are loaded one layer at a time. This is an independent correctness
reference, not a timed inference implementation or a CTranslate2 fallback.
"""

import argparse
import gc
import json

from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from benchmark import make_prompts
from kv_reference import QuantizedReferenceCache
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)
from validate import compare, log_softmax


class StreamingQwenMoe:
    def __init__(self, snapshot, int4=False, kv_cache_bits=0):
        self.snapshot = Path(snapshot)
        self.int4 = int4
        self.kv_cache_bits = kv_cache_bits
        self.config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        if self.config.model_type != "qwen3_moe":
            raise ValueError("This reference supports Qwen3 MoE source checkpoints")
        if self.config.use_sliding_window or any(
            kind != "full_attention"
            for kind in (getattr(self.config, "layer_types", None) or [])
        ):
            raise ValueError("This reference requires full attention in every layer")
        self.config._attn_implementation = "eager"
        self.rotary = Qwen3MoeRotaryEmbedding(self.config).eval()
        self.embedding_rows = {}

    def _layer_tensor(self, checkpoint, available, name):
        if name in available:
            return checkpoint.get_tensor(name).float()
        # Transformers 5.x stores experts in stacked parameters in memory,
        # while canonical safetensors checkpoints retain per-expert keys.
        for suffix in ("gate_up_proj", "down_proj"):
            if name.endswith("mlp.experts." + suffix):
                base = name[: -len(suffix)]
                projections = (
                    ("gate_proj", "up_proj")
                    if suffix == "gate_up_proj"
                    else ("down_proj",)
                )
                values = [
                    torch.cat(
                        [
                            checkpoint.get_tensor(
                                f"{base}{expert}.{projection}.weight"
                            ).float()
                            for projection in projections
                        ],
                        dim=0,
                    )
                    for expert in range(self.config.num_experts)
                ]
                return torch.stack(values)
        raise KeyError(f"Missing source layer tensor: {name}")

    def make_cache(self):
        return (
            QuantizedReferenceCache(self.config, self.kv_cache_bits)
            if self.kv_cache_bits
            else DynamicCache(config=self.config)
        )

    @torch.inference_mode()
    def forward(self, ids, cache=None, offset=0, last_logits=0):
        if self.kv_cache_bits and cache is None:
            cache = self.make_cache()
        c = self.config
        # Source checkpoints used here are a single safetensors file. No full
        # model allocation and no quantization/converter code is reused.
        with safe_open(
            self.snapshot / "model.safetensors", framework="pt", device="cpu"
        ) as f:
            available = set(f.keys())
            embeddings = f.get_slice("model.embed_tokens.weight")
            for token in set(ids):
                if token not in self.embedding_rows:
                    self.embedding_rows[token] = embeddings[token : token + 1].float()
            hidden = torch.cat([self.embedding_rows[token] for token in ids]).unsqueeze(
                0
            )
            positions = torch.arange(offset, offset + len(ids)).unsqueeze(0)
            keys = torch.arange(offset + len(ids))
            allowed = keys[None, :] <= positions[0, :, None]
            mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]
            position_embeddings = self.rotary(hidden, positions)
            for index in range(c.num_hidden_layers):
                with torch.device("meta"):
                    layer = Qwen3MoeDecoderLayer(c, index)
                prefix = f"model.layers.{index}."
                state = {
                    name: self._layer_tensor(f, available, prefix + name)
                    for name in layer.state_dict()
                }
                if self.int4:
                    for name, value in state.items():
                        if value.ndim < 2 or name == "mlp.gate.weight":
                            continue
                        flat = value.reshape(-1, value.shape[-1])
                        for begin in range(0, flat.shape[1], 64):
                            block = flat[:, begin : begin + 64]
                            scale = block.abs().amax(-1, keepdim=True) / 7
                            scale[scale == 0] = 1
                            block.copy_((block / scale).round().clamp(-7, 7) * scale)
                        value.copy_(value.half().float())
                layer.load_state_dict(state, assign=True)
                layer.eval()
                hidden = layer(
                    hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=mask,
                    position_ids=positions,
                    cache_position=positions[0],
                    past_key_values=cache,
                )
                del layer, state
            norm = Qwen3MoeRMSNorm(c.hidden_size, eps=c.rms_norm_eps).eval()
            norm.weight.data.copy_(f.get_tensor("model.norm.weight").float())
            hidden = norm(hidden)[0]
            if last_logits:
                hidden = hidden[-last_logits:]
            projection = (
                "lm_head.weight"
                if "lm_head.weight" in f.keys()
                else "model.embed_tokens.weight"
            )
            matrix = f.get_slice(projection)
            output = np.empty((hidden.shape[0], c.vocab_size), dtype=np.float32)
            for begin in range(0, c.vocab_size, 4096):
                end = min(begin + 4096, c.vocab_size)
                block = matrix[begin:end].float()
                output[:, begin:end] = torch.nn.functional.linear(hidden, block).numpy()
            return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--int4",
        action="store_true",
        help="Independent group-64 symmetric INT4 weight reference",
    )
    parser.add_argument("--kv-cache-bits", type=int, choices=[0, 4, 5, 8], default=0)
    parser.add_argument("--lengths", nargs="+", type=int, default=[7, 33])
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument(
        "--last-logits-only",
        action="store_true",
        help="Project/save only the final prefill position; omit prompt perplexity",
    )
    args = parser.parse_args()
    if args.output_tokens < 1 or any(length < 2 for length in args.lengths):
        parser.error("Use positive output tokens and lengths of at least two")
    torch.set_num_threads(4)
    reference = StreamingQwenMoe(
        args.snapshot, int4=args.int4, kv_cache_bits=args.kv_cache_bits
    )
    if args.prompts:
        prompts = json.loads(args.prompts.read_text())
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
        prompts = make_prompts(tokenizer, args.lengths)
    report = {
        "backend": "transformers_streaming_fp32",
        "device": "cpu",
        "weight_reference": "int4_group64_rounded_fp16"
        if args.int4
        else "source_float32",
        "kv_cache_bits": args.kv_cache_bits,
        "snapshot": str(args.snapshot.resolve()),
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "threads": torch.get_num_threads(),
        "routing": {
            "experts": reference.config.num_experts,
            "topk": reference.config.num_experts_per_tok,
            "normalize_topk": reference.config.norm_topk_prob,
        },
        "cases": [],
    }
    arrays = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, ids in enumerate(prompts):
        cache = reference.make_cache()
        full = reference.forward(
            ids, cache, last_logits=1 if args.last_logits_only else 0
        )
        cached, generated = [full[-1]], []
        for step in range(args.output_tokens):
            token = int(cached[-1].argmax())
            generated.append(token)
            if step + 1 < args.output_tokens:
                cached.append(reference.forward([token], cache, len(ids) + step)[-1])
        del cache
        gc.collect()
        cached = np.stack(cached)
        expected = reference.forward(
            ids + generated[:-1], last_logits=args.output_tokens
        )
        nll = None
        if not args.last_logits_only:
            nll = 0.0
            for begin in range(0, len(ids) - 1, 32):
                end = min(begin + 32, len(ids) - 1)
                nll -= float(
                    log_softmax(full[begin:end])[
                        np.arange(end - begin), ids[begin + 1 : end + 1]
                    ].sum()
                )
            nll /= len(ids) - 1
        arrays.update(
            {
                f"ids_{i}": np.asarray(ids, dtype=np.int32),
                (f"last_logits_{i}" if args.last_logits_only else f"logits_{i}"): full,
                f"cached_{i}": cached,
                f"generated_{i}": np.asarray(generated, dtype=np.int32),
            }
        )
        case = {
            "length": len(ids),
            "generated_ids": generated,
            "nll": nll,
            "perplexity": float(np.exp(nll)) if nll is not None else None,
            "cached_vs_full": compare(expected, cached),
        }
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        np.savez(args.output.with_suffix(".npz"), **arrays)
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()

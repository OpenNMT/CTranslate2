"""Bounded-memory Transformers reference for dense Gemma and Phi models.

Load one decoder layer at a time from pinned safetensors. Embeddings read only
requested rows, and the vocabulary projection reads output-row tiles. All model
operations remain the independent Transformers implementation; no CT2 kernels,
converter code, or packed CT2 weights are used.
"""

import contextlib
import json

from pathlib import Path

import torch
import torch.nn.functional as F

from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache


class StreamingDenseReference:
    def __init__(
        self,
        snapshot,
        weight_reference="source",
        attention="eager",
        compute_dtype="float32",
    ):
        if weight_reference not in ("source", "float16", "int4_float16"):
            raise ValueError("Unsupported dense reference weight format")
        self.snapshot = Path(snapshot)
        self.weight_reference = weight_reference
        if compute_dtype not in ("float32", "float16"):
            raise ValueError("Unsupported reference compute dtype")
        self.compute_dtype = getattr(torch, compute_dtype)
        config = AutoConfig.from_pretrained(self.snapshot, local_files_only=True)
        if config.model_type not in ("gemma2", "gemma3_text", "phi3"):
            raise ValueError(
                "Dense streaming reference supports Gemma2, Gemma3 text, and Phi3"
            )
        if config.model_type == "gemma2" and attention != "eager":
            raise ValueError(
                "Gemma2 reference requires eager attention for softcapping"
            )
        self.load_dtype = getattr(config, "dtype", None) or getattr(
            config, "torch_dtype", torch.float32
        )
        if isinstance(self.load_dtype, str):
            self.load_dtype = getattr(torch, self.load_dtype)
        config.dtype = self.compute_dtype
        config._attn_implementation = attention
        self.config = config
        self._stack = contextlib.ExitStack()
        index = self.snapshot / "model.safetensors.index.json"
        if index.exists():
            self.tensor_files = json.loads(index.read_text())["weight_map"]
            files = sorted(set(self.tensor_files.values()))
        else:
            files = ["model.safetensors"]
            self.tensor_files = None
        self.readers = {
            name: self._stack.enter_context(
                safe_open(self.snapshot / name, framework="pt", device="cpu")
            )
            for name in files
        }
        if self.tensor_files is None:
            self.tensor_files = {key: files[0] for key in self.readers[files[0]].keys()}
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(config).eval()
        core = self.model.model
        for name, module in list(core.named_children()):
            if "rotary_emb" in name:
                with torch.device("cpu"):
                    setattr(core, name, type(module)(config=module.config))
        embedding = core.embed_tokens
        if hasattr(embedding, "embed_scale"):
            embedding.embed_scale = torch.tensor(config.hidden_size**0.5)
        self.embedding_rows = {}
        # Row remapping changes the vocabulary range. padding_idx only controls
        # gradients; inference must still read the checkpoint's actual row.
        embedding.padding_idx = None
        embedding.register_forward_pre_hook(self._load_embedding)
        embedding.register_forward_hook(self._release_embedding)
        for index, layer in enumerate(core.layers):
            self._stream_module(layer, f"model.layers.{index}.")
        self._stream_module(core.norm, "model.norm.")
        self.model.lm_head = _StreamingHead(self)

    def _key(self, name):
        if name not in self.tensor_files and name == "lm_head.weight":
            if not self.config.tie_word_embeddings:
                raise KeyError("Untied source checkpoint has no language-model head")
            name = "model.embed_tokens.weight"
        if name not in self.tensor_files:
            raise KeyError(f"Source checkpoint is missing {name}")
        return name

    def _slice(self, name):
        name = self._key(name)
        return self.readers[self.tensor_files[name]].get_slice(name)

    def _weight(self, name):
        name = self._key(name)
        value = self.readers[self.tensor_files[name]].get_tensor(name)
        if self.weight_reference == "source":
            return value.float()
        if self.weight_reference == "float16":
            return value.half().float()
        value = value.to(self.load_dtype).float()
        if value.ndim == 2 and not name.startswith("lm_head."):
            # Independent group-64 symmetric quantization, including the stored
            # FP16 dequantized-weight rounding used by the reference contract.
            rows, depth = value.shape
            tail = (-depth) % 64
            groups = F.pad(value, (0, tail)) if tail else value
            groups = groups.reshape(rows, -1, 64)
            scale = groups.abs().amax(-1, keepdim=True) / 7
            scale[scale == 0] = 1
            scale.clamp_(min=torch.finfo(torch.float32).tiny)
            quantized = (groups / scale).round().clamp(-7, 7)
            return (quantized * scale).half().float().reshape(rows, -1)[:, :depth]
        return value.half().float()

    def _stream_module(self, module, prefix):
        parameters = list(module.named_parameters())

        def load(target, args):
            state = {
                name: self._weight(prefix + name).to(self.compute_dtype)
                for name, _ in parameters
            }
            target.load_state_dict(state, assign=True, strict=True)

        def release(target, args, output):
            target.to("meta")

        module.register_forward_pre_hook(load)
        module.register_forward_hook(release)

    def _load_embedding(self, module, args):
        ids = args[0]
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        source = self._slice("model.embed_tokens.weight")
        for token in unique.tolist():
            if token not in self.embedding_rows:
                value = source[token : token + 1]
                if self.weight_reference == "int4_float16":
                    value = value.to(self.load_dtype).half()
                elif self.weight_reference == "float16":
                    value = value.half()
                self.embedding_rows[token] = value.to(self.compute_dtype)
        rows = torch.cat([self.embedding_rows[token] for token in unique.tolist()])
        module.weight = torch.nn.Parameter(rows, requires_grad=False)
        return (inverse.reshape_as(ids), *args[1:])

    def _release_embedding(self, module, args, output):
        module.weight = torch.nn.Parameter(
            torch.empty(self.config.vocab_size, self.config.hidden_size, device="meta"),
            requires_grad=False,
        )

    def make_cache(self):
        return DynamicCache(config=self.config)

    @torch.inference_mode()
    def forward(self, ids, cache=None, last_logits=1):
        if not ids:
            raise ValueError("Reference input must not be empty")
        result = self.model(
            input_ids=torch.tensor([ids]),
            past_key_values=cache,
            use_cache=cache is not None,
            logits_to_keep=last_logits,
        )
        return result.logits[0].numpy()

    def close(self):
        self._stack.close()


class _StreamingHead(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        # Bypass Module registration: the reference owns its model, which owns
        # this head. Registering the model again would create a module cycle.
        object.__setattr__(self, "reference", reference)

    def forward(self, hidden):
        reference = self.reference
        source = reference._slice("lm_head.weight")
        pieces = []
        for start in range(0, reference.config.vocab_size, 4096):
            value = source[start : start + 4096]
            if reference.weight_reference == "int4_float16":
                value = value.to(reference.load_dtype).half()
            elif reference.weight_reference == "float16":
                value = value.half()
            # CT2 keeps its vocabulary projection in FP32 even when decoder
            # activations are FP16. Keep that contract explicit in this oracle.
            pieces.append(
                F.linear(hidden.float(), value.to(reference.compute_dtype).float())
            )
        return torch.cat(pieces, dim=-1)


def main():
    import argparse

    import numpy as np
    import transformers

    from benchmark import make_prompts
    from validate import compare

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--weight-reference",
        choices=["source", "float16", "int4_float16"],
        default="source",
    )
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument(
        "--compute-dtype", choices=["float32", "float16"], default="float32"
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[33, 513])
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache-atol", type=float, default=0.002)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output_tokens < 1:
        parser.error("output-tokens must be positive")
    torch.set_num_threads(args.threads)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.snapshot, local_files_only=True
    )
    prompts = make_prompts(tokenizer, args.lengths)
    reference = StreamingDenseReference(
        args.snapshot, args.weight_reference, args.attention, args.compute_dtype
    )
    arrays = {}
    report = {
        "backend": "streaming_transformers",
        "compute_dtype": args.compute_dtype,
        "projection_dtype": "float32",
        "snapshot": str(args.snapshot.resolve()),
        "weight_reference": args.weight_reference,
        "attention": args.attention,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cases": [],
    }
    try:
        for index, ids in enumerate(prompts):
            cache = reference.make_cache()
            logits = reference.forward(ids, cache=cache)
            generated, cached = [], []
            for step in range(args.output_tokens):
                if not np.isfinite(logits).all():
                    raise AssertionError("Nonfinite independent reference logits")
                cached.append(logits[-1].copy())
                token = int(logits[-1].argmax())
                generated.append(token)
                if step + 1 < args.output_tokens:
                    logits = reference.forward([token], cache=cache)
            del cache
            cached = np.stack(cached)
            full = reference.forward(
                ids + generated[:-1], last_logits=args.output_tokens
            )
            metrics = compare(full, cached)
            arrays.update(
                {
                    f"ids_{index}": np.asarray(ids, dtype=np.int32),
                    f"last_logits_{index}": cached[:1],
                    f"cached_{index}": cached,
                    f"generated_{index}": np.asarray(generated, dtype=np.int32),
                }
            )
            record = {
                "length": len(ids),
                "generated_ids": generated,
                "cached_vs_full": metrics,
            }
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if metrics["max_abs"] > args.cache_atol:
                raise AssertionError("Independent cached/full reference disagreement")
    finally:
        reference.close()


if __name__ == "__main__":
    main()

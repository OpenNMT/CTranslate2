"""Local, deterministic complete decoder tests; no model downloads."""

import copy

from pathlib import Path

import numpy as np
import pytest
import test_utils

import ctranslate2


def _host(storage):
    return np.array(storage.to_device(ctranslate2.Device.cpu), copy=True).astype(
        np.float32
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "qwen3_moe",
        "qwen3_moe_top1",
        "qwen3_moe_all",
        "qwen3_moe_mixed",
        "mixtral",
        "mixtral_window",
        "qwen3_moe_window",
        "qwen3_moe_mqa",
        "qwen3_moe_mqa_window",
        "qwen2_moe",
        "qwen2_moe_normalized",
        "qwen3",
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        pytest.param("mps", marks=test_utils.require_mps),
        pytest.param("mps_flash", marks=test_utils.require_mps),
        "cpu_int4",
        pytest.param("mps_int4", marks=test_utils.require_mps),
        pytest.param("mps_int4_flash", marks=test_utils.require_mps),
    ],
)
def test_tiny_decoder_logits_greedy_and_cache(tmp_path, architecture, backend):
    _check_tiny_decoder(tmp_path, architecture, backend)


def _check_tiny_decoder(tmp_path, architecture, backend, matched_precision=False):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="4.51.3")
    from ctranslate2.converters.transformers import (
        MixtralLoader,
        Qwen2MoeLoader,
        Qwen3Loader,
        Qwen3MoeLoader,
    )

    torch.set_num_threads(2)
    torch.manual_seed(42)
    options = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1 if "mqa" in architecture else 2,
        head_dim=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    if architecture.startswith("qwen3_moe"):
        config = transformers.Qwen3MoeConfig(
            **options,
            moe_intermediate_size=48,
            num_experts=4,
            num_experts_per_tok=(
                1
                if architecture.endswith("top1")
                else 4
                if architecture.endswith("all")
                else 2
            ),
            norm_topk_prob=True,
            use_sliding_window=architecture.endswith("window"),
            sliding_window=8,
            mlp_only_layers=[1] if architecture.endswith("mixed") else [],
        )
        model_class, loader = transformers.Qwen3MoeForCausalLM, Qwen3MoeLoader()
    elif architecture.startswith("mixtral"):
        config = transformers.MixtralConfig(
            **options,
            num_local_experts=4,
            num_experts_per_tok=2,
            sliding_window=8 if architecture.endswith("window") else None,
        )
        model_class, loader = transformers.MixtralForCausalLM, MixtralLoader()
    elif architecture.startswith("qwen2_moe"):
        config = transformers.Qwen2MoeConfig(
            **options,
            moe_intermediate_size=48,
            shared_expert_intermediate_size=40,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=architecture.endswith("normalized"),
            use_sliding_window=False,
        )
        model_class, loader = transformers.Qwen2MoeForCausalLM, Qwen2MoeLoader()
    else:
        config = transformers.Qwen3Config(**options)
        model_class, loader = transformers.Qwen3ForCausalLM, Qwen3Loader()
    config._attn_implementation = "eager"
    source = model_class(config).eval()
    # The normalized Qwen2 fixture from Transformers 5.x has a discontinuous
    # TopK cutoff: random router rows select different experts after FP16
    # activation rounding, even with a correct FP32 router projection.
    # Signed, power-of-two multiples preserve two token-dependent expert pairs
    # and predictable binary scaling. Near-cutoff routing has a separate native
    # regression that compares identical representable inputs, not FP16 vs FP32
    # hidden states. Do not change the seed or relax the logit/token assertions.
    with torch.no_grad():
        for name, module in source.named_modules():
            if (
                architecture == "qwen2_moe_normalized"
                and not matched_precision
                and name.endswith(".gate")
                and tuple(module.weight.shape) == (4, 32)
            ):
                direction = module.weight[0].clone()
                for index, scale in enumerate((1.0, -1.0, 0.125, -0.125)):
                    module.weight[index].copy_(direction * scale)
    # Independent torch reference for the saved groupwise format. Packing is
    # tested separately; the decoder reference consumes actual quantized values.
    reference_source = copy.deepcopy(source)
    if "int4" in backend:
        with torch.no_grad():
            for name, value in reference_source.named_parameters():
                if (
                    value.ndim >= 2
                    and "embed_tokens" not in name
                    and "lm_head" not in name
                    and not name.endswith(".gate.weight")
                    and "shared_expert_gate" not in name
                ):
                    flat = value.reshape(-1, value.shape[-1])
                    for begin in range(0, flat.shape[1], 64):
                        block = flat[:, begin : begin + 64]
                        scale = block.abs().amax(-1, keepdim=True) / 7
                        scale[scale == 0] = 1
                        block.copy_((block / scale).round().clamp(-7, 7) * scale)
                    value.copy_(value.half().float())
                else:
                    # The INT4 file stores embeddings, router, head, norms and
                    # shared scalar gates in FP16 too. Router rounding can
                    # change expert selection at ties in newer model seeds.
                    value.copy_(value.half().float())
    if matched_precision:
        # Preserve the original near-tied random router. Match MPS's explicit
        # precision boundaries instead of expecting FP32 and FP16 hidden states
        # to select the same experts. No routes or logits are forced/replayed.
        reference_source.half()
        for block in reference_source.model.layers:
            block.mlp.gate.float()
            block.mlp.gate.register_forward_pre_hook(
                lambda module, args: (args[0].float(),)
            )
        reference_source.lm_head.weight = torch.nn.Parameter(
            reference_source.lm_head.weight.detach().float(), requires_grad=False
        )
        reference_source.lm_head.register_forward_pre_hook(
            lambda module, args: (args[0].float(),)
        )
    references = []
    with torch.inference_mode():
        for length in (7, 33, 65) if architecture.endswith("window") else (7, 33):
            prompt = [3 + (i * 7) % 60 for i in range(length)]
            full = (
                reference_source(torch.tensor([prompt]), use_cache=False)
                .logits[0]
                .numpy()
                .copy()
            )
            sequence, generated, cached = list(prompt), [], []
            for _ in range(4):
                logits = reference_source(
                    torch.tensor([sequence]), use_cache=False
                ).logits[0, -1]
                cached.append(logits.numpy().copy())
                token = int(logits.argmax())
                generated.append(token)
                sequence.append(token)
            references.append((prompt, full, np.stack(cached), generated))
    if architecture == "qwen3_moe" and backend == "cpu":
        import sys

        reference_dir = tmp_path / "source"
        source.save_pretrained(reference_dir)
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[2] / "tools/benchmark/llm")
        )
        try:
            from reference_moe import StreamingQwenMoe

            reference = StreamingQwenMoe(reference_dir)
            with torch.inference_mode():
                for prompt, full, _, _ in references:
                    np.testing.assert_allclose(
                        reference.forward(prompt), full, atol=2e-5, rtol=0
                    )
        finally:
            sys.path.pop(0)
    spec = loader.get_model_spec(copy.deepcopy(source))
    spec.register_vocabulary([f"t{i}" for i in range(64)])
    spec.config.bos_token = "t0"
    spec.config.eos_token = "t1"
    spec.config.unk_token = "t2"
    spec.config.layer_norm_epsilon = config.rms_norm_eps
    spec.validate()
    cpu = backend.startswith("cpu")
    dtype = "float32" if cpu else "float16"
    spec.optimize(quantization="int4_float16" if "int4" in backend else dtype)
    spec.save(tmp_path)
    generator = ctranslate2.Generator(
        str(tmp_path),
        device="cpu" if cpu else "mps",
        compute_type=dtype,
        flash_attention=backend.endswith("flash"),
        inter_threads=1,
        intra_threads=2,
    )
    tolerance = 2e-5 if cpu and "int4" not in backend else 0.003
    for prompt, full, cached, tokens in references:
        actual = _host(generator.forward_batch([prompt]))[0]
        np.testing.assert_allclose(actual, full, atol=tolerance, rtol=0)
        log_probs = actual - np.logaddexp.reduce(actual, axis=-1, keepdims=True)
        scored = generator.score_batch([[f"t{i}" for i in prompt]], max_input_length=0)[
            0
        ]
        np.testing.assert_allclose(
            scored.log_probs,
            log_probs[:-1, prompt[1:]].diagonal(),
            atol=0.002,
            rtol=0,
        )
        for _ in range(2):
            result = generator.generate_batch(
                [[f"t{i}" for i in prompt]],
                include_prompt_in_result=False,
                beam_size=1,
                sampling_topk=1,
                max_length=4,
                min_length=4,
                end_token=[],
                return_logits_vocab=True,
            )[0]
            assert result.sequences_ids[0] == tokens
            # Includes the result-copy handoff after final hypothesis sorting.
            # Small GPU copies exposed stale/zero snapshots before that fix.
            actual_cached = np.stack(
                [_host(step).reshape(-1) for step in result.logits[0]]
            )
            np.testing.assert_allclose(actual_cached, cached, atol=tolerance, rtol=0)

    # Different prompt lengths and multiple hypotheses exercise the sparse FFN
    # after cache expansion/reordering, including shrinking active beam batches.
    prompts = [[f"t{i}" for i in prompt] for prompt, *_ in references]
    options = dict(
        include_prompt_in_result=False,
        beam_size=4,
        num_hypotheses=4,
        max_length=4,
        min_length=4,
        end_token=[],
    )
    batched = generator.generate_batch(prompts, **options)
    for prompt, result in zip(prompts, batched):
        single = generator.generate_batch([prompt], **options)[0]
        assert result.sequences_ids == single.sequences_ids

    if architecture == "qwen3_moe" and backend == "mps_flash":
        import os
        import subprocess
        import sys

        # The backend random counter is process-wide. A seed supplied before
        # device initialization must reproduce sampling in fresh processes.
        code = """
import sys, numpy as np, ctranslate2
ctranslate2.set_random_seed(42)
model = ctranslate2.Generator(sys.argv[1], device='mps', compute_type='float16',
                             flash_attention=True, inter_threads=1)
result = model.generate_batch([['t3', 't10', 't17', 't24', 't31', 't38', 't45']],
    include_prompt_in_result=False, beam_size=1, max_length=8, min_length=8,
    end_token=[], sampling_temperature=.6, sampling_topk=10, sampling_topp=.7,
    return_logits_vocab=True)[0]
logits = np.stack([np.array(x.to_device(ctranslate2.Device.cpu)).reshape(-1)
                   for x in result.logits[0]])
np.savez(sys.argv[2], ids=result.sequences_ids[0], logits=logits)
"""
        paths = [tmp_path / f"sampling-{i}.npz" for i in range(2)]
        for path in paths:
            child = subprocess.run(
                [sys.executable, "-c", code, str(tmp_path), str(path)],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert child.returncode == 0, child.stdout + child.stderr
        with np.load(paths[0]) as first, np.load(paths[1]) as second:
            np.testing.assert_array_equal(first["ids"], second["ids"])
            np.testing.assert_array_equal(first["logits"], second["logits"])
            for token, raw in zip(first["ids"], first["logits"]):
                order = np.lexsort((np.arange(raw.size), -raw.astype(np.float32)))[:10]
                scaled = (raw[order] * np.asarray(1 / 0.6, dtype=raw.dtype)).astype(
                    np.float32
                )
                probabilities = np.exp(scaled - scaled.max())
                probabilities /= probabilities.sum()
                allowed = order[(np.cumsum(probabilities) - probabilities) < 0.7]
                assert token in allowed


def test_moe_spec_rejects_invalid_or_unsupported_routing():
    from ctranslate2.specs.transformer_spec import MoEFeedForwardSpec

    for experts, topk in ((0, 1), (4, 0), (4, 5)):
        with pytest.raises(ValueError):
            MoEFeedForwardSpec(experts, topk)
    assert MoEFeedForwardSpec(4, 2, normalize_topk=False).normalize_topk == 0


@test_utils.require_mps
@pytest.mark.parametrize("backend", ["mps_int4", "mps_int4_flash"])
def test_original_qwen2_near_cutoff_matches_fp16_reference(tmp_path, backend):
    _check_tiny_decoder(
        tmp_path, "qwen2_moe_normalized", backend, matched_precision=True
    )


def test_moe_loaders_reject_unsupported_attention_schedules():
    from types import SimpleNamespace

    from ctranslate2.converters.transformers import _set_decoder_attention_windows
    from ctranslate2.specs.transformer_spec import TransformerDecoderModelSpec

    for kinds, window in (
        (["linear_attention"], 8),
        (["full_attention"] * 2, 8),
        (["sliding_attention"], 0),
    ):
        spec = TransformerDecoderModelSpec.from_config(1, 4)
        with pytest.raises(ValueError):
            _set_decoder_attention_windows(
                spec, SimpleNamespace(layer_types=kinds, sliding_window=window)
            )

"""Complete tiny-model KV format checks against independent source caches."""

import copy
import sys

from pathlib import Path

import numpy as np
import pytest
import test_utils

import ctranslate2


def host(storage):
    return np.array(storage.to_device(ctranslate2.Device.cpu), copy=True).astype(
        np.float32
    )


@test_utils.require_mps
@pytest.mark.parametrize("bits", [8, 5, 4])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("weight_int4", [False, True])
def test_quantized_cache_source_logits_full_cached_and_beams(
    tmp_path, bits, moe, weight_int4
):
    import torch
    import transformers

    from ctranslate2.converters.transformers import Qwen3Loader, Qwen3MoeLoader

    tools = Path(__file__).resolve().parents[2] / "tools/benchmark/llm"
    sys.path.insert(0, str(tools))
    try:
        from kv_reference import QuantizedReferenceCache
    finally:
        sys.path.pop(0)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    options = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    if moe:
        config = transformers.Qwen3MoeConfig(
            **options,
            moe_intermediate_size=48,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
        source = transformers.Qwen3MoeForCausalLM(config).eval()
        loader = Qwen3MoeLoader()
    else:
        config = transformers.Qwen3Config(**options)
        source = transformers.Qwen3ForCausalLM(config).eval()
        loader = Qwen3Loader()
    source.config._attn_implementation = "eager"
    spec = loader.get_model_spec(copy.deepcopy(source))
    spec.register_vocabulary([f"t{i}" for i in range(64)])
    spec.config.bos_token, spec.config.eos_token, spec.config.unk_token = (
        "t0",
        "t1",
        "t2",
    )
    spec.config.layer_norm_epsilon = config.rms_norm_eps
    spec.validate()
    spec.optimize(quantization="int4_float16" if weight_int4 else "float16")
    spec.save(tmp_path)
    if weight_int4:
        with torch.no_grad():
            for name, value in source.named_parameters():
                if (
                    value.ndim >= 2
                    and "embed_tokens" not in name
                    and "lm_head" not in name
                    and not name.endswith(".gate.weight")
                ):
                    rows = value.reshape(-1, value.shape[-1])
                    for begin in range(0, rows.shape[-1], 64):
                        block = rows[:, begin : begin + 64]
                        scale = block.abs().amax(-1, keepdim=True) / 7
                        scale[scale == 0] = 1
                        block.copy_((block / scale).round().clamp(-7, 7) * scale)
                    value.copy_(value.half().float())
    generator = ctranslate2.Generator(
        str(tmp_path),
        device="mps",
        compute_type="float16",
        flash_attention=True,
        kv_cache_type=f"int{bits}",
        inter_threads=1,
        intra_threads=2,
    )
    assert generator.kv_cache_type == f"int{bits}"
    with torch.inference_mode():
        for length in (7, 33, 65):
            ids = [(i * 7 + 3) % 63 for i in range(length)]
            inputs = torch.tensor([ids])
            expected = (
                source(
                    inputs,
                    use_cache=True,
                    past_key_values=QuantizedReferenceCache(config, bits),
                )
                .logits[0]
                .numpy()
            )
            full = host(generator.forward_batch([ids]))[0]
            np.testing.assert_allclose(full, expected, atol=0.015, rtol=0)
            result = generator.generate_batch(
                [[f"t{i}" for i in ids]],
                max_length=4,
                min_length=4,
                end_token=[],
                include_prompt_in_result=False,
                beam_size=1,
                sampling_topk=1,
                return_logits_vocab=True,
            )[0]
            cache = QuantizedReferenceCache(config, bits)
            logits = source(inputs, use_cache=True, past_key_values=cache).logits[:, -1]
            for index, token in enumerate(result.sequences_ids[0]):
                actual = host(result.logits[0][index]).reshape(-1)
                np.testing.assert_allclose(
                    actual, logits[0].numpy(), atol=0.015, rtol=0
                )
                assert token == int(logits.argmax(-1))
                logits = source(
                    torch.tensor([[token]]), use_cache=True, past_key_values=cache
                ).logits[:, -1]
            # A fresh full pass over each generated prefix checks that quantized
            # incremental append retains the same semantics as full prefill.
            for index in range(4):
                prefix = ids + result.sequences_ids[0][:index]
                repeated = host(generator.forward_batch([prefix]))[0, -1]
                np.testing.assert_allclose(
                    repeated,
                    host(result.logits[0][index]).reshape(-1),
                    atol=0.015,
                    rtol=0,
                )
        prompts = [["t3", "t7", "t5"] * 3, ["t9", "t6"] * 17]
        options = dict(
            beam_size=3,
            max_length=5,
            min_length=5,
            end_token=[],
            include_prompt_in_result=False,
            num_hypotheses=3,
        )
        batched = generator.generate_batch(prompts, **options)
        separate = [generator.generate_batch([p], **options)[0] for p in prompts]
        assert [x.sequences_ids for x in batched] == [x.sequences_ids for x in separate]
    if moe and not weight_int4:
        reference_dir = tmp_path / "source-reference"
        source.save_pretrained(reference_dir)
        sys.path.insert(0, str(tools))
        try:
            from reference_moe import StreamingQwenMoe

            reference = StreamingQwenMoe(reference_dir, kv_cache_bits=bits)
            with torch.inference_mode():
                expected = (
                    source(
                        torch.tensor([[3, 5, 7]]),
                        use_cache=True,
                        past_key_values=QuantizedReferenceCache(config, bits),
                    )
                    .logits[0]
                    .numpy()
                )
            np.testing.assert_allclose(
                reference.forward([3, 5, 7]), expected, atol=2e-5, rtol=0
            )
        finally:
            sys.path.pop(0)
    # Explicit FP16 preserves the default output, and fp16 is a supported alias.
    default = ctranslate2.Generator(
        str(tmp_path), device="mps", compute_type="float16", flash_attention=True
    )
    fp16 = ctranslate2.Generator(
        str(tmp_path),
        device="mps",
        compute_type="float16",
        flash_attention=True,
        kv_cache_type="fp16",
    )
    assert fp16.kv_cache_type == default.kv_cache_type == "float16"
    np.testing.assert_array_equal(
        host(default.forward_batch([[3, 5, 7]])), host(fp16.forward_batch([[3, 5, 7]]))
    )
    for options in (
        {"kv_cache_type": "int6"},
        {"kv_cache_type": "int4", "flash_attention": False},
    ):
        with pytest.raises(ValueError):
            ctranslate2.Generator(
                str(tmp_path), device="mps", compute_type="float16", **options
            )
    with pytest.raises(ValueError):
        ctranslate2.Generator(
            str(tmp_path), device="cpu", kv_cache_type="int4", flash_attention=True
        )

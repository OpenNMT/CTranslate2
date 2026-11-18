"""Download-free Gemma/Phi regressions against independent Transformers models."""

import copy
import sys

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_utils

import ctranslate2


def _host(value):
    value = value.to_device(ctranslate2.Device.cpu).to(ctranslate2.DataType.float32)
    return np.array(value, copy=True)


@pytest.fixture(scope="module", params=["gemma2", "gemma3", "phi3"])
def source_model(request, tmp_path_factory):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers", minversion="4.57")
    from ctranslate2.converters.transformers import (
        Gemma2Loader,
        Gemma3Loader,
        Phi3Loader,
    )

    torch.set_num_threads(2)
    torch.manual_seed(42)
    options = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    name = request.param
    if name == "gemma2":
        config = tf.Gemma2Config(
            **options,
            query_pre_attn_scalar=4,
            sliding_window=4,
            attn_logit_softcapping=5.0,
            final_logit_softcapping=3.0,
        )
        loader = Gemma2Loader()
    elif name == "gemma3":
        config = tf.Gemma3TextConfig(
            **options,
            query_pre_attn_scalar=4,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
        )
        loader = Gemma3Loader()
    else:
        config = tf.Phi3Config(
            **{**options, "pad_token_id": 63},
            # Exercise a fractional expansion ratio as well as factor switching.
            max_position_embeddings=60,
            original_max_position_embeddings=8,
            rope_scaling={
                "type": "longrope",
                "short_factor": [1.00017, 1.3333337, 3.14159265, 17.123456],
                "long_factor": [2.00013, 3.3333337, 5.14159265, 23.123456],
            },
        )
        loader = Phi3Loader()
    config._attn_implementation = "eager"
    source = tf.AutoModelForCausalLM.from_config(config).float().eval()
    with torch.no_grad():
        spec = loader.get_model_spec(copy.deepcopy(source))
    spec.register_vocabulary([f"t{i}" for i in range(64)])
    loader.set_config(
        spec.config,
        source,
        SimpleNamespace(bos_token="t0", eos_token="t1", unk_token="t2"),
    )
    assert spec.config.layer_norm_epsilon == config.rms_norm_eps
    spec.validate()
    paths = {}
    for quantization in ("float32", "float16", "int4_float16"):
        saved = copy.deepcopy(spec)
        saved.optimize(quantization=quantization)
        if name == "phi3":
            for layer in saved.decoder.layer:
                assert (
                    str(layer.self_attention.rotary_scaling_long_factor.dtype)
                    == "float32"
                )
                assert (
                    str(layer.self_attention.rotary_scaling_short_factor.dtype)
                    == "float32"
                )
        path = tmp_path_factory.mktemp(name + "_" + quantization)
        saved.save(path)
        paths[quantization] = path
    return name, source, paths


_MODES = [
    ("cpu", "float32", False),
    pytest.param("mps", "float32", False, marks=test_utils.require_mps),
    pytest.param("mps", "float16", False, marks=test_utils.require_mps),
    pytest.param("mps", "float16", True, marks=test_utils.require_mps),
]


@pytest.mark.parametrize("device,dtype,flash", _MODES)
def test_gemma_phi_full_prefill_and_reused_rotary_tables(
    source_model, device, dtype, flash
):
    import torch

    name, source, paths = source_model
    generator = ctranslate2.Generator(
        str(paths[dtype]),
        device=device,
        compute_type=dtype,
        flash_attention=flash,
        intra_threads=2,
    )
    tolerance = 0.002 if dtype == "float16" else 0.00001
    # Cross attention windows and both LongRoPE regimes, then return to a short
    # request after a large table has been allocated. Capacity must not select
    # which rotary factors apply to the new request.
    for length in (3, 7, 9, 33, 7, 3):
        ids = [3 + (i * 7) % 60 for i in range(length)]
        with torch.inference_mode():
            expected = source(torch.tensor([ids]), use_cache=False).logits[0].numpy()
        actual = _host(generator.forward_batch([ids]))[0]
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        if name == "gemma2":
            assert np.max(np.abs(actual)) <= 3.0


@pytest.mark.parametrize("device,dtype,flash", _MODES)
def test_gemma_phi_cached_greedy_logits(source_model, device, dtype, flash):
    import torch

    _, source, paths = source_model
    generator = ctranslate2.Generator(
        str(paths[dtype]),
        device=device,
        compute_type=dtype,
        flash_attention=flash,
        intra_threads=2,
    )
    tolerance = 0.002 if dtype == "float16" else 0.00001
    # Keep each Phi request within one regime: the source itself retains old
    # cached keys when its dynamic LongRoPE factors change mid-generation.
    for length in (3, 9):
        ids = [3 + (i * 7) % 60 for i in range(length)]
        for repeat in range(2):
            result = generator.generate_batch(
                [[f"t{i}" for i in ids]],
                beam_size=1,
                sampling_topk=1,
                max_length=3,
                min_length=3,
                end_token=[],
                include_prompt_in_result=False,
                return_logits_vocab=True,
            )[0]
            for index, token in enumerate(result.sequences_ids[0]):
                prefix = ids + result.sequences_ids[0][:index]
                with torch.inference_mode():
                    expected = (
                        source(torch.tensor([prefix]), use_cache=False)
                        .logits[0, -1]
                        .numpy()
                    )
                actual = _host(result.logits[0][index]).reshape(-1)
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=tolerance,
                    rtol=0,
                    err_msg=f"{device} {dtype} length={length} repeat={repeat} step={index}",
                )
                assert token == int(expected.argmax())
                full = _host(generator.forward_batch([prefix]))[0, -1]
                np.testing.assert_allclose(actual, full, atol=tolerance, rtol=0)


@test_utils.require_mps
@pytest.mark.parametrize("source_model", ["gemma2"], indirect=True)
def test_gemma2_softcap_rejects_unsupported_quantized_cache(source_model):
    name, _, paths = source_model
    assert name == "gemma2"
    with pytest.raises(ValueError, match="softcapping"):
        ctranslate2.Generator(
            str(paths["float16"]),
            device="mps",
            compute_type="float16",
            flash_attention=True,
            kv_cache_type="int4",
        )


@pytest.mark.parametrize("source_model", ["phi3"], indirect=True)
@pytest.mark.parametrize("device,dtype,flash", _MODES)
def test_phi3_static_prompt_changes_rotary_regime(source_model, device, dtype, flash):
    import torch

    _, source, paths = source_model
    generator = ctranslate2.Generator(
        str(paths[dtype]),
        device=device,
        compute_type=dtype,
        flash_attention=flash,
        intra_threads=2,
    )
    tolerance = 0.002 if dtype == "float16" else 0.00001
    static = [3, 10, 17, 24, 31, 38]
    for suffix in ([4, 11, 18], [4], [4, 11, 18]):
        result = generator.generate_batch(
            [[f"t{i}" for i in suffix]],
            static_prompt=[f"t{i}" for i in static],
            cache_static_prompt=True,
            max_length=2,
            min_length=2,
            end_token=[],
            include_prompt_in_result=False,
            return_logits_vocab=True,
        )[0]
        for step, value in enumerate(result.logits[0]):
            prefix = static + list(suffix) + result.sequences_ids[0][:step]
            with torch.inference_mode():
                expected = (
                    source(torch.tensor([prefix]), use_cache=False)
                    .logits[0, -1]
                    .numpy()
                )
            np.testing.assert_allclose(
                _host(value).reshape(-1), expected, atol=tolerance, rtol=0
            )


@pytest.mark.parametrize("source_model", ["phi3"], indirect=True)
@pytest.mark.parametrize("device,dtype,flash", _MODES)
def test_phi3_mixed_length_batch_uses_batch_rotary_regime(
    source_model, device, dtype, flash
):
    import torch

    _, source, paths = source_model
    generator = ctranslate2.Generator(
        str(paths[dtype]),
        device=device,
        compute_type=dtype,
        flash_attention=flash,
        intra_threads=2,
    )
    ids = [[3, 10, 17], [3 + (i * 7) % 60 for i in range(9)]]
    results = generator.generate_batch(
        [[f"t{i}" for i in row] for row in ids],
        max_length=2,
        min_length=2,
        end_token=[],
        include_prompt_in_result=False,
        return_logits_vocab=True,
    )
    tolerance = 0.002 if dtype == "float16" else 0.00001
    for step in range(2):
        rows = [
            row + result.sequences_ids[0][:step] for row, result in zip(ids, results)
        ]
        width = max(map(len, rows))
        inputs = torch.tensor([[63] * (width - len(row)) + row for row in rows])
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows])
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            expected = (
                source(
                    inputs,
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=False,
                )
                .logits[:, -1]
                .numpy()
            )
        for index, result in enumerate(results):
            # Unequal prompts retain forced-prefix logit rows in this API.
            # Align the final rows with the generated tokens being compared.
            logits = result.logits[0][-len(result.sequences_ids[0]) :]
            np.testing.assert_allclose(
                _host(logits[step]).reshape(-1),
                expected[index],
                atol=tolerance,
                rtol=0,
            )


@pytest.mark.parametrize("weight_reference", ["source", "float16"])
@pytest.mark.parametrize("compute_dtype", ["float32", "float16"])
def test_streaming_dense_reference_matches_full_source(
    source_model, tmp_path, weight_reference, compute_dtype
):
    import torch

    tools = Path(__file__).resolve().parents[2] / "tools/benchmark/llm"
    sys.path.insert(0, str(tools))
    try:
        from reference_dense import StreamingDenseReference
    finally:
        sys.path.pop(0)
    _, source, _ = source_model
    source.save_pretrained(tmp_path)
    expected_model = copy.deepcopy(source)
    if weight_reference == "float16":
        with torch.no_grad():
            for value in expected_model.parameters():
                value.copy_(value.half().float())
    if compute_dtype == "float16":
        with torch.no_grad():
            for value in expected_model.parameters():
                value.data = value.data.half()
        # Keep rotary frequency buffers in FP32, as from_pretrained does.
        # A tied head must be detached before changing its dtype, so the
        # embedding remains FP16 like the rest of the decoder.
        head = copy.deepcopy(expected_model.lm_head).float()
        head.register_forward_pre_hook(lambda module, args: (args[0].float(),))
        expected_model.lm_head = head
    reference = StreamingDenseReference(
        tmp_path, weight_reference, compute_dtype=compute_dtype
    )
    try:
        ids = [3 + (i * 7) % 60 for i in range(9)]
        cache = reference.make_cache()
        actual = reference.forward(ids, cache=cache, last_logits=2)
        with torch.inference_mode():
            expected = (
                expected_model(torch.tensor([ids]), use_cache=False)
                .logits[0, -2:]
                .numpy()
            )
        np.testing.assert_allclose(actual, expected, atol=0.000001, rtol=0)
        token = int(actual[-1].argmax())
        actual = reference.forward([token], cache=cache)
        with torch.inference_mode():
            expected = (
                expected_model(torch.tensor([ids + [token]]), use_cache=False)
                .logits[0, -1:]
                .numpy()
            )
        np.testing.assert_allclose(actual, expected, atol=0.000001, rtol=0)
    finally:
        reference.close()


@test_utils.require_mps
@pytest.mark.parametrize("flash", [False, True])
def test_gemma_phi_int4_logits_and_cache(source_model, tmp_path, flash):
    tools = Path(__file__).resolve().parents[2] / "tools/benchmark/llm"
    sys.path.insert(0, str(tools))
    try:
        from reference_dense import StreamingDenseReference
    finally:
        sys.path.pop(0)
    _, source, paths = source_model
    source.save_pretrained(tmp_path)
    reference = StreamingDenseReference(tmp_path, "int4_float16")
    generator = ctranslate2.Generator(
        str(paths["int4_float16"]),
        device="mps",
        compute_type="float16",
        flash_attention=flash,
        kv_cache_type="float16" if flash else "auto",
        intra_threads=2,
    )
    try:
        for length in (3, 9):
            ids = [3 + (i * 7) % 60 for i in range(length)]
            result = generator.generate_batch(
                [[f"t{i}" for i in ids]],
                max_length=3,
                min_length=3,
                end_token=[],
                beam_size=1,
                include_prompt_in_result=False,
                return_logits_vocab=True,
            )[0]
            cache = reference.make_cache()
            expected = reference.forward(ids, cache=cache)
            for step, value in enumerate(result.logits[0]):
                actual = _host(value).reshape(1, -1)
                np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0)
                if step + 1 < len(result.logits[0]):
                    expected = reference.forward(
                        [result.sequences_ids[0][step]], cache=cache
                    )
            full = reference.forward(ids + result.sequences_ids[0][:-1], last_logits=3)
            cached = np.stack([_host(value).reshape(-1) for value in result.logits[0]])
            np.testing.assert_allclose(cached, full, atol=0.005, rtol=0)
    finally:
        reference.close()


@test_utils.require_mps
@pytest.mark.parametrize("source_model", ["gemma2"], indirect=True)
@pytest.mark.parametrize("quantization", ["float16", "int4_float16"])
@pytest.mark.parametrize("cache_type", ["auto", "float16"])
def test_gemma2_large_residuals_and_half_cache(
    source_model, tmp_path, quantization, cache_type
):
    import torch

    from ctranslate2.converters.transformers import Gemma2Loader

    source = copy.deepcopy(source_model[1]).half().float()
    with torch.no_grad():
        for layer in source.model.layers:
            layer.post_attention_layernorm.weight.fill_(-1)
            layer.post_feedforward_layernorm.weight.fill_(60000)
            layer.mlp.gate_proj.weight.fill_(0.02)
            layer.mlp.up_proj.weight.fill_(0.02)
            layer.mlp.down_proj.weight.fill_(0.02)
    maxima = []
    handle = source.model.norm.register_forward_pre_hook(
        lambda module, args: maxima.append(float(args[0].abs().max()))
    )
    ids = [3, 10, 17, 24, 31, 38, 45]
    with torch.inference_mode():
        expected = source(torch.tensor([ids]), use_cache=False).logits[0].numpy()
    handle.remove()
    assert maxima[0] > 65504
    spec = Gemma2Loader().get_model_spec(copy.deepcopy(source))
    spec.register_vocabulary([f"t{i}" for i in range(64)])
    spec.config.layer_norm_epsilon = source.config.rms_norm_eps
    spec.validate()
    spec.optimize(quantization=quantization)
    spec.save(tmp_path)
    generator = ctranslate2.Generator(
        str(tmp_path),
        device="mps",
        compute_type="float16",
        kv_cache_type=cache_type,
        flash_attention=True,
        intra_threads=2,
    )
    for _ in range(2):
        actual = _host(generator.forward_batch([ids]))[0]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0)
        result = generator.generate_batch(
            [[f"t{i}" for i in ids]],
            beam_size=1,
            sampling_topk=1,
            max_length=4,
            min_length=4,
            end_token=[],
            include_prompt_in_result=False,
            return_logits_vocab=True,
        )[0]
        for step, token in enumerate(result.sequences_ids[0]):
            prefix = ids + result.sequences_ids[0][:step]
            with torch.inference_mode():
                reference = (
                    source(torch.tensor([prefix]), use_cache=False)
                    .logits[0, -1]
                    .numpy()
                )
            logits = _host(result.logits[0][step]).reshape(-1)
            assert np.isfinite(logits).all()
            np.testing.assert_allclose(logits, reference, atol=0.002, rtol=0)
            assert token == int(reference.argmax())
            full = _host(generator.forward_batch([prefix]))[0, -1]
            np.testing.assert_allclose(logits, full, atol=0.002, rtol=0)

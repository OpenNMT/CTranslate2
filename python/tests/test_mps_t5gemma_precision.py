"""T5Gemma2 MPS FP16 precision regressions without converter or mask fixes."""

import copy

import numpy as np
import pytest
import test_utils

import ctranslate2


def _host(value):
    value = value.to_device(ctranslate2.Device.cpu).to(ctranslate2.DataType.float32)
    return np.array(value, copy=True)


@pytest.fixture(
    scope="module",
    params=[(1, False), (4, False), (1, True)],
    ids=["mqa", "mha", "mqa_window"],
)
def merged_model(request, tmp_path_factory):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers", minversion="5.9")
    from ctranslate2.converters.transformers import T5Gemma2Loader

    torch.set_num_threads(2)
    torch.manual_seed(42)
    kv_heads, window = request.param
    side = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=8,
        layer_types=["full_attention"] * 2,
        sliding_window=16,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        query_pre_attn_scalar=8,
    )
    config = tf.T5Gemma2Config(
        encoder=dict(
            text_config={
                **side,
                "sliding_window": 4 if window else 16,
                "layer_types": ["sliding_attention", "full_attention"]
                if window
                else side["layer_types"],
            },
            vision_config=dict(
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
            ),
        ),
        decoder={
            **side,
            "sliding_window": 4 if window else 16,
            "layer_types": ["sliding_attention", "full_attention"]
            if window
            else side["layer_types"],
        },
        image_token_index=63,
        eoi_token_index=62,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    config._attn_implementation = "eager"
    source = tf.T5Gemma2ForConditionalGeneration(config).eval()
    # Use a distinct output projection to detect accidental embedding fallback,
    # and avoid near-uniform logits whose beam cutoff ties at FP16 ULPs.
    with torch.no_grad():
        head = getattr(source.lm_head, "out_proj", source.lm_head)
        head.weight = torch.nn.Parameter(head.weight.detach().clone() * 2)
    # A plain head makes this precision fixture independent of the converter
    # follow-up that unwraps output projections.
    source.lm_head = head
    spec = T5Gemma2Loader().get_model_spec(copy.deepcopy(source))
    vocab = [f"t{i}" for i in range(64)]
    spec.register_source_vocabulary(vocab)
    spec.register_target_vocabulary(vocab)
    spec.config.bos_token = "t2"
    spec.config.eos_token = "t1"
    spec.config.unk_token = "t3"
    spec.config.decoder_start_token = "t2"
    spec.config.layer_norm_epsilon = 1e-6
    spec.validate()
    spec.optimize(quantization="float32")
    path = tmp_path_factory.mktemp("merged_attention")
    spec.save(path)
    return source, path


@pytest.mark.parametrize("merged_model", [(1, False)], indirect=True)
def test_half_conversion_preserves_float32_norm_scale(merged_model):
    import torch

    from ctranslate2.converters.transformers import T5Gemma2Loader

    source = copy.deepcopy(merged_model[0])
    with torch.no_grad():
        source.model.encoder.text_model.layers[
            0
        ].post_feedforward_layernorm.weight.fill_(4096)
    spec = T5Gemma2Loader().get_model_spec(source)
    vocab = [f"t{i}" for i in range(64)]
    spec.register_source_vocabulary(vocab)
    spec.register_target_vocabulary(vocab)
    spec.validate()
    spec.optimize(quantization="float16")
    gamma = spec.encoder.layer[0].post_feedforward_layer_norm.gamma
    assert gamma.dtype == "float32"
    np.testing.assert_array_equal(gamma.numpy(), np.full(32, 4097, dtype=np.float32))
    assert spec.encoder.layer[0].self_attention.linear[0].weight.dtype == "float16"


@test_utils.require_mps
@pytest.mark.parametrize("saved_dtype", ["float32", "float16"])
def test_t5gemma2_float16_large_residuals(merged_model, tmp_path, saved_dtype):
    import torch

    from ctranslate2.converters.transformers import T5Gemma2Loader

    source = copy.deepcopy(merged_model[0])
    # Two aligned finite branches exceed FP16's range in the residual sum,
    # while normalized attention/MLP inputs remain small and representable.
    with torch.no_grad():
        for side in (source.model.encoder.text_model, source.model.decoder):
            for layer in side.layers:
                layer.post_self_attn_layernorm.weight.fill_(-1)
                layer.post_feedforward_layernorm.weight.fill_(60000)
                layer.mlp.gate_proj.weight.fill_(0.02)
                layer.mlp.up_proj.weight.fill_(0.02)
                layer.mlp.down_proj.weight.fill_(0.02)
    maximum = []
    handle = source.model.encoder.text_model.norm.register_forward_pre_hook(
        lambda module, args: maximum.append(float(args[0].abs().max()))
    )
    ids = [4, 5, 6, 7, 8]
    with torch.inference_mode():
        expected = (
            source(
                input_ids=torch.tensor([ids]),
                decoder_input_ids=torch.tensor([[2, 10, 11]]),
            )
            .logits[0]
            .numpy()
        )
    handle.remove()
    assert maximum[0] > 65504
    assert np.isfinite(expected).all()

    spec = T5Gemma2Loader().get_model_spec(copy.deepcopy(source))
    vocab = [f"t{i}" for i in range(64)]
    spec.register_source_vocabulary(vocab)
    spec.register_target_vocabulary(vocab)
    spec.config.bos_token = "t2"
    spec.config.eos_token = "t1"
    spec.config.unk_token = "t3"
    spec.config.decoder_start_token = "t2"
    spec.config.layer_norm_epsilon = 1e-6
    spec.validate()
    spec.optimize(quantization=saved_dtype)
    spec.save(tmp_path)
    translator = ctranslate2.Translator(
        str(tmp_path), device="mps", compute_type="float16", intra_threads=2
    )
    for _ in range(2):
        result = translator.translate_batch(
            [[f"t{i}" for i in ids]],
            target_prefix=[["t10", "t11"]],
            beam_size=1,
            max_decoding_length=3,
            min_decoding_length=3,
            end_token=[],
            return_logits_vocab=True,
        )[0]
        actual = np.stack([_host(x).reshape(-1) for x in result.logits[0]])
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=0.003, rtol=0)

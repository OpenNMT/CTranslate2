"""CPU-only converter regressions; no checkpoint downloads are needed."""

import copy

import pytest


@pytest.mark.parametrize("loader_name", ["T5GemmaLoader", "T5Gemma2Loader"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_norm_unit_offset_is_added_in_float32(loader_name, dtype):
    torch = pytest.importorskip("torch")
    from ctranslate2.converters import transformers as converters
    from ctranslate2.specs.common_spec import LayerNormSpec

    norm = torch.nn.LayerNorm(8).to(getattr(torch, dtype))
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0, 0.125, 255, 256, 511, 512, 2048, 4096]))
    spec = LayerNormSpec(rms_norm=True)
    getattr(converters, loader_name)().set_layer_norm(spec, norm)
    assert spec.gamma.dtype == torch.float32
    torch.testing.assert_close(
        spec.gamma, norm.weight.detach().float() + 1, rtol=0, atol=0
    )


@pytest.fixture(scope="module")
def source_model():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers", minversion="5.9")
    torch.manual_seed(42)
    side = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        query_pre_attn_scalar=8,
    )
    config = tf.T5Gemma2Config(
        encoder=dict(
            text_config=side,
            vision_config=dict(
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
            ),
        ),
        decoder=side,
        image_token_index=63,
        eoi_token_index=62,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    return tf.T5Gemma2ForConditionalGeneration(config).eval()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_decoder_embedding_scale_uses_original_scalar(source_model, dtype):
    torch = pytest.importorskip("torch")
    from ctranslate2.converters.transformers import T5Gemma2Loader

    source = copy.deepcopy(source_model).to(getattr(torch, dtype))
    decoder = source.model.decoder.embed_tokens
    expected = source.config.decoder.hidden_size**0.5
    assert float(decoder.embed_scale) != expected
    encoder_scale = float(source.model.encoder.text_model.embed_tokens.embed_scale)
    spec = T5Gemma2Loader().get_model_spec(source)
    assert spec.decoder.scale_embeddings == expected
    assert spec.encoder.scale_embeddings == encoder_scale


@pytest.mark.parametrize("wrapped", [False, True])
def test_output_projection_preserves_untied_weights(source_model, wrapped):
    torch = pytest.importorskip("torch")
    from ctranslate2.converters.transformers import T5Gemma2Loader

    source = copy.deepcopy(source_model)
    head = torch.nn.Linear(32, 64, bias=False)
    with torch.no_grad():
        head.weight.fill_(0.125)
    if wrapped:
        wrapper = torch.nn.Module()
        wrapper.out_proj = head
        source.lm_head = wrapper
    else:
        source.lm_head = head
    expected = head.weight.detach().clone()
    spec = T5Gemma2Loader().get_model_spec(source)
    torch.testing.assert_close(spec.decoder.projection.weight, expected, rtol=0, atol=0)

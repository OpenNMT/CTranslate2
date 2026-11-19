import numpy as np
import pytest
import test_utils

import ctranslate2


def _build_model(architecture, final_softcap):
    import torch
    import transformers

    torch.manual_seed(42)
    kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        query_pre_attn_scalar=4,
        attn_logit_softcapping=None,
        final_logit_softcapping=final_softcap,
        sliding_window=64,
    )
    if architecture == "T5Gemma":
        config = transformers.T5GemmaConfig(encoder=kwargs, decoder=kwargs)
    else:
        config_class = (
            transformers.Gemma2Config
            if architecture == "Gemma2"
            else transformers.Gemma3TextConfig
        )
        config = config_class(**kwargs)
    model_class = getattr(transformers, architecture + "ForCausalLM", None)
    if architecture == "T5Gemma":
        model_class = transformers.T5GemmaForConditionalGeneration
    return model_class(config).eval()


@test_utils.only_on_linux
@pytest.mark.parametrize("architecture", ["Gemma2", "Gemma3", "T5Gemma"])
@pytest.mark.parametrize("final_softcap", [None, 0.0, 0.25])
def test_gemma_loader_preserves_final_logit_softcapping(architecture, final_softcap):
    import torch

    from ctranslate2.converters import transformers as converters
    from ctranslate2.specs import model_spec

    model = _build_model(architecture, final_softcap)
    with torch.no_grad():
        spec = getattr(converters, architecture + "Loader")().get_model_spec(model)

    if final_softcap:
        assert spec.decoder.final_logit_softcapping == np.float32(final_softcap)
    else:
        assert spec.decoder.final_logit_softcapping == model_spec.OPTIONAL


@test_utils.only_on_linux
@pytest.mark.parametrize("final_softcap", [None, 0.25])
def test_gemma2_final_logit_softcapping_matches_transformers(final_softcap, tmp_dir):
    import torch

    from ctranslate2.converters.transformers import Gemma2Loader

    # Keep attention uncapped to isolate the final logit transformation. A small
    # final cap makes the effect measurable on this randomly initialized model.
    model = _build_model("Gemma2", final_softcap)
    ids = [2, 5, 9, 14, 11]
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([ids])).logits.numpy()
        loader = Gemma2Loader()
        spec = loader.get_model_spec(model)

    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [
        "tok%d" % i for i in range(4, model.config.vocab_size)
    ]
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<bos>"
    spec.config.eos_token = "<eos>"
    spec.config.unk_token = "<unk>"
    spec.config.layer_norm_epsilon = model.config.rms_norm_eps
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(str(tmp_dir))

    generator = ctranslate2.Generator(
        str(tmp_dir), device="cpu", compute_type="float32", intra_threads=1
    )
    actual = np.asarray(generator.forward_batch([[tokens[i] for i in ids]]))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

import numpy as np
import pytest
import test_utils

import ctranslate2


@test_utils.only_on_linux
@pytest.mark.parametrize("architecture", ["Gemma2", "Gemma3", "T5Gemma"])
@pytest.mark.parametrize("final_softcap", [None, 0.0, 0.25])
def test_gemma_final_logit_softcapping(architecture, final_softcap, tmp_dir):
    import torch
    import transformers

    from ctranslate2.converters import transformers as converters
    from ctranslate2.specs import model_spec

    torch.manual_seed(42)
    loader = getattr(converters, architecture + "Loader")()
    model_class = getattr(transformers, loader.architecture_name)
    kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        query_pre_attn_scalar=4,
        attn_logit_softcapping=None,  # Isolate the final logit transformation.
        final_logit_softcapping=final_softcap,
        sliding_window=64,
    )
    if architecture == "T5Gemma":
        kwargs = dict(encoder=kwargs, decoder=kwargs)
    model = model_class(model_class.config_class(**kwargs)).eval()
    ids = [2, 5, 9, 14, 11]
    expected = None
    with torch.no_grad():
        if architecture == "Gemma2" and final_softcap != 0:
            expected = model(input_ids=torch.tensor([ids])).logits.numpy()
        spec = loader.get_model_spec(model)

    assert spec.decoder.final_logit_softcapping == (
        np.float32(final_softcap) if final_softcap else model_spec.OPTIONAL
    )
    if expected is None:
        return

    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [str(i) for i in range(4, 64)]
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

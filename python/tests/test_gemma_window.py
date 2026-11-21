"""Parity at window boundaries, including mixed attention and padded batches."""

import numpy as np
import pytest
import test_utils

import ctranslate2


@test_utils.only_on_linux
@pytest.mark.parametrize("scalar", [128, 144])
def test_gemma2_27b_query_scale(scalar, tmp_dir):
    import torch
    import transformers
    from ctranslate2.converters.transformers import Gemma2Loader

    torch.manual_seed(42)
    config = transformers.Gemma2Config(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        query_pre_attn_scalar=scalar,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
    )
    config._attn_implementation = "eager"
    model = transformers.Gemma2ForCausalLM(config).eval()
    ids = [2, 5, 9, 14, 11]
    with torch.no_grad():
        expected = model(torch.tensor([ids]), use_cache=False).logits.numpy()
        loader = Gemma2Loader()
        spec = loader.get_model_spec(model)
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [str(i) for i in range(4, 64)]
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<bos>"
    spec.config.eos_token = "<eos>"
    spec.config.unk_token = "<unk>"
    spec.config.layer_norm_epsilon = config.rms_norm_eps
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(str(tmp_dir))
    generator = ctranslate2.Generator(
        str(tmp_dir), device="cpu", compute_type="float32", intra_threads=1
    )
    actual = np.asarray(generator.forward_batch([[tokens[i] for i in ids]]))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@test_utils.only_on_linux
@pytest.mark.parametrize("representation", ["legacy", "modern"])
def test_gemma3_linear_rope(representation, tmp_dir):
    import torch
    import transformers
    from ctranslate2.converters.transformers import Gemma3Loader

    torch.manual_seed(42)
    config = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        query_pre_attn_scalar=8,
        sliding_window=64,
        rope_scaling={"rope_type": "linear", "factor": 8.0},
        rope_theta=1000000.0,
        rope_local_base_freq=10000.0,
    )
    config.layer_types = ["sliding_attention", "full_attention"]
    config._attn_implementation = "eager"
    model = transformers.Gemma3ForCausalLM(config).eval()
    ids = [2, 5, 9, 14, 11, 6, 8, 12, 17, 15]
    with torch.no_grad():
        expected = model(torch.tensor([ids]), use_cache=False).logits.numpy()
    # Keep weights and the reference fixed while exercising both serialized layouts.
    if representation == "legacy":
        if hasattr(config, "rope_parameters"):
            del config.rope_parameters
        config.rope_scaling = {"rope_type": "linear", "factor": 8.0}
        config.rope_theta = 1000000.0
        config.rope_local_base_freq = 10000.0
    else:
        config.rope_parameters = {
            "full_attention": {
                "rope_type": "linear",
                "factor": 8.0,
                "rope_theta": 1000000.0,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
    loader = Gemma3Loader()
    spec = loader.get_model_spec(model)
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [str(i) for i in range(4, 64)]
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<bos>"
    spec.config.eos_token = "<eos>"
    spec.config.unk_token = "<unk>"
    spec.config.layer_norm_epsilon = config.rms_norm_eps
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(str(tmp_dir))
    generator = ctranslate2.Generator(
        str(tmp_dir), device="cpu", compute_type="float32", intra_threads=1
    )
    actual = np.asarray(generator.forward_batch([[tokens[i] for i in ids]]))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@test_utils.only_on_linux
@pytest.mark.parametrize("cap", [None, 0.0, 0.25])
def test_t5gemma_final_softcap(cap, tmp_dir):
    import torch
    import transformers
    from ctranslate2.converters.transformers import T5GemmaLoader

    torch.manual_seed(42)
    module = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        query_pre_attn_scalar=8,
        sliding_window=64,
        attn_logit_softcapping=None,
        final_logit_softcapping=cap,
    )
    config = transformers.T5GemmaConfig(
        encoder=dict(module), decoder=dict(module, is_decoder=True), vocab_size=64
    )
    config._attn_implementation = "eager"
    model = transformers.T5GemmaForConditionalGeneration(config).eval()
    source, target = [2, 4, 5, 6], [7, 8, 9]
    with torch.no_grad():
        # A zero cap is deliberately disabled by CT2; HF would divide by zero.
        config.decoder.final_logit_softcapping = None if cap == 0 else cap
        logits = model(
            input_ids=torch.tensor([source]),
            decoder_input_ids=torch.tensor([[2] + target]),
            use_cache=False,
        ).logits[0]
        expected = (
            logits.log_softmax(-1)
            .gather(1, torch.tensor(target + [1])[:, None])[:, 0]
            .numpy()
        )
        config.decoder.final_logit_softcapping = cap
        loader = T5GemmaLoader()
        spec = loader.get_model_spec(model)
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"] + [str(i) for i in range(4, 64)]
    loader.set_vocabulary(spec, tokens)
    spec.config.bos_token = "<bos>"
    spec.config.eos_token = "<eos>"
    spec.config.unk_token = "<unk>"
    spec.config.decoder_start_token = "<bos>"
    spec.config.layer_norm_epsilon = config.decoder.rms_norm_eps
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(str(tmp_dir))
    translator = ctranslate2.Translator(
        str(tmp_dir), device="cpu", compute_type="float32", intra_threads=1
    )
    result = translator.score_batch(
        [[tokens[i] for i in source]], [[tokens[i] for i in target]], max_input_length=0
    )[0]
    np.testing.assert_allclose(result.log_probs, expected, atol=2e-5, rtol=2e-5)

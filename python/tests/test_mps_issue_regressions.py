"""Download-free GPT-2 regression for large finite GELU inputs."""

import copy

import numpy as np
import pytest
import test_utils

import ctranslate2


def _host(value):
    value = value.to_device(ctranslate2.Device.cpu).to(ctranslate2.DataType.float32)
    return np.array(value, copy=True)


@pytest.mark.parametrize(
    "dtype",
    ["float32", "float16", "bfloat16"],
)
@test_utils.require_mps
def test_gpt2_large_gelu_logits_and_generation(tmp_path, dtype):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from ctranslate2.converters.transformers import GPT2Loader

    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = tf.GPT2LMHeadModel(
        tf.GPT2Config(
            vocab_size=64,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_positions=64,
            bos_token_id=0,
            eos_token_id=1,
        )
    ).eval()
    # Finite large GELU inputs reproduce the real GPT-2 failure deterministically.
    with torch.no_grad():
        source.transformer.h[0].mlp.c_fc.bias[:2] = torch.tensor([20.0, -20.0])
    ids = [4, 5, 6, 7, 8]
    with torch.inference_mode():
        expected = source(torch.tensor([ids])).logits[0].numpy()
    spec = GPT2Loader().get_model_spec(copy.deepcopy(source))
    spec.register_vocabulary([f"t{i}" for i in range(64)])
    spec.config.bos_token = "t0"
    spec.config.eos_token = "t1"
    spec.config.unk_token = "t2"
    spec.config.layer_norm_epsilon = source.config.layer_norm_epsilon
    spec.validate()
    spec.optimize(quantization="float32")
    spec.save(tmp_path)
    model = ctranslate2.Generator(
        str(tmp_path),
        device="mps",
        compute_type=dtype,
        intra_threads=2,
        inter_threads=1,
    )
    logits = _host(model.forward_batch([ids]))[0]
    np.testing.assert_allclose(logits, expected, atol=0.01, rtol=0)
    for beam in (1, 2):
        result = model.generate_batch(
            [[f"t{i}" for i in ids]],
            beam_size=beam,
            max_length=4,
            min_length=4,
            end_token=[],
            include_prompt_in_result=False,
            return_scores=True,
        )[0]
        assert np.isfinite(result.scores).all()
        assert len(result.sequences_ids[0]) == 4
        assert all(0 <= token < 64 for token in result.sequences_ids[0])

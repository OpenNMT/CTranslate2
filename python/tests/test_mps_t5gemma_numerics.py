"""Download-free regressions for T5Gemma2 masking and FP16 numerics."""

import copy

import numpy as np
import pytest
import test_utils

import ctranslate2


def _host(value):
    value = value.to_device(ctranslate2.Device.cpu).to(ctranslate2.DataType.float32)
    return np.array(value, copy=True)


def test_t5gemma2_bfloat16_norm_offset_is_added_in_float32():
    torch = pytest.importorskip("torch")
    from ctranslate2.converters.transformers import T5Gemma2Loader
    from ctranslate2.specs.common_spec import LayerNormSpec

    norm = torch.nn.LayerNorm(6).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([255.0, 256.0, 511.0, 512.0, -512.0, 0.125]))
    spec = LayerNormSpec(rms_norm=True)
    T5Gemma2Loader().set_layer_norm(spec, norm)
    np.testing.assert_array_equal(
        spec.gamma.numpy(), norm.weight.detach().float().numpy() + 1
    )


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
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_t5gemma2_embedding_scale_retains_original_scalar(merged_model, dtype):
    import torch

    from ctranslate2.converters.transformers import T5Gemma2Loader

    source = copy.deepcopy(merged_model[0]).to(getattr(torch, dtype))
    expected = source.config.decoder.hidden_size**0.5
    encoder_scale = float(source.model.encoder.text_model.embed_tokens.embed_scale)
    assert float(source.model.decoder.embed_tokens.embed_scale) != expected
    spec = T5Gemma2Loader().get_model_spec(source)
    assert spec.encoder.scale_embeddings == encoder_scale
    assert spec.decoder.scale_embeddings == expected


@pytest.mark.parametrize("merged_model", [(1, False)], indirect=True)
def test_t5gemma2_half_conversion_preserves_large_norm_offset(merged_model):
    import torch

    from ctranslate2.converters.transformers import T5Gemma2Loader

    source = copy.deepcopy(merged_model[0]).half()
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


@pytest.mark.parametrize("beam", [1, 2, 4])
@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", "float32"),
        pytest.param("mps", "float32", marks=test_utils.require_mps),
        pytest.param("mps", "float16", marks=test_utils.require_mps),
    ],
)
def test_merged_attention_padded_batch_beams_and_source(
    merged_model, device, dtype, beam
):
    import torch

    source, path = merged_model
    translator = ctranslate2.Translator(
        str(path),
        device=device,
        compute_type=dtype,
        intra_threads=2,
        inter_threads=1,
    )
    ids = [[4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18]]
    inputs = [[f"t{i}" for i in row] for row in ids]
    options = dict(
        beam_size=beam,
        max_decoding_length=4,
        min_decoding_length=4,
        end_token=[],
        return_scores=True,
        return_logits_vocab=beam == 1,
    )
    if dtype == "float16" and beam > 1:
        # FP16 cumulative beam scores can tie at a 0.002 ULP. Check the
        # candidate set and each score rather than ordering equal candidates.
        options["num_hypotheses"] = beam
    # Repeated requests exercise cached encoder projections, padding removal,
    # first-step beam expansion, and subsequent beam cache reordering.
    for _ in range(2):
        batch = translator.translate_batch(inputs, **options)
        for index, row in enumerate(inputs):
            alone = translator.translate_batch([row], **options)[0]
            actual = dict(zip(map(tuple, batch[index].hypotheses), batch[index].scores))
            expected_scores = dict(zip(map(tuple, alone.hypotheses), alone.scores))
            if actual.keys() != expected_scores.keys():
                # A candidate at the beam cutoff can change after FP16 score
                # rounding. Allow only one such replacement, independently
                # verify its probability, and require it to tie the cutoff.
                assert dtype == "float16" and beam > 1
                unshared = actual.keys() ^ expected_scores.keys()
                assert len(unshared) == 2
                for scores in (actual, expected_scores):
                    for candidate in scores.keys() & unshared:
                        candidate_ids = [int(t[1:]) for t in candidate]
                        with torch.inference_mode():
                            reference = (
                                source(
                                    input_ids=torch.tensor([ids[index]]),
                                    decoder_input_ids=torch.tensor(
                                        [[2] + candidate_ids[:-1]]
                                    ),
                                )
                                .logits[0]
                                .log_softmax(-1)
                            )
                            score = (
                                reference[
                                    torch.arange(len(candidate_ids)),
                                    candidate_ids,
                                ]
                                .mean()
                                .item()
                            )
                        assert abs(scores[candidate] - score) < 0.003
                        assert abs(scores[candidate] - min(scores.values())) < 0.003
            shared = actual.keys() & expected_scores.keys()
            np.testing.assert_allclose(
                [actual[tokens] for tokens in shared],
                [expected_scores[tokens] for tokens in shared],
                atol=0.003,
                rtol=0,
            )
            assert np.isfinite(batch[index].scores).all()
            if beam == 1:
                logits = np.stack(
                    [_host(x).reshape(-1) for x in batch[index].logits[0]]
                )
                separate = np.stack([_host(x).reshape(-1) for x in alone.logits[0]])
                np.testing.assert_allclose(logits, separate, atol=0.003, rtol=0)
                decoded = [2] + [int(t[1:]) for t in batch[index].hypotheses[0][:-1]]
                with torch.inference_mode():
                    expected = (
                        source(
                            input_ids=torch.tensor([ids[index]]),
                            decoder_input_ids=torch.tensor([decoded]),
                        )
                        .logits[0]
                        .numpy()
                    )
                np.testing.assert_allclose(logits, expected, atol=0.003, rtol=0)


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", "float32"),
        pytest.param("mps", "float32", marks=test_utils.require_mps),
        pytest.param("mps", "float16", marks=test_utils.require_mps),
    ],
)
def test_merged_attention_causal_prefix_scoring(merged_model, device, dtype):
    import torch

    source, path = merged_model
    translator = ctranslate2.Translator(
        str(path),
        device=device,
        compute_type=dtype,
        intra_threads=2,
        inter_threads=1,
    )
    source_ids = [[4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]
    inputs = [[f"t{i}" for i in row] for row in source_ids]
    targets = [["t10", "t11", "t12", "t13", "t14"], ["t20", "t21"]]
    results = translator.score_batch(inputs, targets)
    for ids, result in zip(source_ids, results):
        expected_ids = [int(token[1:]) for token in result.tokens]
        with torch.inference_mode():
            logits = source(
                input_ids=torch.tensor([ids]),
                decoder_input_ids=torch.tensor([[2] + expected_ids[:-1]]),
            ).logits[0]
            expected = logits.log_softmax(-1)[
                torch.arange(len(expected_ids)), expected_ids
            ].numpy()
        np.testing.assert_allclose(result.log_probs, expected, atol=0.01, rtol=0)

"""Merged-attention mask regressions independent of FP16 execution."""

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
    # Give the fixture a plain, untied head so it does not depend on the
    # separate converter fix for wrapped output projections.
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


@pytest.mark.parametrize("beam", [1, 2, 4])
@test_utils.on_available_devices
def test_merged_attention_padded_batch_beams_and_source(merged_model, device, beam):
    import torch

    source, path = merged_model
    translator = ctranslate2.Translator(
        str(path),
        device=device,
        compute_type="float32",
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
    for _ in range(2):
        batch = translator.translate_batch(inputs, **options)
        for index, row in enumerate(inputs):
            alone = translator.translate_batch([row], **options)[0]
            assert batch[index].hypotheses == alone.hypotheses
            assert np.isfinite(batch[index].scores).all()
            np.testing.assert_allclose(
                batch[index].scores, alone.scores, atol=0.003, rtol=0
            )
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


@test_utils.on_available_devices
def test_merged_attention_causal_prefix_scoring(merged_model, device):
    import torch

    source, path = merged_model
    translator = ctranslate2.Translator(
        str(path),
        device=device,
        compute_type="float32",
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

# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np

import ctranslate2

from ctranslate2.specs.model_spec import OPTIONAL, index_spec
from ctranslate2.specs import transformer_spec
from ctranslate2.converters import opennmt_tf


_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data"
)


def _get_model_path():
    return os.path.join(_TEST_DATA_DIR, "models", "v2", "aren-transliteration")


def _get_transliterator():
    return ctranslate2.Translator(_get_model_path())


def test_invalid_model_path():
    with pytest.raises(RuntimeError):
        ctranslate2.Translator("xxx")


def test_invalid_device_settings():
    model_path = _get_model_path()
    with pytest.raises(ValueError):
        ctranslate2.Translator(model_path, device_index=[])
    with pytest.raises(ValueError):
        ctranslate2.Translator(model_path, device_index=[0, 1])


def test_contains_model(tmpdir):
    assert ctranslate2.contains_model(_get_model_path())

    model_dir = tmpdir.join("model")
    model_dir.ensure(dir=1)
    assert not ctranslate2.contains_model(str(model_dir))
    model_dir.join("model.bin").ensure(file=1)
    assert ctranslate2.contains_model(str(model_dir))


def test_get_supported_compute_types():
    assert ctranslate2.get_supported_compute_types("cpu") == {"float", "int16", "int8"}


def test_translator_properties():
    translator = ctranslate2.Translator(_get_model_path(), inter_threads=2)
    assert translator.model_is_loaded
    assert translator.device == "cpu"
    assert translator.device_index == [0, 0]
    assert translator.num_translators == 2
    assert translator.num_queued_batches == 0


def test_compute_type():
    model_path = _get_model_path()
    with pytest.raises(ValueError):
        ctranslate2.Translator(model_path, compute_type="float64")
    with pytest.raises(TypeError):
        ctranslate2.Translator(model_path, compute_type=["int8", "int16"])
    ctranslate2.Translator(model_path, compute_type="int8")
    ctranslate2.Translator(model_path, compute_type={"cuda": "float16", "cpu": "int8"})


@pytest.mark.parametrize("max_batch_size", [0, 1])
def test_batch_translation(max_batch_size):
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        max_batch_size=max_batch_size,
        return_scores=True,
    )
    assert len(output) == 2
    assert output[0].hypotheses == [["a", "t", "z", "m", "o", "n"]]
    assert output[1].hypotheses == [["a", "c", "h", "i", "s", "o", "n"]]
    assert output[0].scores[0] < 0
    assert not output[0].attention

    expected_repr = "TranslationResult(hypotheses=%s, scores=%s, attention=[])" % (
        output[0].hypotheses,
        output[0].scores,
    )
    assert repr(output[0]) == expected_repr

    # Check backward compatibility with previous result format.
    assert len(output[0]) == 1  # One hypothesis.
    assert len(output[1]) == 1
    assert output[0][0]["tokens"] == ["a", "t", "z", "m", "o", "n"]
    assert output[0][0]["score"] < 0
    assert "attention" not in output[0][0]
    assert output[1][0]["tokens"] == ["a", "c", "h", "i", "s", "o", "n"]


def test_batch_translation_async():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        asynchronous=True,
    )
    assert output[0].result().hypotheses == [["a", "t", "z", "m", "o", "n"]]
    assert output[1].result().hypotheses == [["a", "c", "h", "i", "s", "o", "n"]]
    assert output[0].done()
    assert output[1].done()


def test_file_translation(tmpdir):
    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(input_path, "w") as input_file:
        input_file.write("آ ت ز م و ن")
        input_file.write("\n")
        input_file.write("آ ت ش ي س و ن")
        input_file.write("\n")
    translator = _get_transliterator()
    stats = translator.translate_file(input_path, output_path)
    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "a t z m o n"
        assert lines[1].strip() == "a c h i s o n"
    assert stats.num_tokens == 13
    assert stats.num_examples == 2
    assert isinstance(stats.total_time_in_ms, float)

    expected_repr = (
        "TranslationStats(num_tokens=13, num_examples=2, total_time_in_ms=%s)"
        % stats.total_time_in_ms
    )
    assert repr(stats) == expected_repr

    # For backward compatibility, ensure that stats is tuple-like.
    num_tokens, num_examples, total_time_in_ms = stats
    assert stats[0] == num_tokens
    assert stats[1] == num_examples
    assert stats[2] == total_time_in_ms


def test_raw_file_translation(tmpdir):
    input_path = str(tmpdir.join("input.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(input_path, "w") as input_file:
        input_file.write("آتزمون")
        input_file.write("\n")
        input_file.write("آتشيسون")
        input_file.write("\n")

    translator = ctranslate2.Translator(_get_model_path())
    tokenize_fn = lambda text: list(text)
    detokenize_fn = lambda tokens: "".join(tokens)

    with pytest.raises(ValueError):
        translator.translate_file(
            input_path, output_path, source_tokenize_fn=tokenize_fn
        )
    with pytest.raises(ValueError):
        translator.translate_file(
            input_path, output_path, target_detokenize_fn=detokenize_fn
        )

    translator.translate_file(
        input_path,
        output_path,
        source_tokenize_fn=tokenize_fn,
        target_detokenize_fn=detokenize_fn,
    )

    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "atzmon"
        assert lines[1].strip() == "achison"


def test_file_translation_with_prefix(tmpdir):
    source_path = str(tmpdir.join("input.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(source_path, "w") as source_file:
        source_file.write("آ ت ز م و ن")
        source_file.write("\n")
        source_file.write("آ ت ش ي س و ن")
        source_file.write("\n")
    with open(target_path, "w") as target_file:
        target_file.write("a t s\n")

    translator = _get_transliterator()

    with pytest.raises(RuntimeError):
        # One line is missing from target_path.
        translator.translate_file(
            source_path,
            output_path,
            target_path=target_path,
        )

    with open(target_path, "a") as target_file:
        target_file.write("\n")  # No prefix.

    translator.translate_file(
        source_path,
        output_path,
        target_path=target_path,
    )

    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "a t s u m o n"
        assert lines[1].strip() == "a c h i s o n"


def test_raw_file_translation_with_prefix(tmpdir):
    source_path = str(tmpdir.join("input.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    with open(source_path, "w") as source_file:
        source_file.write("آتزمون")
        source_file.write("\n")
        source_file.write("آتشيسون")
        source_file.write("\n")
    with open(target_path, "w") as target_file:
        # Write target in reverse to use a different tokenization.
        target_file.write("sta\n")
        target_file.write("\n")

    translator = ctranslate2.Translator(_get_model_path())
    source_tokenize_fn = lambda text: list(text)
    target_tokenize_fn = lambda text: list(reversed(list(text)))
    detokenize_fn = lambda tokens: "".join(tokens)

    with pytest.raises(ValueError):
        # Target tokenization is missing.
        translator.translate_file(
            source_path,
            output_path,
            target_path=target_path,
            source_tokenize_fn=source_tokenize_fn,
            target_detokenize_fn=detokenize_fn,
        )

    translator.translate_file(
        source_path,
        output_path,
        target_path=target_path,
        source_tokenize_fn=source_tokenize_fn,
        target_tokenize_fn=target_tokenize_fn,
        target_detokenize_fn=detokenize_fn,
    )

    with open(output_path) as output_file:
        lines = output_file.readlines()
        assert lines[0].strip() == "atsumon"
        assert lines[1].strip() == "achison"


def test_empty_translation():
    translator = _get_transliterator()
    assert translator.translate_batch([]) == []


def test_invalid_translation_options():
    translator = _get_transliterator()
    with pytest.raises(ValueError):
        translator.translate_batch(
            [["آ", "ت", "ز", "م", "و", "ن"]],
            min_decoding_length=10,
            max_decoding_length=5,
        )


def test_hard_target_prefix():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], None],
    )
    assert output[0].hypotheses[0][:3] == ["a", "t", "s"]
    assert output[1].hypotheses[0] == ["a", "c", "h", "i", "s", "o", "n"]


def test_strongly_biased_target_prefix():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], None],
        prefix_bias_beta=0.9999999,
    )
    assert output[0].hypotheses[0][:3] == ["a", "t", "s"]
    assert output[1].hypotheses[0] == ["a", "c", "h", "i", "s", "o", "n"]


def test_weakly_biased_target_prefix():
    translator = _get_transliterator()
    unconstrained_output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        return_scores=True,
    )
    weakly_biased_output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ت", "ش", "ي", "س", "و", "ن"]],
        target_prefix=[["a", "t", "s"], ["s", "i", "o"]],
        prefix_bias_beta=0.0000001,
        return_scores=True,
    )
    assert (
        unconstrained_output[0].hypotheses[0] == weakly_biased_output[0].hypotheses[0]
    )
    assert (
        abs(unconstrained_output[0].scores[0] - weakly_biased_output[0].scores[0])
        < 0.00001
    )

    assert (
        unconstrained_output[1].hypotheses[0] == weakly_biased_output[1].hypotheses[0]
    )
    assert (
        abs(unconstrained_output[1].scores[0] - weakly_biased_output[1].scores[0])
        < 0.00001
    )


def test_num_hypotheses():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=4, num_hypotheses=2
    )
    assert len(output[0].hypotheses) == 2


def test_max_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], max_decoding_length=2
    )
    assert output[0].hypotheses[0] == ["a", "t"]


def test_min_decoding_length():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], min_decoding_length=7
    )
    assert len(output[0].hypotheses[0]) > 6  # 6 is the expected target length.


def test_return_attention():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], return_attention=True
    )
    attention = output[0].attention[0]
    assert len(attention) == 6  # Target length.
    for vector in attention:
        assert len(vector) == 6  # Source length.
        assert all(isinstance(value, float) for value in vector)


def test_ignore_scores():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=1, return_scores=False
    )
    assert not output[0].scores


def test_return_alternatives():
    translator = _get_transliterator()
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]],
        target_prefix=[["a", "t"]],
        num_hypotheses=10,
        return_alternatives=True,
    )
    assert len(output[0].hypotheses) == 10
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]
    assert output[0].hypotheses[1] == ["a", "t", "s", "u", "m", "o", "n"]


def test_score_api(tmpdir):
    source = [
        ["آ", "ت", "ز", "م", "و", "ن"],
        ["آ", "ت", "ش", "ي", "س", "و", "ن"],
    ]
    target = [
        ["a", "t", "z", "m", "o", "n"],
        ["a", "c", "h", "i", "s", "o", "n"],
    ]
    expected = [
        [-0.106023, -0.065410, -0.056002, -0.447953, -0.230714, -0.092184],
        [-0.072660, -0.300309, -0.181187, -0.395671, -0.025631, -0.123466, -0.002034],
    ]

    translator = _get_transliterator()
    all_scores = translator.score_batch(source, target)
    for scores, expected_scores in zip(all_scores, expected):
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-4)

    def _write_tokens(batch_tokens, path):
        with open(path, "w") as f:
            for tokens in batch_tokens:
                f.write(" ".join(tokens))
                f.write("\n")

    source_path = str(tmpdir.join("source.txt"))
    target_path = str(tmpdir.join("target.txt"))
    output_path = str(tmpdir.join("output.txt"))
    _write_tokens(source, source_path)
    _write_tokens(target, target_path)

    stats = translator.score_file(source_path, target_path, output_path)
    assert stats.num_examples == 2
    assert stats.num_tokens == 13

    with open(output_path) as output_file:
        scores = [np.mean(scores) for scores in expected]
        for line, expected_tokens, expected_score in zip(output_file, target, scores):
            tokens = line.strip().split()
            assert float(tokens[0]) == pytest.approx(expected_score, 1e-4)
            assert tokens[1] == "|||"
            assert tokens[2:] == expected_tokens


@pytest.mark.parametrize("to_cpu", [False, True])
def test_model_unload(to_cpu):
    batch = [["آ", "ت", "ز", "م", "و", "ن"]]
    translator = _get_transliterator()
    translator.unload_model(to_cpu=to_cpu)
    if not to_cpu:
        with pytest.raises(RuntimeError, match="unloaded"):
            translator.translate_batch(batch)
    translator.load_model()
    output = translator.translate_batch(batch)
    assert len(output) == 1
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


_FRAMEWORK_DATA_EXIST = os.path.isdir(
    os.path.join(_TEST_DATA_DIR, "models", "transliteration-aren-all")
)


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
@pytest.mark.parametrize(
    "model_path,src_vocab,tgt_vocab,model_spec",
    [
        (
            "v1/checkpoint",
            "ar.vocab",
            "en.vocab",
            ctranslate2.specs.TransformerSpec(6, 8),
        ),
        (
            "v2/checkpoint",
            "ar.vocab",
            "en.vocab",
            ctranslate2.specs.TransformerSpec(6, 8),
        ),
    ],
)
def test_opennmt_tf_model_conversion(
    tmpdir, model_path, src_vocab, tgt_vocab, model_spec
):
    model_path = os.path.join(
        _TEST_DATA_DIR, "models", "transliteration-aren-all", "opennmt_tf", model_path
    )
    src_vocab = os.path.join(model_path, src_vocab)
    tgt_vocab = os.path.join(model_path, tgt_vocab)
    converter = ctranslate2.converters.OpenNMTTFConverter(
        model_spec,
        src_vocab,
        tgt_vocab,
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    assert os.path.isfile(os.path.join(output_dir, "source_vocabulary.txt"))
    assert os.path.isfile(os.path.join(output_dir, "target_vocabulary.txt"))
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
@pytest.mark.parametrize("quantization", ["float16", "int16", "int8", "int8_float16"])
def test_opennmt_tf_model_quantization(tmpdir, quantization):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )
    converter = ctranslate2.converters.OpenNMTTFConverter(
        ctranslate2.specs.TransformerSpec(6, 8),
        os.path.join(model_path, "ar.vocab"),
        os.path.join(model_path, "en.vocab"),
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, quantization=quantization)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_tf_variables_conversion(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )
    src_vocab = os.path.join(model_path, "ar.vocab")
    tgt_vocab = os.path.join(model_path, "en.vocab")
    _, variables = opennmt_tf.load_model(model_path)
    converter = ctranslate2.converters.OpenNMTTFConverter(
        ctranslate2.specs.TransformerSpec(6, 8),
        src_vocab,
        tgt_vocab,
        variables=variables,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_tf_model_conversion_invalid_vocab(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_tf",
        "v2",
        "checkpoint",
    )
    # Swap source and target vocabularies.
    converter = ctranslate2.converters.OpenNMTTFConverter(
        ctranslate2.specs.TransformerSpec(6, 8),
        os.path.join(model_path, "en.vocab"),
        os.path.join(model_path, "ar.vocab"),
        model_path=model_path,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    with pytest.raises(ValueError):
        converter.convert(output_dir)


def _build_model_with_shared_embeddings(tmpdir):
    import opennmt

    vocab = opennmt.data.Vocab()
    for i in range(10):
        vocab.add(str(i))
    vocab_path = str(tmpdir.join("vocab.txt"))
    vocab.serialize(vocab_path)

    num_layers = 3
    num_heads = 4
    model = opennmt.models.Transformer(
        opennmt.inputters.WordEmbedder(32),
        opennmt.inputters.WordEmbedder(32),
        num_layers,
        num_units=32,
        num_heads=num_heads,
        ffn_inner_dim=64,
        share_embeddings=opennmt.models.EmbeddingsSharingLevel.ALL,
    )
    model.initialize({"source_vocabulary": vocab_path, "target_vocabulary": vocab_path})
    model.create_variables()
    return model, vocab_path


def test_opennmt_tf_shared_embeddings_conversion(tmpdir):
    # Issue https://github.com/OpenNMT/CTranslate2/issues/118
    import tensorflow as tf

    model, vocab_path = _build_model_with_shared_embeddings(tmpdir)

    checkpoint_prefix = str(tmpdir.join("ckpt"))
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.write(checkpoint_prefix)

    converter = ctranslate2.converters.OpenNMTTFConverter(
        model.ctranslate2_spec,
        vocab_path,
        vocab_path,
        model_path=checkpoint_prefix,
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)

    assert os.path.isfile(os.path.join(output_dir, "shared_vocabulary.txt"))

    # Check that the translation runs.
    translator = ctranslate2.Translator(output_dir)
    translator.translate_batch([["1", "2", "3"]], max_decoding_length=10)


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_py_model_conversion(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        "aren_7000.pt",
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_opennmt_py_relative_transformer(tmpdir):
    model_path = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "opennmt_py",
        "aren_relative_6000.pt",
    )
    converter = ctranslate2.converters.OpenNMTPyConverter(model_path)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"], ["آ", "ر", "ث", "ر"]]
    )
    assert output[0].hypotheses[0] == ["a", "t", "z", "o", "m", "o", "n"]
    assert output[1].hypotheses[0] == ["a", "r", "t", "h", "e", "r"]


@pytest.mark.skipif(not _FRAMEWORK_DATA_EXIST, reason="Data files are not available")
def test_fairseq_model_conversion(tmpdir):
    data_dir = os.path.join(
        _TEST_DATA_DIR,
        "models",
        "transliteration-aren-all",
        "fairseq",
    )
    converter = ctranslate2.converters.FairseqConverter(
        os.path.join(data_dir, "model.pt"), data_dir
    )
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
    assert output[0].hypotheses[0] == ["a", "t", "z", "m", "o", "n"]


def test_layer_spec_validate():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([5], dtype=np.float16)

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.zeros([5], dtype=np.float32)
            self.b = np.zeros([5], dtype=np.float16)
            self.c = np.zeros([5], dtype=np.int32)
            self.d = OPTIONAL
            self.e = SubSpec()
            self.f = True

    spec = Spec()
    spec.validate()
    assert spec.a.dtype == np.float32
    assert spec.b.dtype == np.float32
    assert spec.c.dtype == np.int32
    assert spec.d == OPTIONAL
    assert spec.e.a.dtype == np.float32
    assert spec.f.dtype == np.int8


def test_layer_spec_optimize():
    class SubSpec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([6], dtype=np.float32)

    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.a = np.ones([5], dtype=np.float32)
            self.b = np.ones([5], dtype=np.float32)
            self.c = np.zeros([5], dtype=np.int32)
            self.d = np.dtype("float32").type(3.14)
            self.weight = np.ones([5, 4], dtype=np.float32)
            self.sub = SubSpec()

    spec = Spec()
    spec.optimize(quantization="int16")
    assert spec.a.dtype == np.float32
    assert spec.b == "a"
    assert spec.c.dtype == np.int32
    assert spec.d.dtype == np.float32
    assert spec.weight.dtype == np.int16
    assert spec.weight_scale.dtype == np.float32

    spec = Spec()
    spec.optimize(quantization="float16")
    assert spec.a.dtype == np.float16
    assert spec.b == "a"
    assert spec.c.dtype == np.int32
    assert spec.d.dtype == np.float32
    assert spec.weight.dtype == np.float16
    assert spec.sub.a.dtype == np.float16


def test_int8_quantization():
    class Spec(ctranslate2.specs.LayerSpec):
        def __init__(self):
            self.weight = np.array([[-10, -3, 5, 2], [0, 0, 0, 0]], dtype=np.float32)

    spec = Spec()
    spec.optimize(quantization="int8")
    assert np.array_equal(
        spec.weight, np.array([[-127, -38, 63, 25], [0, 0, 0, 0]], dtype=np.int8)
    )
    assert np.array_equal(spec.weight_scale, np.array([12.7, 1], dtype=np.float32))


def test_index_spec():
    spec = ctranslate2.specs.TransformerSpec(6, 8)
    assert isinstance(
        index_spec(spec, "encoder/layer_5"),
        transformer_spec.TransformerEncoderLayerSpec,
    )
    assert isinstance(
        index_spec(spec, "encoder/layer_5/ffn"), transformer_spec.FeedForwardSpec
    )

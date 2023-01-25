import os

import pytest
import test_utils

import ctranslate2


def test_marian_model_conversion(tmpdir):
    model_dir = os.path.join(test_utils.get_data_dir(), "models", "opus-mt-ende")
    converter = ctranslate2.converters.OpusMTConverter(model_dir)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir)
    translator = ctranslate2.Translator(output_dir)
    output = translator.translate_batch([["▁Hello", "▁world", "!"]])
    assert output[0].hypotheses[0] == ["▁Hallo", "▁Welt", "!"]


@pytest.mark.parametrize(
    "quantization", [None, "int8", "int16", "float16", "int8_float16"]
)
def test_marian_model_quantization(tmpdir, quantization):
    model_dir = os.path.join(test_utils.get_data_dir(), "models", "opus-mt-ende")
    converter = ctranslate2.converters.OpusMTConverter(model_dir)
    output_dir = str(tmpdir.join("ctranslate2_model"))
    converter.convert(output_dir, quantization=quantization)

    compute_types = ["default"] + list(ctranslate2.get_supported_compute_types("cpu"))

    for compute_type in compute_types:
        translator = ctranslate2.Translator(output_dir, compute_type=compute_type)
        output = translator.translate_batch([["▁Hello", "▁world", "!"]])
        assert output[0].hypotheses[0] == ["▁Hallo", "▁Welt", "!"]

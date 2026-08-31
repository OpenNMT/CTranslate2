import os

import pytest

import ctranslate2


def _require_mps():
    if os.environ.get("CT2_TEST_MPS") != "1":
        pytest.skip("Set CT2_TEST_MPS=1 to run tests on a real Metal device")

    assert hasattr(ctranslate2, "get_mps_device_count")
    assert ctranslate2.get_mps_device_count() > 0


def _get_model_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "data",
        "models",
        "v2",
        "aren-transliteration",
    )


def test_mps_is_explicit_opt_in():
    _require_mps()

    translator = ctranslate2.Translator(_get_model_path(), device="auto")
    assert translator.device == "cpu"


def test_mps_float16_translation_matches_cpu():
    _require_mps()

    source = [["آ", "ت", "ز", "م", "و", "ن"]]
    options = dict(beam_size=1, max_decoding_length=12)
    cpu = ctranslate2.Translator(
        _get_model_path(), device="cpu", compute_type="float32"
    )
    mps = ctranslate2.Translator(
        _get_model_path(), device="mps", compute_type="float16"
    )

    expected = cpu.translate_batch(source, **options)[0].hypotheses[0]
    output = mps.translate_batch(source, **options)[0].hypotheses[0]
    assert output == expected

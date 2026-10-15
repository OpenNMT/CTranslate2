import os

import numpy as np
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


def _get_int8_model_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "data",
        "models",
        "v2",
        "aren-transliteration-i8",
    )


def test_mps_is_explicit_opt_in():
    _require_mps()

    translator = ctranslate2.Translator(_get_model_path(), device="auto")
    assert translator.device == "cpu"


def test_mps_storage_does_not_expose_cuda_array_interface():
    _require_mps()

    array = np.arange(4, dtype=np.float32)
    storage = ctranslate2.StorageView.from_array(array).to_device(
        ctranslate2.Device.mps
    )

    assert storage.device == "mps"
    assert not hasattr(storage, "__cuda_array_interface__")


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


@pytest.mark.parametrize("requested_compute_type", ["int8", "int8_float16"])
def test_mps_int8_requests_use_float16_path(requested_compute_type, monkeypatch):
    _require_mps()
    monkeypatch.delenv("CT2_MPS_CACHE_INT8_FP16", raising=False)

    source = [["آ", "ت", "ز", "م", "و", "ن"]]
    options = dict(beam_size=1, max_decoding_length=12)
    float16 = ctranslate2.Translator(
        _get_model_path(), device="mps", compute_type="float16"
    )
    requested = ctranslate2.Translator(
        _get_model_path(), device="mps", compute_type=requested_compute_type
    )

    assert requested.compute_type == "float16"
    expected = float16.translate_batch(source, **options)[0].hypotheses[0]
    output = requested.translate_batch(source, **options)[0].hypotheses[0]
    assert output == expected


def test_mps_stored_int8_model_is_converted_once(monkeypatch):
    _require_mps()
    monkeypatch.delenv("CT2_MPS_CACHE_INT8_FP16", raising=False)

    translator = ctranslate2.Translator(
        _get_int8_model_path(), device="mps", compute_type="int8"
    )

    assert translator.compute_type == "float16"
    result = translator.translate_batch(
        [["آ", "ت", "ز", "م", "و", "ن"]], beam_size=1, max_decoding_length=12
    )[0]
    assert result.hypotheses[0] == ["a", "t", "z", "m", "o", "n"]

"""Tests specific to the ROCm/HIP GPU backend."""

import numpy as np
import pytest

import ctranslate2

from test_utils import require_rocm


@require_rocm
class TestROCmDeviceDetection:
    def test_device_count_positive(self):
        assert ctranslate2.get_cuda_device_count() >= 1

    def test_device_name_non_empty(self):
        name = ctranslate2.get_device_name("cuda", 0)
        assert isinstance(name, str) and len(name) > 0


@require_rocm
class TestROCmComputeTypes:
    def test_float32_supported(self):
        types = ctranslate2.get_supported_compute_types("cuda")
        assert "float32" in types

    def test_float16_supported(self):
        types = ctranslate2.get_supported_compute_types("cuda")
        assert "float16" in types

    def test_bfloat16_supported(self):
        types = ctranslate2.get_supported_compute_types("cuda")
        assert "bfloat16" in types

    def test_int8_supported(self):
        types = ctranslate2.get_supported_compute_types("cuda")
        assert "int8" in types


@require_rocm
class TestROCmStorageView:
    def test_allocate_on_gpu(self):
        x = ctranslate2.StorageView([4], ctranslate2.DataType.FLOAT32, device="cuda")
        assert x.device == "cuda"
        assert x.shape == [4]

    def test_cpu_to_gpu_roundtrip(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = ctranslate2.StorageView.from_array(data)
        x_gpu = x.to("cuda")
        x_back = x_gpu.to("cpu")
        np.testing.assert_array_equal(np.array(x_back), data)

    def test_float16_on_gpu(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
        x = ctranslate2.StorageView.from_array(data, device="cuda")
        assert x.dtype == ctranslate2.DataType.FLOAT16
        assert x.device == "cuda"

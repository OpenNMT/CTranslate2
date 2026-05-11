"""Tests specific to the ROCm/HIP GPU backend."""

import numpy as np
import pytest

import ctranslate2

from test_utils import require_rocm


@require_rocm
class TestROCmDeviceDetection:
    def test_device_count_positive(self):
        assert ctranslate2.get_cuda_device_count() >= 1

    def test_device_enum_accessible(self):
        assert ctranslate2.Device.cuda is not None


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
        data = np.zeros(4, dtype=np.float32)
        x = ctranslate2.StorageView.from_array(data)
        x_gpu = x.to_device(ctranslate2.Device.cuda)
        assert x_gpu.device == "cuda"
        assert x_gpu.shape == [4]

    def test_cpu_to_gpu_roundtrip(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = ctranslate2.StorageView.from_array(data)
        x_gpu = x.to_device(ctranslate2.Device.cuda)
        x_back = x_gpu.to_device(ctranslate2.Device.cpu)
        np.testing.assert_array_equal(np.array(x_back), data)

    def test_float16_on_gpu(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = ctranslate2.StorageView.from_array(data)
        x_gpu = x.to_device(ctranslate2.Device.cuda)
        x_fp16 = x_gpu.to(ctranslate2.DataType.float16)
        assert x_fp16.dtype == ctranslate2.DataType.float16
        assert x_fp16.device == "cuda"

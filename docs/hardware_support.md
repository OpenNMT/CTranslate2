# Hardware support

```{attention}
The information below is only valid for the prebuilt binaries. If you compiled the project from the sources, the supported hardware will depend on the selected backend and compilation flags.
```

## CPU

* x86-64 processors supporting at least SSE 4.1
* AArch64/ARM64 processors

On x86-64, prebuilt binaries are configured to automatically select the best backend and instruction set architecture for the platform (AVX, AVX2, or AVX512). In particular, they are compiled with both [Intel MKL](https://software.intel.com/en-us/mkl) and [oneDNN](https://github.com/oneapi-src/oneDNN) so that Intel MKL is only used on Intel processors where it performs best, whereas oneDNN is used on other x86-64 processors such as AMD.

```{tip}
See the [environment variables](environment_variables.md) `CT2_USE_MKL` and `CT2_FORCE_CPU_ISA` to control this behavior.
```

## GPU

### NVIDIA

* NVIDIA GPUs with a Compute Capability greater or equal to 3.5

The driver requirement depends on the CUDA version. See the [CUDA Compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for more information.

```{note}
**NVIDIA Jetson (aarch64):** Jetson devices are supported when building from source with `-DWITH_CUDA=ON`. The prebuilt aarch64 wheels on PyPI are CPU-only. See {ref}`installation:nvidia jetson (aarch64 + cuda)` for build instructions.
```

### AMD (ROCm)

* AMD RDNA 2 and RDNA 3 GPUs (RX 6000 and RX 7000 series)

Prebuilt Python wheels for ROCm are available on the [releases page](https://github.com/OpenNMT/CTranslate2/releases/). To build from source on Windows, see the {doc}`building_rocm_windows` guide.

```{note}
The ROCm backend is exposed through the same `device="cuda"` API as NVIDIA GPUs. CTranslate2 uses a unified device abstraction for both backends.
```

### Apple Metal Performance Shaders

The experimental MPS backend supports Apple Silicon Macs running macOS 11 or
newer. It is not included in the prebuilt Python wheels: build from source
with `-DWITH_MPS=ON`, then select `device="mps"` in the C++ or Python API.

The backend supports FP32, FP16, BF16, and experimental signed INT8
computation. FP16 is the recommended compute type and is selected by
`compute_type="auto"`. Requests for `int8` and `int8_float16` also resolve to
FP16 by default because native INT8 is slower on current Apple GPUs and can
change decoding decisions. See [Quantization](quantization.md) for the native
INT8 opt-in and exact type conversions, and [Installation](installation.md)
for a build example.

The MPS backend cannot be enabled together with CUDA or HIP in the same build.
FlashAttention, AWQ INT4, INT16 GEMM, distributed collectives, and
packed/shifted-u8 INT8 GEMM are not currently implemented on MPS.

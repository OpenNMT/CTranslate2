# Building with ROCm on Windows

This guide describes how to build CTranslate2 with AMD GPU support (ROCm/HIP) on Windows. It was validated on the following system:

| Component | Version |
| --- | --- |
| OS | Windows 11 (Build 26200) |
| GPU | AMD Radeon RX 7900 XTX (gfx1100, RDNA 3) |
| ROCm | 7.2.0 |
| Python | 3.11 |

## Supported GPUs

ROCm on Windows supports AMD RDNA 2 and RDNA 3 GPUs (RX 6000 and RX 7000 series). The HIP architecture target for each GPU can be found in the [ROCm GPU compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

Common targets:

| GPU | HIP architecture |
| --- | --- |
| RX 6700, 6800, 6900 | `gfx1030` |
| RX 7600, 7700 | `gfx1102` |
| RX 7800, 7900 | `gfx1101` |
| RX 7900 XTX, 7900 XT | `gfx1100` |

## Prerequisites

### 1. Visual Studio Build Tools 2022

The MSVC compiler, CMake, and Ninja are all provided by the Visual Studio Build Tools workload. No separate CMake installation is needed.

Download the bootstrapper from Microsoft and run a silent installation:

```powershell
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile vs_BuildTools.exe

.\vs_BuildTools.exe --quiet --wait --norestart `
    --add Microsoft.VisualStudio.Workload.VCTools `
    --add Microsoft.VisualStudio.Component.VC.CMake.Project `
    --add Microsoft.VisualStudio.Component.Windows11SDK.22621 `
    --includeRecommended
```

```{note}
Pass all arguments as a single string. Passing them as a PowerShell array (`-ArgumentList @(...)`) causes the installer to exit with error code 87 (invalid parameter).
```

Verify the installation:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -products * -format json
```

### 2. ROCm via Python wheels

AMD distributes ROCm for Windows as Python wheels. This is the method used by the official CTranslate2 CI and does not require the AMD HIP SDK installer.

```powershell
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz

rocm-sdk init
```

The `rocm-sdk init` command extracts the development headers and compiler to a local directory. Retrieve the path for later use:

```powershell
$env:ROCM_PATH = python -c "from rocm_sdk._devel import get_devel_root; print(get_devel_root())"
```

This installs AMD Clang (the HIP compiler), HIP headers, `hipblas.dll`, `amdhip64_7.dll`, and the AMDGCN bitcode libraries needed for GPU kernel compilation.

### 3. Intel oneAPI MKL and oneDNN

CTranslate2 uses oneDNN for its CPU backend on Windows. oneDNN in turn requires Intel MKL for optimal performance.

**Install Intel MKL (devel component only):**

Download the offline installer (≈ 2.5 GB) and install only the `mkl.devel` component:

```bat
:: Extract the installer
intel-oneapi-base-toolkit-2025.3.0.372_offline.exe -s -x -f oneapi_extracted

:: Install only the MKL development files
oneapi_extracted\bootstrapper.exe -s --action install ^
    --components=intel.oneapi.win.mkl.devel ^
    --eula=accept ^
    -p=NEED_VS2017_INTEGRATION=0 ^
    -p=NEED_VS2019_INTEGRATION=0
```

**Build oneDNN 3.10.2 from source:**

```bat
curl -L -O https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v3.10.2.tar.gz
```

```python
# Use Python to extract — tar.exe -C is unreliable on Windows
import tarfile
tarfile.open("v3.10.2.tar.gz", "r:gz").extractall(".")
```

Run the following in a **VS Developer Command Prompt** (to make MSVC available):

```bat
cd oneDNN-3.10.2
cmake -DCMAKE_BUILD_TYPE=Release ^
      -DONEDNN_LIBRARY_TYPE=STATIC ^
      -DONEDNN_BUILD_EXAMPLES=OFF ^
      -DONEDNN_BUILD_TESTS=OFF ^
      -DONEDNN_ENABLE_WORKLOAD=INFERENCE ^
      "-DONEDNN_ENABLE_PRIMITIVE=CONVOLUTION;REORDER" ^
      -DONEDNN_BUILD_GRAPH=OFF .
cmake --build . --config Release --target install --parallel
```

```{note}
The install step writes to `C:\Program Files (x86)\oneDNN` and requires administrator privileges. Run the install step from an elevated prompt or via `Start-Process -Verb RunAs`.
```

## Clone the repository

```bash
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
```

If you already cloned without `--recursive`, initialize the submodules explicitly:

```bash
git submodule update --init --recursive
```

The required submodules are `spdlog`, `cpu_features`, `cutlass`, `googletest`, `ruy`, `thrust`, and `cxxopts`. CMake will fail with `add_subdirectory: source directory does not exist` if any are missing.

## Build the C++ library

Create a build script (e.g. `build_rocm.bat`) with all required environment variables:

```bat
@echo off

:: --- ROCm environment ---
set ROCM_PATH=<output of: python -c "from rocm_sdk._devel import get_devel_root; print(get_devel_root())">
set HIP_PLATFORM=amd
set HIP_PATH=%ROCM_PATH%
set HIP_DEVICE_LIB_PATH=%ROCM_PATH%/lib/llvm/amdgcn/bitcode
set HIP_CLANG_ROOT=%ROCM_PATH%/lib/llvm

:: Windows SDK resource compiler (required when using Clang as the C/C++ compiler)
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%

:: --- Paths (use forward slashes for CMake) ---
set CMAKE_EXE=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe
set NINJA_EXE=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe
set RC_EXE=C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe
set INSTALL_PREFIX=C:/path/to/ctranslate2-install

"%CMAKE_EXE%" -GNinja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -S . -B build ^
    -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
    -DCMAKE_C_COMPILER="%ROCM_PATH%/lib/llvm/bin/clang.exe" ^
    -DCMAKE_CXX_COMPILER="%ROCM_PATH%/lib/llvm/bin/clang++.exe" ^
    -DCMAKE_RC_COMPILER="%RC_EXE%" ^
    "-DCMAKE_CXX_FLAGS=-Wno-deprecated-literal-operator" ^
    "-DCMAKE_HIP_FLAGS=-Wno-deprecated-literal-operator" ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib;C:/Program Files (x86)/oneDNN" ^
    -DBUILD_CLI=OFF ^
    -DWITH_DNNL=ON ^
    -DWITH_HIP=ON ^
    "-DCMAKE_HIP_ARCHITECTURES=gfx1100"

"%CMAKE_EXE%" --build build --config Release --parallel
"%CMAKE_EXE%" --install build --config Release
```

Replace `gfx1100` with the architecture of your GPU (see the table above). To target multiple GPUs, separate the values with semicolons: `"gfx1100;gfx1101"`.

```{important}
**Use forward slashes in all CMake paths.** Backslashes are interpreted as escape sequences inside CMake cache strings. A path like `C:\Program Files` becomes invalid (`\P` is not a recognized escape). This applies to all `-D` arguments passed to CMake.
```

```{note}
**CMake cannot locate Ninja or `rc.exe` automatically** when Clang is the compiler. Both must be specified explicitly:

- `-DCMAKE_MAKE_PROGRAM` — full path to `ninja.exe` (bundled with VS Build Tools)
- `-DCMAKE_RC_COMPILER` — full path to `rc.exe` from the Windows SDK

Without `rc.exe`, CMake aborts with: `No CMAKE_RC_COMPILER could be found`.
```

```{note}
**CMake cache must be cleared between configuration attempts.** If you change a compiler path or fix a path format error, delete the entire `build/` directory before re-running CMake. Stale cache entries (especially compiler paths) are not overwritten by new `-D` arguments.
```

A successful configuration ends with:

```
-- HIP Compiler: .../clang++.exe
-- CMAKE_HIP_ARCHITECTURES: gfx1100
-- Configuring done
-- Generating done
-- Build files have been written to: .../build
```

## Build the Python module

Copy the required DLLs into the Python package directory before building:

```powershell
$install = "C:/path/to/ctranslate2-install"
Copy-Item "$install/bin/ctranslate2.dll" python/ctranslate2/
Copy-Item "C:/Program Files (x86)/Intel/oneAPI/2025.3/bin/libiomp5md.dll" python/ctranslate2/
```

Then install the Python package from a **VS Developer Command Prompt**:

```bat
set CTRANSLATE2_ROOT=C:/path/to/ctranslate2-install
set CMAKE_BUILD_PARALLEL_LEVEL=8
cd python
pip install "pybind11==2.11.1" setuptools wheel
pip install . --no-build-isolation
```

## Verify the installation

When using the ROCm build, the AMD runtime DLLs (`amdhip64_7.dll`, `hipblas.dll`) are not on the system PATH by default. Add the ROCm binary directory before importing the module:

```python
import os
from rocm_sdk._devel import get_devel_root

os.add_dll_directory(os.path.join(str(get_devel_root()), "bin"))

import ctranslate2

print(ctranslate2.__version__)
print(ctranslate2.get_supported_compute_types("cuda"))  # cuda = HIP device
print(ctranslate2._ext.get_cuda_device_count())
```

Expected output (RX 7900 XTX):

```
4.7.1
{'float32', 'float16', 'bfloat16', 'int8', 'int8_float16', 'int8_bfloat16', 'int8_float32'}
1
```

```{note}
CTranslate2 uses `Device::CUDA` as a unified enum for both CUDA and HIP backends. When using the Python API, specify `device="cuda"` to run inference on an AMD GPU.
```

## Known limitations

The following features are currently not available when building with `-DWITH_HIP=ON`:

- **Flash Attention** (`-DWITH_FLASH_ATTN=ON`) — mutually exclusive with `WITH_HIP`
- **Tensor parallelism** (`-DWITH_TENSOR_PARALLEL=ON`) — mutually exclusive with `WITH_HIP`
- **AWQ quantization** — GPU kernels are not yet ported to HIP; AWQ models fall back to CPU execution
- **Asynchronous memory allocator** — disabled on Windows/HIP builds due to instability; synchronous allocation is used instead

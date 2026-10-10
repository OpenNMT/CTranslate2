import importlib
import os
import sys


def _register_nvidia_pip_dll_directories(import_module=importlib.import_module):
    """Point Windows DLL search mechanisms at pip-installed nvidia-cublas-cu12/
    nvidia-cudnn-cu12 packages' bin directories, if present.

    The Python wheel does not bundle cuBLAS/cuDNN (see CONTRIBUTING.md,
    "CUDA support in Python wheels"). cuBLAS is loaded with a plain
    Win32 LoadLibraryA() call (src/cuda/cublas_stub.cc), which does NOT
    consult directories registered with os.add_dll_directory() -- that
    only affects loaders that opt into LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
    which a bare LoadLibraryA() does not. Verified empirically: with
    only os.add_dll_directory() registered, LoadLibraryA("cublas64_12.dll")
    still fails; adding the same directory to PATH (part of the legacy
    search order LoadLibraryA does use) is what actually makes it
    resolve. So both are registered here, same as os.add_dll_directory()
    is kept for any other loader in the process that does honor it.

    Users who `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` for CUDA
    execution -- the standard pip-only setup, also used by faster-whisper
    -- get those libraries installed as sibling packages, but nothing
    here points either mechanism at them, so on Windows they hit "Could
    not locate cudnn_cnn_infer64_8.dll" (#1915) / the same class of
    problem #1826 reports for Linux's LD_LIBRARY_PATH.

    Only called on sys.platform == "win32" (see below); pulled out to a
    plain function -- rather than left inline in the platform-guarded
    module body -- so it can be unit-tested on any host OS by injecting
    a fake `import_module` and monkeypatching os.add_dll_directory/PATH,
    without needing a real Windows+CUDA machine.
    """
    for nvidia_package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            nvidia_module = import_module(nvidia_package)
        except ImportError:
            continue
        for base in nvidia_module.__path__:
            bin_dir = os.path.join(base, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]


if sys.platform == "win32":
    import ctypes
    import glob

    from importlib.resources import files

    module_name = sys.modules[__name__].__name__
    package_dir = str(files(module_name))

    try:
        os.add_dll_directory(package_dir)
        os.add_dll_directory(f"{package_dir}/../_rocm_sdk_core/bin")
        os.add_dll_directory(f"{package_dir}/../_rocm_sdk_libraries_custom/bin")
    except (FileNotFoundError, OSError):
        pass

    _register_nvidia_pip_dll_directories()

    for library in glob.glob(os.path.join(package_dir, "*.dll")):
        ctypes.CDLL(library)

try:
    from ctranslate2._ext import (
        AsyncGenerationResult,
        AsyncScoringResult,
        AsyncTranslationResult,
        DataType,
        Device,
        Encoder,
        EncoderForwardOutput,
        ExecutionStats,
        GenerationResult,
        GenerationStepResult,
        Generator,
        MpiInfo,
        ScoringResult,
        StorageView,
        TranslationResult,
        Translator,
        contains_model,
        get_cuda_device_count,
        get_supported_compute_types,
        set_random_seed,
    )
    from ctranslate2.extensions import register_extensions
    from ctranslate2.logging import get_log_level, set_log_level

    register_extensions()
    del register_extensions
except ImportError as e:
    # Allow using the Python package without the compiled extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from ctranslate2 import models  # noqa: E402
from ctranslate2.version import __version__  # noqa: E402

# converters and specs import torch (and, for converters, transformers) at module level.
# Those dependencies are only needed to convert models, not to run inference, so import
# these submodules on first use to keep "import ctranslate2" free of them.
_LAZY_SUBMODULES = ("converters", "specs")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))


# A wildcard import resolves ``__all__`` when it is defined and the module globals
# otherwise, so without this the lazy submodules would silently drop out of
# ``from ctranslate2 import *``. Deriving the list keeps the wildcard surface identical
# to what it was before they became lazy; a wildcard import asks for everything, so
# resolving them here is expected.
__all__ = sorted(
    [name for name in globals() if not name.startswith("_")] + list(_LAZY_SUBMODULES)
)

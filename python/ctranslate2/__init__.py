import sys

if sys.platform == "win32":
    import ctypes
    import glob
    import os

    from importlib.resources import files

    module_name = sys.modules[__name__].__name__
    package_dir = str(files(module_name))

    try:
        os.add_dll_directory(package_dir)
        os.add_dll_directory(f"{package_dir}/../_rocm_sdk_core/bin")
        os.add_dll_directory(f"{package_dir}/../_rocm_sdk_libraries_custom/bin")
    except (FileNotFoundError, OSError):
        pass

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

from ctranslate2 import models
from ctranslate2.version import __version__

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

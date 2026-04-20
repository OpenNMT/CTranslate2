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

from ctranslate2 import converters, models, specs
from ctranslate2.version import __version__

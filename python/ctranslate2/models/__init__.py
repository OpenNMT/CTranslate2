"""A collection of models which don't fit in the generic classes :class:`ctranslate2.Translator`
and :class:`ctranslate2.Generator`.
"""

try:
    from ctranslate2._ext import (
        Whisper,
        WhisperGenerationResult,
        WhisperGenerationResultAsync,
        Wav2Vec2,
    )
except ImportError as e:
    # Allow using the Python package without the compiled extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from typing import List, Optional, Tuple

import numpy as np

from ctranslate2.specs import common_spec, model_spec, transformer_spec


class MoonshineConfig(model_spec.ModelConfig):
    """Configuration for the Moonshine model."""

    def __init__(
        self,
        suppress_ids: Optional[List[int]] = None,
        suppress_ids_begin: Optional[List[int]] = None,
        lang_ids: Optional[List[int]] = None,
        alignment_heads: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(
            suppress_ids=suppress_ids,
            suppress_ids_begin=suppress_ids_begin,
            lang_ids=lang_ids,
            alignment_heads=alignment_heads,
        )


class MoonshineSpec(model_spec.LanguageModelSpec):
    """Describes a Whisper model."""

    def __init__(
        self,
        num_encoder_layers,
        num_encoder_heads,
        num_decoder_layers,
        num_decoder_heads,
    ):
        """Initializes the model specification.

        Args:
          num_encoder_layers: The number of encoder layers.
          num_encoder_heads: The number of encoder attention heads.
          num_decoder_layers: The number of decoder layers.
          num_decoder_heads: The number of decoder attention heads.
        """
        super().__init__()
        self.preprocessor = AudioPreprocessSpec()
        self.encoder = transformer_spec.TransformerEncoderSpec(
            num_layers=num_encoder_layers,
            num_heads=num_encoder_heads,
            activation=common_spec.Activation.GELU,
            num_source_embeddings=0,
            rotary_dim=32,
        )
        self.decoder = transformer_spec.TransformerDecoderSpec(
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            activation=common_spec.Activation.SWISH,
            ffn_glu=True,
            with_encoder_attention=True,
            project_in_out=False,
            rotary_dim=32,
        )
        self.decoder.scale_embeddings = False

    @property
    def name(self):
        return "MoonshineSpec"

    @property
    def revision(self):
        return 0

    def get_default_config(self):
        return MoonshineConfig()

    def get_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class AudioPreprocessSpec(model_spec.LayerSpec):
    def __init__(self):
        self.conv1 = common_spec.Conv1DSpec()
        self.layernorm = common_spec.LayerNormSpec()
        self.conv2 = common_spec.Conv1DSpec()
        self.conv3 = common_spec.Conv1DSpec()

from typing import List, Optional, Tuple

import numpy as np

from ctranslate2.specs import common_spec, model_spec, transformer_spec


class WavLMConfig(model_spec.ModelConfig):
    """Configuration for the WavLM model."""

    def __init__(self, layer_norm_epsilon: float = None, **kwargs):
        super().__init__(layer_norm_epsilon=layer_norm_epsilon, **kwargs)


class WavLMSpec(model_spec.LanguageModelSpec):
    def __init__(
        self,
        feat_layers,
        num_layers,
        num_heads,
        return_hidden,
    ):
        super().__init__()
        self.vocab_size = np.dtype("int16").type(0)
        self.encoder = WavLMEncoderSpec(
            feat_layers,
            num_layers,
            num_heads,
            return_hidden,
        )

    @property
    def name(self):
        return "WavLMSpec"

    @property
    def revision(self):
        return 3

    def get_default_config(self):
        return WavLMConfig()

    def get_vocabulary_size(self):
        return 0


class WavLMLayerNormConvLayer(model_spec.LayerSpec):
    def __init__(self):
        self.conv = common_spec.Conv1DSpec()
        self.layer_norm = common_spec.LayerNormSpec()


class WavLMPosEmbedConvLayer(model_spec.LayerSpec):
    def __init__(self):
        self.conv = common_spec.Conv1DSpec()


class WavLMEncoderSpec(model_spec.LayerSpec):
    def __init__(self, feat_layers, num_layers, num_heads, return_hidden):
        self.num_heads = np.dtype("int16").type(num_heads)
        self.feat_layer0 = WavLMLayerNormConvLayer()
        self.feat_layer = [WavLMLayerNormConvLayer() for i in range(feat_layers - 1)]
        self.fp_layer_norm = common_spec.LayerNormSpec()
        self.fp_projection = common_spec.LinearSpec()
        self.pos_conv_embed = WavLMPosEmbedConvLayer()
        self.layer_norm = common_spec.LayerNormSpec()
        self.layer = [
            transformer_spec.TransformerEncoderLayerSpec(gated_relative_attention_bias=True,
                                                         relative_attention_bias=(i == 0)) for i in range(num_layers)
        ]
        # if not return_hidden:
        #     self.lm_head = common_spec.LinearSpec()

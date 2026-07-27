import numpy as np

from ctranslate2.specs import attention_spec, common_spec, model_spec


class Wav2Vec2BertConfig(model_spec.ModelConfig):
    """Configuration for the Wav2Vec2Bert model."""

    def __init__(self):
        return


class Wav2Vec2BertSpec(model_spec.LanguageModelSpec):
    def __init__(
        self,
        num_hidden_layers,
        num_adapter_layers,
        vocab_size,
        return_hidden,
    ):
        super().__init__()
        self.vocab_size = np.dtype("int16").type(vocab_size)
        self.encoder = Wav2Vec2BertEncoderSpec(
            num_adapter_layers,
            num_hidden_layers,
            return_hidden,
        )

    @property
    def name(self):
        return "Wav2Vec2BertSpec"

    @property
    def revision(self):
        return 1

    def get_default_config(self):
        return Wav2Vec2BertConfig()

    def get_vocabulary_size(self):
        return int(self.vocab_size.numpy())


class Wav2Vec2BertFeedForwardSpec(model_spec.LayerSpec):
    def __init__(self, glu=False, rms_norm=False):
        self.linear_0 = common_spec.LinearSpec()
        self.linear_1 = common_spec.LinearSpec()
        if glu:
            self.linear_0_noact = common_spec.LinearSpec()


class EncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.enc_ffn1_layer_norm = common_spec.LayerNormSpec()
        self.enc_ffn1 = Wav2Vec2BertFeedForwardSpec()
        self.enc_attn_layer_norm = common_spec.LayerNormSpec()
        self.enc_attn = attention_spec.MultiHeadAttentionSpec(
            self_attention=True,
            relative_asymmetric_position=True,
        )
        del self.enc_attn.layer_norm
        self.enc_conv_layer_norm = common_spec.LayerNormSpec()
        self.enc_conv_pointwise_conv1 = common_spec.Conv1DSpec()
        del self.enc_conv_pointwise_conv1.bias
        self.enc_conv_depthwise_conv = common_spec.Conv1DSpec()
        del self.enc_conv_depthwise_conv.bias
        self.enc_conv_depthwise_layer_norm = common_spec.LayerNormSpec()
        self.enc_conv_pointwise_conv2 = common_spec.Conv1DSpec()
        del self.enc_conv_pointwise_conv2.bias
        self.enc_ffn2_layer_norm = common_spec.LayerNormSpec()
        self.enc_ffn2 = Wav2Vec2BertFeedForwardSpec()
        self.enc_final_layer_norm = common_spec.LayerNormSpec()


class AdapterSpec(model_spec.LayerSpec):
    def __init__(self):
        self.adpt_residual_layer_norm = common_spec.LayerNormSpec()
        self.adpt_residual_conv = common_spec.Conv1DSpec()
        self.adpt_attn_layer_norm = common_spec.LayerNormSpec()
        self.adpt_attn_conv = common_spec.Conv1DSpec()
        self.adpt_attn_layer = attention_spec.MultiHeadAttentionSpec(
            self_attention=True,
            relative_asymmetric_position=False,
        )
        del self.adpt_attn_layer.layer_norm
        self.adpt_ffn_layer_norm = common_spec.LayerNormSpec()
        self.adpt_ffn = Wav2Vec2BertFeedForwardSpec()


class Wav2Vec2BertEncoderSpec(model_spec.LayerSpec):
    def __init__(self, num_hidden_layers, num_adapter_layers, return_hidden):
        self.fp_layer_norm = common_spec.LayerNormSpec()
        self.fp_projection = common_spec.LinearSpec()
        self.encoder_layers = [EncoderSpec() for _ in range(num_hidden_layers)]
        self.adapter_layers = [AdapterSpec() for _ in range(num_adapter_layers)]
        if not return_hidden:
            self.lm_head = common_spec.LinearSpec()

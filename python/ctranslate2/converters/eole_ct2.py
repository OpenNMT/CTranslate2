import argparse

from eole.config.run import PredictConfig
from eole.constants import PositionEncodingType
from eole.inputters.inputter import vocabs_to_dict
from eole.models.model import get_model_class

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, transformer_spec

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "fast_gelu": common_spec.Activation.GELUTanh,
    "relu": common_spec.Activation.RELU,
    "gated-silu": common_spec.Activation.SWISH,
}


def _get_model_spec_seq2seq(
    config, variables, src_vocabs, tgt_vocabs, num_source_embeddings
):
    """Creates a model specification from the model config."""
    with_relative_position = (
        getattr(config.embeddings, "position_encoding_type", None)
        == PositionEncodingType.Relative
    )
    with_rotary = (
        getattr(config.embeddings, "position_encoding_type", None)
        == PositionEncodingType.Rotary
    )
    if with_rotary:
        raise ValueError(
            "Rotary embeddings are not supported yet for encoder/decoder models"
        )
    with_alibi = (
        getattr(config.embeddings, "position_encoding_type", None)
        == PositionEncodingType.Alibi
    )
    if with_alibi:
        raise ValueError("Alibi is not supported yet for encoder/decoder models")
    activation_fn = getattr(config, "mlp_activation_fn", "relu")

    # Return the first head of the last layer unless the model was trained with alignments.
    if getattr(config.decoder, "lambda_align", 0) == 0:
        alignment_layer = -1
        alignment_heads = 1
    else:
        alignment_layer = config.decoder.alignment_layer
        alignment_heads = config.decoder.alignment_heads

    num_heads = getattr(config.decoder, "heads", 8)
    # num_kv = getattr(config.decoder, "heads_kv", 0)
    # if num_kv == num_heads or num_kv == 0:
    #    num_kv = None
    # rotary_dim = 0 if with_rotary else None
    # rotary_interleave = getattr(config.rope_config, "rotary_interleave", True)
    ffn_glu = activation_fn == "gated-silu"
    sliding_window = getattr(config, "sliding_window", 0)
    if sliding_window != 0:
        raise ValueError(
            "Sliding window is not suported yet for encoder/decoder models"
        )

    model_spec = transformer_spec.TransformerSpec.from_config(
        (config.encoder.layers, config.decoder.layers),
        num_heads,
        with_relative_position=with_relative_position,
        # alibi=with_alibi,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        ffn_glu=ffn_glu,
        rms_norm=config.layer_norm == "rms",
        # rotary_dim=rotary_dim,
        # rotary_interleave=rotary_interleave,
        # num_heads_kv=num_kv,
        # sliding_window=sliding_window,
        alignment_layer=alignment_layer,
        alignment_heads=alignment_heads,
        num_source_embeddings=num_source_embeddings,
        # multi_query_attention=getattr(opt, "multiquery", False),
    )

    set_transformer_spec(model_spec, variables)
    for src_vocab in src_vocabs:
        model_spec.register_source_vocabulary(src_vocab)
    for tgt_vocab in tgt_vocabs:
        model_spec.register_target_vocabulary(tgt_vocab)

    return model_spec


def _get_model_spec_lm(
    config, variables, src_vocabs, tgt_vocabs, num_source_embeddings
):
    """Creates a model specification from the model config."""
    with_relative_position = (
        getattr(config.embeddings, "position_encoding_type", None)
        == PositionEncodingType.Relative
    )
    with_rotary = (
        getattr(config.embeddings, "position_encoding_type", None)
        == PositionEncodingType.Rotary
    )
    with_alibi = (
        getattr(config.embeddings, "position_encoding_type", None)
        == PositionEncodingType.Alibi
    )
    activation_fn = getattr(config, "mlp_activation_fn", "relu")
    num_heads = getattr(config.decoder, "heads", 8)
    num_kv = getattr(config.decoder, "heads_kv", 0)
    if num_kv == num_heads or num_kv == 0:
        num_kv = None
    rotary_dim = 0 if with_rotary else None
    rotary_interleave = getattr(config.rope_config, "rotary_interleave", True)
    ffn_glu = activation_fn == "gated-silu"
    sliding_window = getattr(config, "sliding_window", 0)

    model_spec = transformer_spec.TransformerDecoderModelSpec.from_config(
        config.decoder.layers,
        num_heads,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        ffn_glu=ffn_glu,
        with_relative_position=with_relative_position,
        alibi=with_alibi,
        rms_norm=config.layer_norm == "rms",
        rotary_dim=rotary_dim,
        rotary_interleave=rotary_interleave,
        num_heads_kv=num_kv,
        sliding_window=sliding_window,
        # multi_query_attention=getattr(opt, "multiquery", False),
    )

    set_transformer_decoder(
        model_spec.decoder,
        variables,
        with_encoder_attention=False,
    )

    for tgt_vocab in tgt_vocabs:
        model_spec.register_vocabulary(tgt_vocab)

    return model_spec


def get_vocabs(vocab):
    src_vocabs = [vocab["src"]]
    tgt_vocabs = [vocab["tgt"]]
    return src_vocabs, tgt_vocabs


class EoleConverter(Converter):
    """Converts models generated by OpenNMT-py."""

    def __init__(self, model_path: str):
        """Initializes the OpenNMT-py converter.

        Arguments:
          model_path: Path to the OpenNMT-py PyTorch model (.pt file).
        """
        self._model_path = model_path

    def _load(self):
        import torch

        config = PredictConfig(model_path=self._model_path, src="dummy")

        model_class = get_model_class(config.model)
        model, vocabs, model_config = model_class.for_inference(config)
        vocabs_dict = vocabs_to_dict(vocabs)

        config.model = model_config
        src_vocabs, tgt_vocabs = get_vocabs(vocabs_dict)

        if config.model.decoder.decoder_type == "transformer_lm":
            spec = _get_model_spec_lm(
                config.model,
                model.state_dict(),
                src_vocabs,
                tgt_vocabs,
                num_source_embeddings=len(src_vocabs),
            )
        else:
            spec = _get_model_spec_seq2seq(
                config.model,
                model.state_dict(),
                src_vocabs,
                tgt_vocabs,
                num_source_embeddings=len(src_vocabs),
            )
            spec.config.decoder_start_token = vocabs["decoder_start_token"]

        spec.config.bos_token = vocabs["specials"]["bos_token"]
        spec.config.eos_token = vocabs["specials"]["eos_token"]
        spec.config.unk_token = vocabs["specials"]["unk_token"]
        spec.config.layer_norm_epsilon = getattr(config, "norm_eps", 1e-6)

        return spec


def set_transformer_spec(spec, variables):
    set_transformer_encoder(spec.encoder, variables)
    set_transformer_decoder(spec.decoder, variables)


def set_transformer_encoder(spec, variables):
    set_input_layers(spec, variables, "src_emb")
    set_layer_norm(spec.layer_norm, variables, "encoder.layer_norm")
    for i, layer in enumerate(spec.layer):
        set_transformer_encoder_layer(
            layer, variables, "encoder.transformer_layers.%d" % i
        )


def set_transformer_decoder(spec, variables, with_encoder_attention=True):
    set_input_layers(spec, variables, "tgt_emb")
    set_layer_norm(spec.layer_norm, variables, "decoder.layer_norm")
    for i, layer in enumerate(spec.layer):
        set_transformer_decoder_layer(
            layer,
            variables,
            "decoder.transformer_layers.%d" % i,
            with_encoder_attention=with_encoder_attention,
        )

    set_linear(spec.projection, variables, "generator")


def set_input_layers(spec, variables, scope):
    if hasattr(spec, "position_encodings"):
        set_position_encodings(
            spec.position_encodings,
            variables,
            "%s.pe" % scope,
        )
    else:
        spec.scale_embeddings = False

    embeddings_specs = spec.embeddings
    # encoder embeddings are stored in a list(onmt/ct2 legacy with features)
    if isinstance(embeddings_specs, list):
        embeddings_specs = embeddings_specs[0]
    set_embeddings(embeddings_specs, variables, "%s.embeddings" % scope)


def set_transformer_encoder_layer(spec, variables, scope):
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attn" % scope,
        self_attention=True,
    )
    set_layer_norm(
        spec.self_attention.layer_norm, variables, "%s.input_layernorm" % scope
    )
    set_layer_norm(
        spec.ffn.layer_norm, variables, "%s.post_attention_layernorm" % scope
    )
    set_ffn(spec.ffn, variables, "%s.mlp" % scope)


def set_transformer_decoder_layer(spec, variables, scope, with_encoder_attention=True):
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attn" % scope,
        self_attention=True,
    )
    set_layer_norm(
        spec.self_attention.layer_norm, variables, "%s.input_layernorm" % scope
    )
    if with_encoder_attention:
        set_multi_head_attention(spec.attention, variables, "%s.context_attn" % scope)
        set_layer_norm(
            spec.attention.layer_norm, variables, "%s.precontext_layernorm" % scope
        )
    set_layer_norm(
        spec.ffn.layer_norm, variables, "%s.post_attention_layernorm" % scope
    )
    set_ffn(spec.ffn, variables, "%s.mlp" % scope)


def set_ffn(spec, variables, scope):
    set_linear(spec.linear_0, variables, "%s.gate_up_proj" % scope)
    set_linear(spec.linear_1, variables, "%s.down_proj" % scope)
    if hasattr(spec, "linear_0_noact"):
        set_linear(spec.linear_0_noact, variables, "%s.up_proj" % scope)


def set_multi_head_attention(spec, variables, scope, self_attention=False):
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], variables, "%s.linear_query" % scope)
        set_linear(split_layers[1], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[2], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        set_linear(spec.linear[0], variables, "%s.linear_query" % scope)
        split_layers = [common_spec.LinearSpec() for _ in range(2)]
        set_linear(split_layers[0], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[1], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[1], split_layers)
    set_linear(spec.linear[-1], variables, "%s.final_linear" % scope)
    if hasattr(spec, "relative_position_keys"):
        spec.relative_position_keys = _get_variable(
            variables, "%s.relative_positions_embeddings.weight" % scope
        )
        spec.relative_position_values = spec.relative_position_keys


def set_layer_norm(spec, variables, scope):
    try:
        spec.gamma = _get_variable(variables, "%s.weight" % scope)
    except KeyError:
        # Compatibility with older models using a custom LayerNorm module.
        spec.gamma = _get_variable(variables, "%s.a_2" % scope)
        spec.beta = _get_variable(variables, "%s.b_2" % scope)
    try:
        spec.beta = _get_variable(variables, "%s.bias" % scope)
    except KeyError:
        pass


def set_linear(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)
    bias = variables.get("%s.bias" % scope)
    if bias is not None:
        spec.bias = bias


def set_embeddings(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)


def set_position_encodings(spec, variables, scope):
    spec.encodings = _get_variable(variables, "%s.pe" % scope).squeeze()


def _get_variable(variables, name):
    return variables[name]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", required=True, help="Model path.")
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    EoleConverter(args.model_path).convert_from_args(args)


if __name__ == "__main__":
    main()

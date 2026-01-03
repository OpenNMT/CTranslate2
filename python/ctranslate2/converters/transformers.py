import abc
import argparse
import gc
import itertools
import os

from typing import List, Optional

import numpy as np

try:
    import huggingface_hub
    import torch
    import transformers
except ImportError:
    pass

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import (
    attention_spec,
    common_spec,
    model_spec,
    transformer_spec,
    wav2vec2_spec,
    wav2vec2bert_spec,
    whisper_spec,
)

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "gelu_fast": common_spec.Activation.GELUTanh,
    "gelu_new": common_spec.Activation.GELUTanh,
    "gelu_python": common_spec.Activation.GELU,
    "gelu_pytorch_tanh": common_spec.Activation.GELUTanh,
    "quick_gelu": common_spec.Activation.GELUSigmoid,
    "relu": common_spec.Activation.RELU,
    "silu": common_spec.Activation.SWISH,
    "swish": common_spec.Activation.SWISH,
}

_SUPPORTED_ROPE_SCALING = {
    "linear": attention_spec.RotaryScalingType.Linear,
    "su": attention_spec.RotaryScalingType.Su,
    "llama3": attention_spec.RotaryScalingType.Llama3,
    "longrope": attention_spec.RotaryScalingType.Su,
}

_SUPPORTED_QUANTIZATION = {
    "gemm": common_spec.Quantization.AWQ_GEMM,
    "gemv": common_spec.Quantization.AWQ_GEMV,
}

_MODEL_LOADERS = {}


def register_loader(config_name):
    """Registers a model loader for this configuration name."""

    def decorator(cls):
        _MODEL_LOADERS[config_name] = cls()
        return cls

    return decorator


class TransformersConverter(Converter):
    """Converts models from Hugging Face Transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        activation_scales: Optional[str] = None,
        copy_files: Optional[List[str]] = None,
        load_as_float16: bool = False,
        revision: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        trust_remote_code: bool = False,
    ):
        """Initializes the converter.

        Arguments:
          model_name_or_path: Name of the pretrained model to download, or path to the
            directory containing the pretrained model.
          activation_scales: Path to the pre-computed activation scales. Models may
            use them to rescale some weights to smooth the intermediate activations
            and improve the quantization accuracy. See
            https://github.com/mit-han-lab/smoothquant.
          copy_files: List of filenames to copy from the Hugging Face model to the
            converted model directory.
          load_as_float16: Load the model weights as float16. More precisely, the model
            will be loaded with ``from_pretrained(..., dtype=torch.float16)``.
          revision: Revision of the model to download from the Hugging Face Hub.
          low_cpu_mem_usage: Enable the flag ``low_cpu_mem_usage`` when loading the model
            with ``from_pretrained``.
          trust_remote_code: Allow converting models using custom code.
        """
        self._model_name_or_path = model_name_or_path
        self._activation_scales = activation_scales
        self._copy_files = copy_files
        self._load_as_float16 = load_as_float16
        self._revision = revision
        self._low_cpu_mem_usage = low_cpu_mem_usage
        self._trust_remote_code = trust_remote_code

    def _load(self):
        with torch.no_grad():
            config = transformers.AutoConfig.from_pretrained(
                self._model_name_or_path, trust_remote_code=self._trust_remote_code
            )

            config_name = config.__class__.__name__
            loader = _MODEL_LOADERS.get(config_name)

            if loader is None:
                raise ValueError(
                    "No conversion is registered for the model configuration %s "
                    "(supported configurations are: %s)"
                    % (config_name, ", ".join(sorted(_MODEL_LOADERS.keys())))
                )

            model_class = getattr(transformers, loader.architecture_name)
            tokenizer_class = transformers.AutoTokenizer

            kwargs = {
                "dtype": (
                    torch.float16
                    if self._load_as_float16
                    else getattr(config, "dtype", None)
                    or getattr(config, "torch_dtype", None)
                )
            }

            if self._revision:
                kwargs["revision"] = self._revision
            if self._low_cpu_mem_usage:
                kwargs["low_cpu_mem_usage"] = self._low_cpu_mem_usage
            if self._trust_remote_code:
                kwargs["trust_remote_code"] = self._trust_remote_code

            model = self.load_model(model_class, self._model_name_or_path, **kwargs)

            tokenizer_kwargs = {}
            if self._trust_remote_code:
                tokenizer_kwargs["trust_remote_code"] = self._trust_remote_code

            tokenizer = self.load_tokenizer(
                tokenizer_class, self._model_name_or_path, **tokenizer_kwargs
            )

            spec = loader(model, tokenizer)

            if self._activation_scales:
                activation_scales = torch.load(
                    self._activation_scales, map_location="cpu"
                )
                loader.smooth_activation(spec, activation_scales)

            if self._copy_files:
                for filename in self._copy_files:
                    spec.register_file(self.get_model_file(filename))

            return spec

    def load_model(self, model_class, model_name_or_path, **kwargs):
        return model_class.from_pretrained(model_name_or_path, **kwargs)

    def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
        return tokenizer_class.from_pretrained(model_name_or_path, **kwargs)

    def get_model_file(self, filename):
        if os.path.isdir(self._model_name_or_path):
            path = os.path.join(self._model_name_or_path, filename)
        else:
            try:
                path = huggingface_hub.hf_hub_download(
                    repo_id=self._model_name_or_path, filename=filename
                )
            except huggingface_hub.utils.EntryNotFoundError:
                path = None

        if path is None or not os.path.isfile(path):
            raise ValueError(
                "File %s does not exist in model %s"
                % (filename, self._model_name_or_path)
            )

        return path


class ModelLoader(abc.ABC):
    """Base class for loading Transformers models into a CTranslate2 model specification."""

    @property
    def architecture_name(self):
        return None

    @abc.abstractmethod
    def get_model_spec(self, model):
        raise NotImplementedError()

    def __call__(self, model, tokenizer):
        spec = self.get_model_spec(model)
        self.set_config(spec.config, model, tokenizer)

        tokens = self.get_vocabulary(model, tokenizer)
        self.set_vocabulary(spec, tokens)

        return spec

    def get_vocabulary(self, model, tokenizer):
        return [
            token
            for token, _ in sorted(
                tokenizer.get_vocab().items(), key=lambda item: item[1]
            )
        ]

    def set_vocabulary(self, spec, tokens):
        pass

    def set_config(self, config, model, tokenizer):
        pass

    def set_layer_norm(self, spec, module):
        spec.gamma = module.weight
        spec.beta = module.bias

    def set_linear(self, spec, module, quant_type=common_spec.Quantization.CT2):
        if quant_type == common_spec.Quantization.CT2:
            spec.weight = module.weight
        else:
            spec.weight = module.qweight
            spec.weight_scale = module.scales
            spec.weight_zero = module.qzeros

        if isinstance(module, transformers.Conv1D):
            spec.weight = spec.weight.transpose(0, 1)
        if hasattr(module, "bias") and module.bias is not None:
            spec.bias = module.bias

    def set_embeddings(self, spec, module):
        spec.weight = module.weight

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weight
        offset = getattr(module, "offset", 0)
        if offset > 0:
            spec.encodings = spec.encodings[offset:]

    def smooth_activation(self, spec, activation_scales):
        raise NotImplementedError(
            "No activation smoothing logic is defined for this model"
        )


@register_loader("BartConfig")
class BartLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "BartForConditionalGeneration"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerSpec.from_config(
            (model.config.encoder_layers, model.config.decoder_layers),
            model.config.encoder_attention_heads,
            pre_norm=model.config.normalize_before,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            layernorm_embedding=getattr(model.config, "normalize_embedding", True),
        )

        self.set_encoder(spec.encoder, model.model.encoder)
        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.lm_head)

        final_logits_bias = getattr(model, "final_logits_bias", None)
        if final_logits_bias is not None and final_logits_bias.nonzero().numel() != 0:
            spec.decoder.projection.bias = final_logits_bias.squeeze()

        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]
        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_source_vocabulary(tokens)
        spec.register_target_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.decoder_start_token = tokenizer.convert_ids_to_tokens(
            model.config.decoder_start_token_id
        )

    def set_encoder(self, spec, encoder):
        self.set_common_layers(spec, encoder)

        for layer_spec, layer in zip(spec.layer, encoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_decoder(self, spec, decoder):
        self.set_common_layers(spec, decoder)

        for layer_spec, layer in zip(spec.layer, decoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            if hasattr(layer, "encoder_attn"):
                self.set_attention(
                    layer_spec.attention,
                    layer.encoder_attn,
                    self_attention=False,
                )
                self.set_layer_norm(
                    layer_spec.attention.layer_norm,
                    layer.encoder_attn_layer_norm,
                )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_attention(self, spec, attention, self_attention=False):
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], attention.q_proj)
        self.set_linear(split_layers[1], attention.k_proj)
        self.set_linear(split_layers[2], attention.v_proj)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], attention.out_proj)

    def set_common_layers(self, spec, module):
        import math

        if not hasattr(module, "embed_scale"):
            embed_scale = (
                math.sqrt(module.config.d_model)
                if module.config.scale_embedding
                else 1.0
            )
        else:
            embed_scale = module.embed_scale
        spec.scale_embeddings = embed_scale
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_embeddings(
            (
                spec.embeddings[0]
                if isinstance(spec.embeddings, list)
                else spec.embeddings
            ),
            module.embed_tokens,
        )

        if hasattr(module, "layer_norm"):
            self.set_layer_norm(spec.layer_norm, module.layer_norm)
        if hasattr(module, "layernorm_embedding"):
            self.set_layer_norm(spec.layernorm_embedding, module.layernorm_embedding)


@register_loader("MarianConfig")
class MarianMTLoader(BartLoader):
    @property
    def architecture_name(self):
        return "MarianMTModel"

    def get_model_spec(self, model):
        model.config.normalize_before = False
        model.config.normalize_embedding = False
        spec = super().get_model_spec(model)
        self._remove_pad_weights(spec)
        return spec

    def set_config(self, config, model, tokenizer):
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

        # The decoder start token can be any token because the decoder always starts
        # from a zero embedding.
        config.decoder_start_token = tokenizer.eos_token

    def set_decoder(self, spec, decoder):
        spec.start_from_zero_embedding = True
        super().set_decoder(spec, decoder)

    def get_vocabulary(self, model, tokenizer):
        # The <pad> token is added by Transformers to start the decoder from a zero embedding,
        # but we already have a dedicated option "start_from_zero_embedding". We remove this token
        # to match the original Marian vocabulary and prevent this token from being generated.
        tokens = super().get_vocabulary(model, tokenizer)
        if tokens[-1] == "<pad>":
            tokens.pop()
        return tokens

    def _remove_pad_weights(self, spec):
        vocab_specs = [
            spec.encoder.embeddings[0],
            spec.decoder.embeddings,
            spec.decoder.projection,
        ]

        # Weights may be shared so we check against the expected size to prevent
        # updating the same weight multiple times.
        new_vocab_size = vocab_specs[0].weight.shape[0] - 1

        for vocab_spec in vocab_specs:
            if vocab_spec.weight.shape[0] == new_vocab_size + 1:
                vocab_spec.weight = vocab_spec.weight[:-1]
            if (
                isinstance(vocab_spec, common_spec.LinearSpec)
                and vocab_spec.has_bias()
                and vocab_spec.bias.shape[0] == new_vocab_size + 1
            ):
                vocab_spec.bias = vocab_spec.bias[:-1]


@register_loader("M2M100Config")
class M2M100Loader(BartLoader):
    @property
    def architecture_name(self):
        return "M2M100ForConditionalGeneration"

    def get_model_spec(self, model):
        model.config.normalize_before = True
        model.config.normalize_embedding = False
        return super().get_model_spec(model)

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weights[module.offset :]

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        # Workaround for issue https://github.com/OpenNMT/CTranslate2/issues/1039.
        if tokens[-1] == tokenizer.unk_token:
            tokens.insert(tokenizer.unk_token_id, tokens.pop())

        for token in tokenizer.additional_special_tokens:
            if token not in tokens:
                tokens.append(token)

        num_madeup_words = getattr(
            tokenizer, "num_madeup_words", model.config.vocab_size - len(tokens)
        )
        if num_madeup_words > 0:
            tokens += ["madeupword%d" % i for i in range(num_madeup_words)]

        return tokens


@register_loader("MBartConfig")
class MBartLoader(BartLoader):
    @property
    def architecture_name(self):
        return "MBartForConditionalGeneration"

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

        # MBart-25 passes the language code as the decoder start token.
        if model.config.tokenizer_class in ("MBartTokenizer", None):
            config.decoder_start_token = None
        else:
            config.decoder_start_token = tokenizer.eos_token


@register_loader("PegasusConfig")
class PegasusLoader(BartLoader):
    @property
    def architecture_name(self):
        return "PegasusForConditionalGeneration"

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.pad_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.decoder_start_token = tokenizer.pad_token


@register_loader("OPTConfig")
class OPTLoader(BartLoader):
    @property
    def architecture_name(self):
        return "OPTForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=model.config.do_layer_norm_before,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            project_in_out=model.config.word_embed_proj_dim != model.config.hidden_size,
        )

        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def smooth_activation(self, spec, activation_scales):
        for i, layer in enumerate(spec.decoder.layer):
            layer_scope = "model.decoder.layers.%d" % i

            utils.smooth_activation(
                layer.self_attention.layer_norm,
                layer.self_attention.linear[0],
                activation_scales["%s.self_attn.q_proj" % layer_scope],
            )

            utils.smooth_activation(
                layer.ffn.layer_norm,
                layer.ffn.linear_0,
                activation_scales["%s.fc1" % layer_scope],
            )

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, decoder):
        super().set_decoder(spec, decoder)

        if decoder.project_in is not None:
            self.set_linear(spec.project_in, decoder.project_in)
        if decoder.project_out is not None:
            self.set_linear(spec.project_out, decoder.project_out)
        if decoder.final_layer_norm is not None:
            self.set_layer_norm(spec.layer_norm, decoder.final_layer_norm)

    def set_common_layers(self, spec, module):
        spec.scale_embeddings = False
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_embeddings(spec.embeddings, module.embed_tokens)

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        i = 0
        while len(tokens) % 8 != 0:
            symbol = "madeupword{:04d}".format(i)
            if symbol not in tokens:
                tokens.append(symbol)
            i += 1

        return tokens


@register_loader("GPTBigCodeConfig")
class GPTBigCodeMHALoader(ModelLoader):
    @property
    def architecture_name(self):
        return "GPTBigCodeForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            multi_query_attention=True,
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.wte)
        self.set_position_encodings(spec.position_encodings, module.wpe)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(layer_spec.self_attention.layer_norm, layer.ln_1)
            self.set_linear(layer_spec.self_attention.linear[0], layer.attn.c_attn)
            self.set_linear(layer_spec.self_attention.linear[1], layer.attn.c_proj)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.ln_2)
            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.c_fc)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.c_proj)


@register_loader("GPT2Config")
class GPT2Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "GPT2LMHeadModel"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.wte)
        self.set_position_encodings(spec.position_encodings, module.wpe)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(layer_spec.self_attention.layer_norm, layer.ln_1)
            self.set_linear(layer_spec.self_attention.linear[0], layer.attn.c_attn)
            self.set_linear(layer_spec.self_attention.linear[1], layer.attn.c_proj)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.ln_2)
            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.c_fc)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.c_proj)


@register_loader("GPTJConfig")
class GPTJLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "GPTJForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            rotary_dim=model.config.rotary_dim,
            rotary_interleave=False,
            parallel_residual=True,
            shared_layer_norm=True,
        )

        self.set_decoder(
            spec.decoder,
            model.transformer,
            model.config.rotary_dim,
            model.config.n_head,
        )
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module, rotary_dim, num_heads):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.wte)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(layer_spec.shared_layer_norm, layer.ln_1)

            qw = layer.attn.q_proj.weight
            kw = layer.attn.k_proj.weight
            vw = layer.attn.v_proj.weight

            qw = utils.permute_for_sliced_rotary(qw, num_heads, rotary_dim)
            kw = utils.permute_for_sliced_rotary(kw, num_heads, rotary_dim)

            layer_spec.self_attention.linear[0].weight = torch.cat((qw, kw, vw))
            self.set_linear(layer_spec.self_attention.linear[1], layer.attn.out_proj)

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.fc_in)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.fc_out)


@register_loader("CodeGenConfig")
class CodeGenLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "CodeGenForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            rotary_dim=model.config.rotary_dim,
            rotary_interleave=False,
            parallel_residual=True,
            shared_layer_norm=True,
        )

        mp_num = 4
        if hasattr(model.config, "head_dim") and model.config.head_dim in [128, 256]:
            # models forked from "Salesforce/codegen2-1B" and "Salesforce/codegen2-3_7B"
            # use a special setting of mp_num=8, all other using 4
            # these model.config's use a special setting of head_dim
            mp_num = 8

        self.set_decoder(
            spec.decoder,
            model.transformer,
            model.config.rotary_dim,
            model.config.n_head,
            model.config.n_embd,
            mp_num=mp_num,
        )
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            # fix for additional vocab, see GPTNeoX Converter
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module, rotary_dim, num_heads, embed_dim, mp_num):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.wte)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        base_permutation = np.arange(0, mp_num * 3).reshape(-1, 3).T.flatten().tolist()
        local_dim = embed_dim // mp_num
        permutation = torch.cat(
            [torch.arange(i * local_dim, (i + 1) * local_dim) for i in base_permutation]
        )

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(layer_spec.shared_layer_norm, layer.ln_1)
            # [start convert CodeGen to GPT-J format]
            # see https://github.com/fauxpilot/fauxpilot/blob/fb4073a9078dd001ebeb7dfefb8cb2ecc8a88f4b/converter/codegen_gptj_convert.py # noqa
            qkv_proj = layer.attn.qkv_proj.weight

            # GPT-J and CodeGen slice up the qkv projection slightly differently.
            # the following permutation brings Codegen 'qkv_proj'
            # in GPT-J order of qw, vw, kw
            # we permute the *rows* here because the computation is xA.T
            new_qkv_proj = qkv_proj[permutation, :]
            # the name QKV is misleading here; they are actually stored in QVK
            qw, vw, kw = new_qkv_proj.chunk(3, dim=0)
            # [end convert CodeGen to GPT-J.]

            qw = utils.permute_for_sliced_rotary(qw, num_heads, rotary_dim)
            kw = utils.permute_for_sliced_rotary(kw, num_heads, rotary_dim)

            layer_spec.self_attention.linear[0].weight = torch.cat((qw, kw, vw))
            self.set_linear(layer_spec.self_attention.linear[1], layer.attn.out_proj)

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.fc_in)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.fc_out)


@register_loader("GPTNeoXConfig")
class GPTNeoXLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "GPTNeoXForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.hidden_act],
            rotary_dim=int(
                model.config.rotary_pct
                * (model.config.hidden_size // model.config.num_attention_heads)
            ),
            rotary_interleave=False,
            parallel_residual=model.config.use_parallel_residual,
            shared_layer_norm=False,
        )

        self.set_decoder(spec.decoder, model.gpt_neox, model.config.num_attention_heads)
        self.set_linear(spec.decoder.projection, model.embed_out)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module, num_heads):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_in)
        self.set_layer_norm(spec.layer_norm, module.final_layer_norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            if hasattr(layer_spec, "input_layer_norm"):  # Use parallel residual.
                self.set_layer_norm(layer_spec.input_layer_norm, layer.input_layernorm)
                self.set_layer_norm(
                    layer_spec.post_attention_layer_norm, layer.post_attention_layernorm
                )
            else:
                self.set_layer_norm(
                    layer_spec.self_attention.layer_norm, layer.input_layernorm
                )
                self.set_layer_norm(
                    layer_spec.ffn.layer_norm, layer.post_attention_layernorm
                )

            qkv_w = layer.attention.query_key_value.weight
            qkv_b = layer.attention.query_key_value.bias

            qkv_w = (
                qkv_w.reshape(num_heads, 3, -1, qkv_w.shape[-1])
                .swapaxes(0, 1)
                .reshape(-1, qkv_w.shape[-1])
            )
            qkv_b = qkv_b.reshape(num_heads, 3, -1).swapaxes(0, 1).reshape(-1)

            layer_spec.self_attention.linear[0].weight = qkv_w
            layer_spec.self_attention.linear[0].bias = qkv_b

            self.set_linear(layer_spec.self_attention.linear[1], layer.attention.dense)

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.dense_h_to_4h)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.dense_4h_to_h)


@register_loader("WhisperConfig")
class WhisperLoader(BartLoader):
    @property
    def architecture_name(self):
        return "WhisperForConditionalGeneration"

    def get_model_spec(self, model):
        spec = whisper_spec.WhisperSpec(
            model.config.encoder_layers,
            model.config.encoder_attention_heads,
            model.config.decoder_layers,
            model.config.decoder_attention_heads,
        )

        self.set_encoder(spec.encoder, model.model.encoder)
        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.proj_out)

        return spec

    def _get_lang_ids_from_tokenizer(self, tokenizer):
        non_lang_special_tokens = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nocaptions|>",
            "<|notimestamps|>",
        ]
        return [
            token_id
            for token_id, token in zip(
                tokenizer.additional_special_tokens_ids,
                tokenizer.additional_special_tokens,
            )
            if token not in non_lang_special_tokens
        ]

    def set_config(self, config, model, tokenizer):
        gen_config = getattr(model, "generation_config", None)

        if gen_config is not None:
            config.suppress_ids = gen_config.suppress_tokens
            config.suppress_ids_begin = gen_config.begin_suppress_tokens
            if hasattr(gen_config, "alignment_heads"):
                config.alignment_heads = gen_config.alignment_heads
            if hasattr(gen_config, "lang_to_id"):
                config.lang_ids = sorted(gen_config.lang_to_id.values())
        else:
            config.suppress_ids = model.config.suppress_tokens
            config.suppress_ids_begin = model.config.begin_suppress_tokens
            config.alignment_heads = _WHISPER_ALIGNMENT_HEADS.get(model.name_or_path)

        if getattr(config, "lang_ids", None) is None:
            config.lang_ids = self._get_lang_ids_from_tokenizer(tokenizer)

        if config.alignment_heads is None:
            # Use the last half layers for alignment by default.
            num_layers = model.config.decoder_layers
            num_heads = model.config.decoder_attention_heads
            config.alignment_heads = list(
                itertools.product(
                    range(num_layers // 2, num_layers),
                    range(num_heads),
                )
            )

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        # Add timestamp tokens.
        tokens.extend(
            "<|%.2f|>" % (i * 0.02)
            for i in range(model.config.vocab_size - len(tokens))
        )

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_encoder(self, spec, encoder):
        self.set_conv1d(spec.conv1, encoder.conv1)
        self.set_conv1d(spec.conv2, encoder.conv2)
        super().set_encoder(spec, encoder)

    def set_decoder(self, spec, decoder):
        self.set_embeddings(spec.embeddings, decoder.embed_tokens)
        super().set_decoder(spec, decoder)

    def set_common_layers(self, spec, module):
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_conv1d(self, spec, module):
        spec.weight = module.weight
        spec.bias = module.bias


@register_loader("Wav2Vec2Config")
class Wav2Vec2Loader(BartLoader):
    @property
    def architecture_name(self):
        return "Wav2Vec2ForCTC"

    def get_model_spec(self, model):
        return_hidden = getattr(model.wav2vec2.config, "return_hidden", False)
        spec = wav2vec2_spec.Wav2Vec2Spec(
            model.wav2vec2.config.num_feat_extract_layers,
            model.wav2vec2.encoder.config.num_hidden_layers,
            model.wav2vec2.encoder.config.num_attention_heads,
            model.lm_head.weight.shape[0],
            return_hidden,
        )

        # layer component name matching (no duplications saving)
        for layer in model.wav2vec2.encoder.layers:
            layer.self_attn = layer.attention
            layer.self_attn_layer_norm = layer.layer_norm
            layer.activation_fn = layer.feed_forward.intermediate_act_fn
            layer.fc1 = layer.feed_forward.intermediate_dense
            layer.fc2 = layer.feed_forward.output_dense

        self.set_encoder(spec.encoder, model, model.wav2vec2.config)
        return spec

    def set_config(self, config, model, tokenizer):
        return

    def get_vocabulary(self, model, tokenizer):
        return tokenizer.get_vocab()

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_feature_extractor(self, spec, feature_extractor):
        spec.feat_layer0.conv.weight = feature_extractor.conv_layers[0].conv.weight
        spec.feat_layer0.conv.bias = feature_extractor.conv_layers[0].conv.bias
        self.set_layer_norm(
            spec.feat_layer0.layer_norm, feature_extractor.conv_layers[0].layer_norm
        )
        for spec_layer, module_layer in zip(
            spec.feat_layer, feature_extractor.conv_layers[1:]
        ):
            spec_layer.conv.weight = module_layer.conv.weight
            spec_layer.conv.bias = module_layer.conv.bias
            self.set_layer_norm(spec_layer.layer_norm, module_layer.layer_norm)

    def set_feature_projection(self, spec, feature_projection):
        self.set_layer_norm(spec.fp_layer_norm, feature_projection.layer_norm)
        self.set_linear(spec.fp_projection, feature_projection.projection)

    def set_pos_conv_embed(self, spec, encoder, config):
        # forcing parameters to be set because some transformers version initializes garbage numbers
        # conv parameters are float16 so force float32 for the loading
        encoder.pos_conv_embed.conv.weight.data = (
            encoder.pos_conv_embed.conv.weight.data.float()
        )
        encoder.pos_conv_embed.conv.bias.data = encoder.pos_conv_embed.conv.bias.float()
        for param in encoder.pos_conv_embed.parameters():
            param.data = param.data.float()
        encoder.pos_conv_embed(torch.randn((1, 1, config.hidden_size)))
        spec.pos_conv_embed.conv.weight = encoder.pos_conv_embed.conv.weight
        spec.pos_conv_embed.conv.bias = encoder.pos_conv_embed.conv.bias

    def set_encoder(self, spec, model, config):
        self.set_feature_extractor(spec, model.wav2vec2.feature_extractor)
        self.set_feature_projection(spec, model.wav2vec2.feature_projection)
        self.set_pos_conv_embed(spec, model.wav2vec2.encoder, config)
        super().set_encoder(spec, model.wav2vec2.encoder)
        return_hidden = getattr(model.wav2vec2.config, "return_hidden", False)
        if not return_hidden:
            self.set_linear(spec.lm_head, model.lm_head)

    def set_common_layers(self, spec, module):
        self.set_layer_norm(spec.layer_norm, module.layer_norm)


@register_loader("Wav2Vec2BertConfig")
class Wav2Vec2BertLoader(BartLoader):
    @property
    def architecture_name(self):
        return "Wav2Vec2BertForCTC"

    def get_model_spec(self, model):
        return_hidden = getattr(model.wav2vec2_bert.config, "return_hidden", False)
        spec = wav2vec2bert_spec.Wav2Vec2BertSpec(
            model.wav2vec2_bert.config.num_adapter_layers,
            model.wav2vec2_bert.config.num_hidden_layers,
            model.lm_head.weight.shape[0],
            return_hidden,
        )
        self.set_encoder(spec.encoder, model)
        return spec

    def set_config(self, config, model, tokenizer):
        return

    def get_vocabulary(self, model, tokenizer):
        return tokenizer.get_vocab()

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_feature_projection(self, spec, feature_projection):
        self.set_layer_norm(spec.fp_layer_norm, feature_projection.layer_norm)
        self.set_linear(spec.fp_projection, feature_projection.projection)

    def set_attention(
        self, spec, attention, left_max_position=None, right_max_position=None
    ):
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], attention.linear_q)
        self.set_linear(split_layers[1], attention.linear_k)
        self.set_linear(split_layers[2], attention.linear_v)
        utils.fuse_linear(spec.linear[0], split_layers)
        self.set_linear(spec.linear[-1], attention.linear_out)
        if left_max_position or right_max_position:
            spec.relative_asymmetric_position_keys = attention.distance_embedding.weight
            spec.relative_left_max_position = np.dtype("int32").type(left_max_position)
            spec.relative_right_max_position = np.dtype("int32").type(
                right_max_position
            )

    def set_wav2vec2bert_encoder(
        self, spec_layers, layers, left_max_position, right_max_position
    ):
        for slayer, layer in zip(spec_layers, layers):
            self.set_layer_norm(slayer.enc_ffn1_layer_norm, layer.ffn1_layer_norm)
            self.set_linear(slayer.enc_ffn1.linear_0, layer.ffn1.intermediate_dense)
            self.set_linear(slayer.enc_ffn1.linear_1, layer.ffn1.output_dense)
            self.set_attention(
                slayer.enc_attn, layer.self_attn, left_max_position, right_max_position
            )
            self.set_layer_norm(slayer.enc_attn_layer_norm, layer.self_attn_layer_norm)
            self.set_layer_norm(
                slayer.enc_conv_layer_norm, layer.conv_module.layer_norm
            )
            self.set_conv1d(
                slayer.enc_conv_pointwise_conv1, layer.conv_module.pointwise_conv1
            )
            self.set_conv1d(
                slayer.enc_conv_depthwise_conv, layer.conv_module.depthwise_conv
            )
            self.set_layer_norm(
                slayer.enc_conv_depthwise_layer_norm,
                layer.conv_module.depthwise_layer_norm,
            )
            self.set_conv1d(
                slayer.enc_conv_pointwise_conv2, layer.conv_module.pointwise_conv2
            )
            self.set_layer_norm(slayer.enc_ffn2_layer_norm, layer.ffn2_layer_norm)
            self.set_linear(slayer.enc_ffn2.linear_0, layer.ffn2.intermediate_dense)
            self.set_linear(slayer.enc_ffn2.linear_1, layer.ffn2.output_dense)
            self.set_layer_norm(slayer.enc_final_layer_norm, layer.final_layer_norm)

    def set_wav2vec2bert_adapter(self, spec_layers, layers):
        for slayer, layer in zip(spec_layers, layers):
            self.set_layer_norm(
                slayer.adpt_residual_layer_norm, layer.residual_layer_norm
            )
            self.set_conv1d(slayer.adpt_residual_conv, layer.residual_conv)
            self.set_layer_norm(slayer.adpt_attn_layer_norm, layer.self_attn_layer_norm)
            self.set_conv1d(slayer.adpt_attn_conv, layer.self_attn_conv)
            self.set_attention(slayer.adpt_attn_layer, layer.self_attn)
            self.set_layer_norm(slayer.adpt_ffn_layer_norm, layer.ffn_layer_norm)
            self.set_linear(slayer.adpt_ffn.linear_0, layer.ffn.intermediate_dense)
            self.set_linear(slayer.adpt_ffn.linear_1, layer.ffn.output_dense)

    def set_encoder(self, spec, model):
        self.set_feature_projection(spec, model.wav2vec2_bert.feature_projection)
        self.set_wav2vec2bert_encoder(
            spec.encoder_layers,
            model.wav2vec2_bert.encoder.layers,
            model.wav2vec2_bert.config.left_max_position_embeddings,
            model.wav2vec2_bert.config.right_max_position_embeddings,
        )
        self.set_wav2vec2bert_adapter(
            spec.adapter_layers, model.wav2vec2_bert.adapter.layers
        )
        return_hidden = getattr(model.wav2vec2_bert.config, "return_hidden", False)
        if not return_hidden:
            self.set_linear(spec.lm_head, model.lm_head)

    def set_conv1d(self, spec, module):
        spec.weight = module.weight
        if module.bias is not None:
            spec.bias = module.bias

    def set_layer_norm(self, spec, module):
        spec.gamma = module.weight
        if module.bias is not None:
            spec.beta = module.bias


@register_loader("T5Config")
class T5Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "T5ForConditionalGeneration"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerSpec.from_config(
            (model.config.num_layers, model.config.num_decoder_layers),
            model.config.num_heads,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.dense_act_fn],
            ffn_glu=model.config.is_gated_act,
            relative_attention_bias=True,
            rms_norm=True,
        )

        self.set_stack(spec.encoder, model.encoder)
        self.set_stack(spec.decoder, model.decoder, is_decoder=True)
        self.set_linear(spec.decoder.projection, model.lm_head)

        if model.config.tie_word_embeddings:
            spec.decoder.scale_outputs = model.config.d_model**-0.5

        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_source_vocabulary(tokens)
        spec.register_target_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.pad_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        if hasattr(model.config, "decoder_start_token_id"):
            config.decoder_start_token = tokenizer.convert_ids_to_tokens(
                model.config.decoder_start_token_id
            )
        else:
            config.decoder_start_token = tokenizer.pad_token

    def set_stack(self, spec, module, is_decoder=False):
        self.set_layer_norm(spec.layer_norm, module.final_layer_norm)
        self.set_embeddings(
            (
                spec.embeddings[0]
                if isinstance(spec.embeddings, list)
                else spec.embeddings
            ),
            module.embed_tokens,
        )

        spec.scale_embeddings = False

        for i, (layer_spec, block) in enumerate(zip(spec.layer, module.block)):
            self.set_self_attention(layer_spec.self_attention, block.layer[0])

            if i > 0:
                # Reuse relative attention bias from the first layer.
                first_self_attention = spec.layer[0].self_attention
                layer_spec.self_attention.relative_attention_bias = (
                    first_self_attention.relative_attention_bias
                )
                layer_spec.self_attention.relative_attention_max_distance = (
                    first_self_attention.relative_attention_max_distance
                )

            if is_decoder:
                self.set_cross_attention(layer_spec.attention, block.layer[1])

            self.set_ffn(layer_spec.ffn, block.layer[-1])

    def set_ffn(self, spec, module):
        if hasattr(spec, "linear_0_noact"):
            self.set_linear(spec.linear_0, module.DenseReluDense.wi_0)
            self.set_linear(spec.linear_0_noact, module.DenseReluDense.wi_1)
        else:
            self.set_linear(spec.linear_0, module.DenseReluDense.wi)

        self.set_linear(spec.linear_1, module.DenseReluDense.wo)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_self_attention(self, spec, module):
        self.set_attention(spec, module.SelfAttention, self_attention=True)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_cross_attention(self, spec, module):
        self.set_attention(spec, module.EncDecAttention)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_attention(self, spec, attention, self_attention=False):
        spec.queries_scale = 1.0

        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], attention.q)
        self.set_linear(split_layers[1], attention.k)
        self.set_linear(split_layers[2], attention.v)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], attention.o)

        if attention.has_relative_attention_bias:
            spec.relative_attention_bias = attention.relative_attention_bias.weight
            spec.relative_attention_max_distance = np.dtype("int32").type(
                attention.relative_attention_max_distance
            )

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight


@register_loader("MT5Config")
class MT5Loader(T5Loader):
    @property
    def architecture_name(self):
        return "MT5ForConditionalGeneration"


@register_loader("BloomConfig")
class BloomLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "BloomForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layer,
            model.config.n_head,
            pre_norm=True,
            activation=common_spec.Activation.GELUTanh,
            layernorm_embedding=True,
            alibi=True,
            alibi_use_positive_positions=True,
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.word_embeddings)
        self.set_layer_norm(spec.layernorm_embedding, module.word_embeddings_layernorm)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_qkv_linear(
                layer_spec.self_attention.linear[0],
                layer.self_attention.query_key_value,
                layer.self_attention.num_heads,
            )
            self.set_linear(
                layer_spec.self_attention.linear[1], layer.self_attention.dense
            )

            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )
            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.dense_h_to_4h)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.dense_4h_to_h)

    def set_qkv_linear(self, spec, module, num_heads):
        weight = module.weight
        weight = weight.reshape(num_heads, 3, -1, weight.shape[-1])
        weight = weight.transpose(0, 1)
        weight = weight.reshape(-1, weight.shape[-1])

        bias = module.bias
        bias = bias.reshape(num_heads, 3, -1)
        bias = bias.transpose(0, 1)
        bias = bias.reshape(-1)

        spec.weight = weight
        spec.bias = bias


@register_loader("MPTConfig")
class MPTLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "AutoModelForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            model.config.n_layers,
            model.config.n_heads,
            pre_norm=True,
            activation=common_spec.Activation.GELU,
            alibi=True,
        )

        self.set_decoder(spec.decoder, model.transformer)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module):
        self.set_embeddings(spec.embeddings, module.wte)
        self.set_layer_norm(spec.layer_norm, module.norm_f)

        spec.scale_embeddings = False
        spec.projection.weight = spec.embeddings.weight

        for layer_spec, layer in zip(spec.layer, module.blocks):
            self.set_layer_norm(layer_spec.self_attention.layer_norm, layer.norm_1)
            self.set_linear(layer_spec.self_attention.linear[0], layer.attn.Wqkv)
            self.set_linear(layer_spec.self_attention.linear[1], layer.attn.out_proj)

            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.norm_2)
            self.set_linear(layer_spec.ffn.linear_0, layer.ffn.up_proj)
            self.set_linear(layer_spec.ffn.linear_1, layer.ffn.down_proj)

    def set_layer_norm(self, spec, module):
        spec.gamma = module.weight
        spec.beta = torch.zeros_like(spec.gamma)


@register_loader("GemmaConfig")
class GemmaLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "GemmaForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers

        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        if num_heads_kv == num_heads:
            num_heads_kv = None

        activation_config = getattr(
            model.config, "hidden_activation", "gelu_pytorch_tanh"
        )

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=(
                common_spec.Activation.GELU
                if activation_config == "gelu"
                else common_spec.Activation.GELUTanh
            ),
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
            rotary_base=getattr(model.config, "rope_theta", 10000),
            num_heads_kv=num_heads_kv,
            head_dim=model.config.head_dim,
        )

        self.set_decoder(spec.decoder, model.model)
        self.set_linear(spec.decoder.projection, model.lm_head)
        spec.decoder.embeddings.multiply_by_sqrt_depth = model.config.hidden_size**0.5
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = model.config.rms_norm_eps

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight
        spec.layer_norm_use_residual = True

    def set_decoder(self, spec, module):
        spec.scale_embeddings = True
        spec.start_from_zero_embedding = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            wq = layer.self_attn.q_proj.weight
            wk = layer.self_attn.k_proj.weight
            wv = layer.self_attn.v_proj.weight
            wo = layer.self_attn.o_proj.weight

            layer_spec.self_attention.linear[0].weight = torch.cat([wq, wk, wv])
            layer_spec.self_attention.linear[1].weight = wo

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.gate_proj)
            self.set_linear(layer_spec.ffn.linear_0_noact, layer.mlp.up_proj)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.down_proj)

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("Gemma2Config")
class Gemma2Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "Gemma2ForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers

        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        if num_heads_kv == num_heads:
            num_heads_kv = None

        activation_config = getattr(
            model.config, "hidden_activation", "gelu_pytorch_tanh"
        )

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=(
                common_spec.Activation.GELU
                if activation_config == "gelu"
                else common_spec.Activation.GELUTanh
            ),
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
            rotary_base=getattr(model.config, "rope_theta", 10000),
            num_heads_kv=num_heads_kv,
            head_dim=model.config.head_dim,
            pre_post_layer_norm=True,
        )

        self.set_decoder(spec.decoder, model.model)
        self.set_linear(spec.decoder.projection, model.lm_head)
        spec.decoder.embeddings.multiply_by_sqrt_depth = model.config.hidden_size**0.5
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = model.config.rms_norm_eps

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight
        spec.layer_norm_use_residual = True

    def set_decoder(self, spec, module):
        spec.scale_embeddings = True
        spec.start_from_zero_embedding = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(layer_spec.input_layer_norm, layer.input_layernorm)

            self.set_layer_norm(
                layer_spec.post_attention_layer_norm, layer.post_attention_layernorm
            )

            self.set_layer_norm(
                layer_spec.pre_feedforward_layer_norm, layer.pre_feedforward_layernorm
            )

            self.set_layer_norm(
                layer_spec.post_feedforward_layer_norm, layer.post_feedforward_layernorm
            )

            wq = layer.self_attn.q_proj.weight
            wk = layer.self_attn.k_proj.weight
            wv = layer.self_attn.v_proj.weight
            wo = layer.self_attn.o_proj.weight

            layer_spec.self_attention.linear[0].weight = torch.cat([wq, wk, wv])
            layer_spec.self_attention.linear[1].weight = wo

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.gate_proj)
            self.set_linear(layer_spec.ffn.linear_0_noact, layer.mlp.up_proj)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.down_proj)

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("LlamaConfig")
class LlamaLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "LlamaForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers

        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        if num_heads_kv == num_heads:
            num_heads_kv = None

        rope_scaling = getattr(model.config, "rope_scaling", None)
        if rope_scaling:
            rope_type = rope_scaling.get("type") or rope_scaling["rope_type"]
            rotary_scaling_type = _SUPPORTED_ROPE_SCALING.get(rope_type)
            rotary_scaling_factor = rope_scaling["factor"]

            if rotary_scaling_type is None:
                raise NotImplementedError(
                    "RoPE scaling type '%s' is not yet implemented. "
                    "The following RoPE scaling types are currently supported: %s"
                    % (rope_scaling["type"], ", ".join(_SUPPORTED_ROPE_SCALING.keys()))
                )
        else:
            rotary_scaling_type = None
            rotary_scaling_factor = 1

        quantization_config = getattr(model.config, "quantization_config", None)
        if quantization_config:
            quant_type = None
            if quantization_config.quant_method == "awq":
                quant_type = _SUPPORTED_QUANTIZATION.get(quantization_config.version)
            if quant_type is None:
                raise NotImplementedError(
                    "Quantization type '%s' is not yet implemented. "
                    "The following Quantization types are currently supported: %s"
                    % (
                        quantization_config.quant_method,
                        ", ".join(_SUPPORTED_QUANTIZATION.keys()),
                    )
                )
            quant_group_size = quantization_config.group_size
            quant_bits = quantization_config.bits
        else:
            quant_type = common_spec.Quantization.CT2
            quant_group_size = None
            quant_bits = None

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=common_spec.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=getattr(model.config, "rope_theta", 10000),
            num_heads_kv=num_heads_kv,
            quant_type=quant_type,
            quant_group_size=quant_group_size,
            quant_bits=quant_bits,
        )

        self.set_decoder(spec.decoder, model.model, quant_type)
        self.set_linear(spec.decoder.projection, model.lm_head)

        # set extra RoPE parameters for Llama-3.1
        if rotary_scaling_type == attention_spec.RotaryScalingType.Llama3:
            for layer in spec.decoder.layer:
                layer.self_attention.rotary_low_freq_factor = rope_scaling[
                    "low_freq_factor"
                ]
                layer.self_attention.rotary_high_freq_factor = rope_scaling[
                    "high_freq_factor"
                ]
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = (
            tokenizer.unk_token if tokenizer.unk_token is not None else ""
        )
        config.layer_norm_epsilon = model.config.rms_norm_eps

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight

    def set_decoder(self, spec, module, quant_type=common_spec.Quantization.CT2):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(
                split_layers[0], layer.self_attn.q_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[1], layer.self_attn.k_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[2], layer.self_attn.v_proj, quant_type=quant_type
            )

            if quant_type == common_spec.Quantization.CT2:
                utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)
            else:
                cc_dim = 1 if quant_type == common_spec.Quantization.AWQ_GEMM else 0
                utils.fuse_linear_prequant(
                    layer_spec.self_attention.linear[0], split_layers, cc_dim
                )
            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )

            self.set_linear(
                layer_spec.ffn.linear_0, layer.mlp.gate_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_0_noact, layer.mlp.up_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("Gemma3TextConfig")
@register_loader("Gemma3Config")
class Gemma3Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "Gemma3ForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        if num_heads_kv == num_heads:
            num_heads_kv = None

        head_dim = model.config.head_dim

        activation_config = getattr(
            model.config, "hidden_activation", "gelu_pytorch_tanh"
        )

        # Get RoPE parameters
        rope_theta = getattr(model.config, "rope_theta", 1_000_000)  # Global: 1M
        rope_local_base_freq = getattr(
            model.config, "rope_local_base_freq", 10_000
        )  # Local: 10k

        # Get sliding window configuration
        sliding_window = getattr(model.config, "sliding_window", 1024)
        layer_types = getattr(model.config, "layer_types", None)

        quantization_config = getattr(model.config, "quantization_config", None)
        if quantization_config:
            if quantization_config.quant_method == "awq":
                quant_type = _SUPPORTED_QUANTIZATION.get(quantization_config.version)
            if quant_type is None:
                raise NotImplementedError(
                    "Quantization type '%s' is not yet implemented."
                    % quantization_config.quant_method
                )
        else:
            quant_type = common_spec.Quantization.CT2

        # Create base spec using from_config
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=(
                common_spec.Activation.GELU
                if activation_config == "gelu"
                else common_spec.Activation.GELUTanh
            ),
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=head_dim,
            rotary_interleave=False,
            rotary_base=rope_local_base_freq,  # Default to local base freq
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            sliding_window=sliding_window,  # Default to local sliding window
            pre_post_layer_norm=True,
            qk_norm=True,
        )

        # Store layer_types for use in set_decoder
        self._layer_types = layer_types

        # Override per-layer settings for global vs local attention
        for i, layer_type in enumerate(layer_types):
            layer = spec.decoder.layer[i]
            if layer_type == "full_attention":
                layer.self_attention.rotary_base = np.dtype("float32").type(rope_theta)
                layer.self_attention.sliding_window = np.dtype("int32").type(0)
            elif layer_type == "sliding_attention":
                layer.self_attention.rotary_base = np.dtype("float32").type(
                    rope_local_base_freq
                )
                layer.self_attention.sliding_window = np.dtype("int32").type(
                    sliding_window
                )

        self.set_decoder(spec.decoder, model.model, quant_type)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.unk_token = tokenizer.unk_token

        if (
            hasattr(tokenizer, "chat_template")
            and isinstance(tokenizer.chat_template, str)
            and tokenizer.chat_template.strip()
        ):
            config.eos_token = "<end_of_turn>"
        else:
            config.eos_token = tokenizer.eos_token

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight + 1.0

    def set_decoder(self, spec, module, quant_type=common_spec.Quantization.CT2):
        spec.scale_embeddings = True
        spec.start_from_zero_embedding = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)  # Input
        self.set_layer_norm(spec.layer_norm, module.norm)  # Output

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(layer_spec.input_layer_norm, layer.input_layernorm)

            self.set_layer_norm(
                layer_spec.post_attention_layer_norm, layer.post_attention_layernorm
            )

            self.set_layer_norm(
                layer_spec.pre_feedforward_layer_norm, layer.pre_feedforward_layernorm
            )

            self.set_layer_norm(
                layer_spec.post_feedforward_layer_norm, layer.post_feedforward_layernorm
            )

            # Set QK-norm weights (Gemma 3 uses this instead of soft-capping)
            self.set_layer_norm(
                layer_spec.self_attention.q_norm, layer.self_attn.q_norm
            )
            self.set_layer_norm(
                layer_spec.self_attention.k_norm, layer.self_attn.k_norm
            )

            # Set attention projections
            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(
                split_layers[0], layer.self_attn.q_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[1], layer.self_attn.k_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[2], layer.self_attn.v_proj, quant_type=quant_type
            )

            if quant_type == common_spec.Quantization.CT2:
                utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)
            else:
                cc_dim = 1 if quant_type == common_spec.Quantization.AWQ_GEMM else 0
                utils.fuse_linear_prequant(
                    layer_spec.self_attention.linear[0], split_layers, cc_dim
                )

            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )

            # Set FFN weights
            self.set_linear(
                layer_spec.ffn.linear_0, layer.mlp.gate_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_0_noact, layer.mlp.up_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("MistralConfig")
class MistralLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "MistralForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers

        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        if num_heads_kv == num_heads:
            num_heads_kv = None

        sliding_window = getattr(model.config, "sliding_window", 0)

        rope_scaling = getattr(model.config, "rope_scaling", None)
        if rope_scaling:
            rotary_scaling_type = _SUPPORTED_ROPE_SCALING.get(rope_scaling["type"])
            rotary_scaling_factor = rope_scaling["factor"]

            if rotary_scaling_type is None:
                raise NotImplementedError(
                    "RoPE scaling type '%s' is not yet implemented. "
                    "The following RoPE scaling types are currently supported: %s"
                    % (rope_scaling["type"], ", ".join(_SUPPORTED_ROPE_SCALING.keys()))
                )
        else:
            rotary_scaling_type = None
            rotary_scaling_factor = 1

        quantization_config = getattr(model.config, "quantization_config", None)
        if quantization_config:
            if quantization_config.quant_method == "awq":
                quant_type = _SUPPORTED_QUANTIZATION.get(quantization_config.version)
            if quant_type is None:
                raise NotImplementedError(
                    "Quantization type '%s' is not yet implemented. "
                    "The following Quantization types are currently supported: %s"
                    % (
                        quantization_config.quant_method,
                        ", ".join(_SUPPORTED_QUANTIZATION.keys()),
                    )
                )
            quant_group_size = quantization_config.group_size
            quant_bits = quantization_config.bits
        else:
            quant_type = common_spec.Quantization.CT2
            quant_group_size = None
            quant_bits = None

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=common_spec.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=getattr(model.config, "rope_theta", 10000),
            num_heads_kv=num_heads_kv,
            sliding_window=sliding_window,
            quant_type=quant_type,
            quant_group_size=quant_group_size,
            quant_bits=quant_bits,
            head_dim=model.config.head_dim,
        )

        self.set_decoder(spec.decoder, model.model, quant_type=quant_type)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = model.config.rms_norm_eps

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight

    def set_decoder(self, spec, module, quant_type=common_spec.Quantization.CT2):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )
            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(
                split_layers[0], layer.self_attn.q_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[1], layer.self_attn.k_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[2], layer.self_attn.v_proj, quant_type=quant_type
            )

            if quant_type == common_spec.Quantization.CT2:
                utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)
            else:
                cc_dim = 1 if quant_type == common_spec.Quantization.AWQ_GEMM else 0
                utils.fuse_linear_prequant(
                    layer_spec.self_attention.linear[0], split_layers, cc_dim
                )
            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )

            self.set_linear(
                layer_spec.ffn.linear_0, layer.mlp.gate_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_0_noact, layer.mlp.up_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("Qwen2Config")
class Qwen2Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "Qwen2ForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers

        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        if num_heads_kv == num_heads:
            num_heads_kv = None

        rope_scaling = getattr(model.config, "rope_scaling", None)
        if rope_scaling:
            rope_type = rope_scaling.get("type") or rope_scaling["rope_type"]
            rotary_scaling_type = _SUPPORTED_ROPE_SCALING.get(rope_type)
            rotary_scaling_factor = rope_scaling["factor"]

            if rotary_scaling_type is None:
                raise NotImplementedError(
                    "RoPE scaling type '%s' is not yet implemented. "
                    "The following RoPE scaling types are currently supported: %s"
                    % (rope_scaling["type"], ", ".join(_SUPPORTED_ROPE_SCALING.keys()))
                )
        else:
            rotary_scaling_type = None
            rotary_scaling_factor = 1

        # Check for AWQ quantization config
        quantization_config = getattr(model.config, "quantization_config", None)
        if quantization_config:
            quant_type = None
            if quantization_config.quant_method == "awq":
                quant_type = _SUPPORTED_QUANTIZATION.get(quantization_config.version)
            if quant_type is None:
                raise NotImplementedError(
                    "Quantization type '%s' is not yet implemented. "
                    "The following Quantization types are currently supported: %s"
                    % (
                        quantization_config.quant_method,
                        ", ".join(_SUPPORTED_QUANTIZATION.keys()),
                    )
                )
            quant_group_size = quantization_config.group_size
            quant_bits = quantization_config.bits
        else:
            quant_type = common_spec.Quantization.CT2
            quant_group_size = None
            quant_bits = None

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=common_spec.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=getattr(model.config, "rope_theta", 10000),
            num_heads_kv=num_heads_kv,
            quant_type=quant_type,
            quant_group_size=quant_group_size,
            quant_bits=quant_bits,
        )

        self.set_decoder(spec.decoder, model.model, quant_type)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)
        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = (
            tokenizer.bos_token
            if tokenizer.bos_token is not None
            else tokenizer.pad_token
        )
        config.eos_token = tokenizer.eos_token
        config.unk_token = (
            tokenizer.unk_token if tokenizer.unk_token is not None else ""
        )
        config.layer_norm_epsilon = model.config.rms_norm_eps

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight

    def set_decoder(self, spec, module, quant_type=common_spec.Quantization.CT2):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(
                split_layers[0], layer.self_attn.q_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[1], layer.self_attn.k_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[2], layer.self_attn.v_proj, quant_type=quant_type
            )

            if quant_type == common_spec.Quantization.CT2:
                utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)
            else:
                cc_dim = 1 if quant_type == common_spec.Quantization.AWQ_GEMM else 0
                utils.fuse_linear_prequant(
                    layer_spec.self_attention.linear[0], split_layers, cc_dim
                )

            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )

            self.set_linear(
                layer_spec.ffn.linear_0, layer.mlp.gate_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_0_noact, layer.mlp.up_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("Qwen3Config")
class Qwen3Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "Qwen3ForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        head_dim = getattr(
            model.config, "head_dim", model.config.hidden_size // num_heads
        )

        if num_heads_kv == num_heads:
            num_heads_kv = None

        rope_scaling = getattr(model.config, "rope_scaling", None)
        if rope_scaling:
            rope_type = rope_scaling.get("type") or rope_scaling["rope_type"]
            rotary_scaling_type = _SUPPORTED_ROPE_SCALING.get(rope_type)
            rotary_scaling_factor = rope_scaling["factor"]
            if rotary_scaling_type is None:
                raise NotImplementedError(
                    "RoPE scaling type '%s' is not yet implemented. "
                    "The following RoPE scaling types are currently supported: %s"
                    % (rope_scaling["type"], ", ".join(_SUPPORTED_ROPE_SCALING.keys()))
                )
        else:
            rotary_scaling_type = None
            rotary_scaling_factor = 1

        # Check for AWQ quantization config
        quantization_config = getattr(model.config, "quantization_config", None)
        if quantization_config:
            quant_type = None
            if quantization_config.quant_method == "awq":
                quant_type = _SUPPORTED_QUANTIZATION.get(quantization_config.version)
            if quant_type is None:
                raise NotImplementedError(
                    "Quantization type '%s' is not yet implemented. "
                    "The following Quantization types are currently supported: %s"
                    % (
                        quantization_config.quant_method,
                        ", ".join(_SUPPORTED_QUANTIZATION.keys()),
                    )
                )
            quant_group_size = quantization_config.group_size
            quant_bits = quantization_config.bits
        else:
            quant_type = common_spec.Quantization.CT2
            quant_group_size = None
            quant_bits = None

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=common_spec.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=model.config.head_dim,
            rotary_interleave=False,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=getattr(model.config, "rope_theta", 10000),
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            qk_norm=True,
            quant_type=quant_type,
            quant_group_size=quant_group_size,
            quant_bits=quant_bits,
        )

        self.set_decoder(spec.decoder, model.model, quant_type)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)
        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)
        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = (
            tokenizer.bos_token
            if tokenizer.bos_token is not None
            else tokenizer.pad_token
        )
        config.eos_token = tokenizer.eos_token
        config.unk_token = (
            tokenizer.unk_token if tokenizer.unk_token is not None else ""
        )
        config.layer_norm_epsilon = model.config.rms_norm_eps

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight

    def set_decoder(self, spec, module, quant_type=common_spec.Quantization.CT2):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_idx, (layer_spec, layer) in enumerate(zip(spec.layer, module.layers)):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            self.set_layer_norm(
                layer_spec.self_attention.q_norm, layer.self_attn.q_norm
            )
            self.set_layer_norm(
                layer_spec.self_attention.k_norm, layer.self_attn.k_norm
            )

            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(
                split_layers[0], layer.self_attn.q_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[1], layer.self_attn.k_proj, quant_type=quant_type
            )
            self.set_linear(
                split_layers[2], layer.self_attn.v_proj, quant_type=quant_type
            )

            if quant_type == common_spec.Quantization.CT2:
                utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)
            else:
                cc_dim = 1 if quant_type == common_spec.Quantization.AWQ_GEMM else 0
                utils.fuse_linear_prequant(
                    layer_spec.self_attention.linear[0], split_layers, cc_dim
                )

            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )

            self.set_linear(
                layer_spec.ffn.linear_0, layer.mlp.gate_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_0_noact, layer.mlp.up_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("MixFormerSequentialConfig")
class MixFormerSequentialLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "AutoModelForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers=model.config.n_layer,
            num_heads=model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            rotary_dim=model.config.rotary_dim,
            rotary_interleave=False,
            parallel_residual=True,
            shared_layer_norm=True,
        )

        self.set_decoder(spec.decoder, model.layers)
        self.set_linear(spec.decoder.projection, model.layers[-1].linear)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module[0].wte)
        self.set_layer_norm(spec.layer_norm, module[-1].ln)

        for layer_spec, layer in zip(spec.layer, module[1:-1]):
            self.set_layer_norm(layer_spec.shared_layer_norm, layer.ln)
            self.set_linear(layer_spec.self_attention.linear[0], layer.mixer.Wqkv)
            self.set_linear(layer_spec.self_attention.linear[1], layer.mixer.out_proj)
            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.fc2)


@register_loader("PhiConfig")
class PhiLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "AutoModelForCausalLM"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers=model.config.n_layer,
            num_heads=model.config.n_head,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            rotary_dim=model.config.rotary_dim,
            rotary_interleave=False,
            parallel_residual=True,
            shared_layer_norm=True,
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head.linear)
        self.set_layer_norm(spec.decoder.layer_norm, model.lm_head.ln)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embd.wte)

        for layer_spec, layer in zip(spec.layer, module.h):
            self.set_layer_norm(layer_spec.shared_layer_norm, layer.ln)
            self.set_linear(layer_spec.self_attention.linear[0], layer.mixer.Wqkv)
            self.set_linear(layer_spec.self_attention.linear[1], layer.mixer.out_proj)
            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.fc2)


@register_loader("Phi3Config")
class Phi3Loader(ModelLoader):
    @property
    def architecture_name(self):
        return "AutoModelForCausalLM"

    def get_model_spec(self, model):
        num_layers = model.config.num_hidden_layers

        num_heads = model.config.num_attention_heads
        num_heads_kv = getattr(model.config, "num_key_value_heads", num_heads)
        if num_heads_kv == num_heads:
            num_heads_kv = None

        original_max_position_embeddings = getattr(
            model.config, "original_max_position_embeddings", 0
        )
        max_position_embeddings = getattr(model.config, "max_position_embeddings", 0)
        rope_scaling = getattr(model.config, "rope_scaling", None)
        if rope_scaling:
            rotary_scaling_type = _SUPPORTED_ROPE_SCALING.get(rope_scaling["type"])
            rotary_scaling_factor = rope_scaling.get("factor", 1)

            if rotary_scaling_type is None:
                raise NotImplementedError(
                    "RoPE scaling type '%s' is not yet implemented. "
                    "The following RoPE scaling types are currently supported: %s"
                    % (rope_scaling["type"], ", ".join(_SUPPORTED_ROPE_SCALING.keys()))
                )
        else:
            rotary_scaling_type = None
            rotary_scaling_factor = 1

        # Check for AWQ quantization config
        quantization_config = getattr(model.config, "quantization_config", None)
        if quantization_config:
            quant_type = None
            if quantization_config.quant_method == "awq":
                quant_type = _SUPPORTED_QUANTIZATION.get(quantization_config.version)
            if quant_type is None:
                raise NotImplementedError(
                    "Quantization type '%s' is not yet implemented. "
                    "The following Quantization types are currently supported: %s"
                    % (
                        quantization_config.quant_method,
                        ", ".join(_SUPPORTED_QUANTIZATION.keys()),
                    )
                )
            quant_group_size = quantization_config.group_size
            quant_bits = quantization_config.bits
        else:
            quant_type = common_spec.Quantization.CT2
            quant_group_size = None
            quant_bits = None

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            num_layers,
            num_heads,
            activation=common_spec.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=getattr(model.config, "rope_theta", 10000),
            original_max_position_embeddings=original_max_position_embeddings,
            max_position_embeddings=max_position_embeddings,
            num_heads_kv=num_heads_kv,
            quant_type=quant_type,
            quant_group_size=quant_group_size,
            quant_bits=quant_bits,
        )

        self.set_decoder(spec.decoder, model.model, quant_type)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight

    def set_rotary_embeddings(
        self, spec, rotary_scaling_long_factor, rotary_scaling_short_factor
    ):
        spec.rotary_scaling_long_factor = torch.tensor(
            rotary_scaling_long_factor, dtype=torch.float32
        )
        spec.rotary_scaling_short_factor = torch.tensor(
            rotary_scaling_short_factor, dtype=torch.float32
        )

    def set_decoder(self, spec, module, quant_type=common_spec.Quantization.CT2):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            self.set_linear(
                layer_spec.self_attention.linear[0],
                layer.self_attn.qkv_proj,
                quant_type=quant_type,
            )
            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )
            if (
                layer.self_attn.rotary_emb.long_factor is not None
                and layer.self_attn.rotary_emb.short_factor is not None
            ):
                self.set_rotary_embeddings(
                    layer_spec.self_attention,
                    layer.self_attn.rotary_emb.long_factor,
                    layer.self_attn.rotary_emb.short_factor,
                )

            # Handle gate_up_proj differently for AWQ vs regular models
            if quant_type == common_spec.Quantization.CT2:
                gate_proj, up_proj = layer.mlp.gate_up_proj.weight.chunk(2, dim=0)
                layer_spec.ffn.linear_0.weight = gate_proj
                layer_spec.ffn.linear_0_noact.weight = up_proj
            else:
                # AWQ: chunk qweight, scales, and qzeros
                gate_qweight, up_qweight = layer.mlp.gate_up_proj.qweight.chunk(
                    2, dim=1
                )
                gate_scales, up_scales = layer.mlp.gate_up_proj.scales.chunk(2, dim=1)
                gate_qzeros, up_qzeros = layer.mlp.gate_up_proj.qzeros.chunk(2, dim=1)

                layer_spec.ffn.linear_0.weight = gate_qweight
                layer_spec.ffn.linear_0.weight_scale = gate_scales
                layer_spec.ffn.linear_0.weight_zero = gate_qzeros

                layer_spec.ffn.linear_0_noact.weight = up_qweight
                layer_spec.ffn.linear_0_noact.weight_scale = up_scales
                layer_spec.ffn.linear_0_noact.weight_zero = up_qzeros

            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()


@register_loader("RWConfig")
class RWLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "AutoModelForCausalLM"

    def get_falcon_spec(self, model):
        self._num_layers = model.config.n_layer
        self._num_heads = model.config.n_head
        self._num_heads_kv = getattr(model.config, "n_head_kv", None)
        self._num_kv_attr = "num_kv"

    def get_model_spec(self, model):
        self.get_falcon_spec(model)

        if getattr(model.config, "multi_query", False):
            num_heads_kv = 1
        else:
            num_heads_kv = self._num_heads_kv

        spec = transformer_spec.TransformerDecoderModelSpec.from_config(
            self._num_layers,
            self._num_heads,
            pre_norm=True,
            activation=common_spec.Activation.GELU,
            alibi=model.config.alibi,
            alibi_use_positive_positions=True,
            scale_alibi=True,
            rotary_dim=0 if model.config.rotary else None,
            rotary_interleave=False,
            parallel_residual=model.config.parallel_attn,
            shared_layer_norm=num_heads_kv == 1,
            num_heads_kv=num_heads_kv,
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.eos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.eos_token

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.word_embeddings)
        self.set_layer_norm(spec.layer_norm, module.ln_f)

        for layer_spec, layer in zip(spec.layer, module.h):
            if hasattr(layer, "ln_attn"):
                self.set_layer_norm(layer_spec.input_layer_norm, layer.ln_attn)
                self.set_layer_norm(layer_spec.post_attention_layer_norm, layer.ln_mlp)
            elif hasattr(layer_spec, "shared_layer_norm"):
                self.set_layer_norm(layer_spec.shared_layer_norm, layer.input_layernorm)
            else:
                self.set_layer_norm(
                    layer_spec.self_attention.layer_norm, layer.input_layernorm
                )
                self.set_layer_norm(
                    layer_spec.ffn.layer_norm, layer.post_attention_layernorm
                )

            num_kv = getattr(layer.self_attention, self._num_kv_attr)
            if num_kv == 1:
                self.set_linear(
                    layer_spec.self_attention.linear[0],
                    layer.self_attention.query_key_value,
                )
            else:
                self.set_qkv_linear(
                    layer_spec.self_attention.linear[0],
                    layer.self_attention.query_key_value,
                    layer.self_attention.num_heads,
                    num_kv if num_kv < layer.self_attention.num_heads else None,
                )

            self.set_linear(
                layer_spec.self_attention.linear[1], layer.self_attention.dense
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.dense_h_to_4h)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.dense_4h_to_h)

    def set_qkv_linear(self, spec, module, num_heads, num_kv=None):
        weight = module.weight

        if num_kv is None:
            weight = weight.reshape(num_heads, 3, -1, weight.shape[-1])
            weight = weight.transpose(0, 1)
            weight = weight.reshape(-1, weight.shape[-1])
        else:
            head_dim = weight.shape[0] // (num_heads + num_kv * 2)
            weight = weight.reshape(
                -1, num_heads // num_kv + 2, head_dim, weight.shape[-1]
            )
            q, k, v = weight.split([num_heads // num_kv, 1, 1], dim=1)
            weight = torch.cat(
                [
                    q.reshape(num_heads * head_dim, -1),
                    k.reshape(num_kv * head_dim, -1),
                    v.reshape(num_kv * head_dim, -1),
                ]
            )

        spec.weight = weight

        if module.bias is not None:
            bias = module.bias

            if num_kv is None:
                bias = bias.reshape(num_heads, 3, -1)
                bias = bias.transpose(0, 1)
                bias = bias.reshape(-1)
            else:
                bias = bias.reshape(-1, num_heads // num_kv + 2, head_dim)
                q, k, v = bias.split([num_heads // num_kv, 1, 1], dim=1)
                bias = torch.cat(
                    [
                        q.reshape(num_heads * head_dim),
                        k.reshape(num_kv * head_dim),
                        v.reshape(num_kv * head_dim),
                    ]
                )

            spec.bias = bias


@register_loader("FalconConfig")
class FalconLoader(RWLoader):
    def get_falcon_spec(self, model):
        self._num_layers = model.config.num_hidden_layers
        self._num_heads = model.config.num_attention_heads
        self._num_heads_kv = getattr(model.config, "num_kv_heads", None)
        self._num_kv_attr = "num_kv_heads"


@register_loader("DistilBertConfig")
class DistilBertLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "DistilBertModel"

    def get_model_spec(self, model):
        encoder_spec = transformer_spec.TransformerEncoderSpec(
            model.config.n_layers,
            model.config.n_heads,
            pre_norm=False,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation],
            layernorm_embedding=True,
        )
        spec = transformer_spec.TransformerEncoderModelSpec(
            encoder_spec,
        )

        spec.encoder.scale_embeddings = False

        self.set_embeddings(
            spec.encoder.embeddings[0], model.embeddings.word_embeddings
        )
        self.set_position_encodings(
            spec.encoder.position_encodings, model.embeddings.position_embeddings
        )
        self.set_layer_norm(
            spec.encoder.layernorm_embedding, model.embeddings.LayerNorm
        )

        for layer_spec, layer in zip(spec.encoder.layer, model.transformer.layer):
            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(split_layers[0], layer.attention.q_lin)
            self.set_linear(split_layers[1], layer.attention.k_lin)
            self.set_linear(split_layers[2], layer.attention.v_lin)
            utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)

            self.set_linear(
                layer_spec.self_attention.linear[1], layer.attention.out_lin
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.sa_layer_norm
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.ffn.lin1)
            self.set_linear(layer_spec.ffn.linear_1, layer.ffn.lin2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.output_layer_norm)

        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = 1e-12


@register_loader("BertConfig")
class BertLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "BertModel"

    def get_model_spec(self, model):
        assert model.config.position_embedding_type == "absolute"

        encoder_spec = transformer_spec.TransformerEncoderSpec(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=False,
            activation=_SUPPORTED_ACTIVATIONS[model.config.hidden_act],
            layernorm_embedding=True,
            num_source_embeddings=2,
            embeddings_merge=common_spec.EmbeddingsMerge.ADD,
        )

        spec = transformer_spec.TransformerEncoderModelSpec(
            encoder_spec,
            pooling_layer=True,
            pooling_activation=common_spec.Activation.Tanh,
        )

        spec.encoder.scale_embeddings = False

        self.set_embeddings(
            spec.encoder.embeddings[0], model.embeddings.word_embeddings
        )
        self.set_embeddings(
            spec.encoder.embeddings[1], model.embeddings.token_type_embeddings
        )
        self.set_position_encodings(
            spec.encoder.position_encodings, model.embeddings.position_embeddings
        )
        self.set_layer_norm(
            spec.encoder.layernorm_embedding, model.embeddings.LayerNorm
        )

        self.set_linear(spec.pooler_dense, model.pooler.dense)

        for layer_spec, layer in zip(spec.encoder.layer, model.encoder.layer):
            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(split_layers[0], layer.attention.self.query)
            self.set_linear(split_layers[1], layer.attention.self.key)
            self.set_linear(split_layers[2], layer.attention.self.value)
            utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)

            self.set_linear(
                layer_spec.self_attention.linear[1], layer.attention.output.dense
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.attention.output.LayerNorm
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.intermediate.dense)
            self.set_linear(layer_spec.ffn.linear_1, layer.output.dense)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.output.LayerNorm)

        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        extra_ids = model.config.vocab_size - len(tokens)
        for i in range(extra_ids):
            tokens.append("<extra_id_%d>" % i)

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = model.config.layer_norm_eps


@register_loader("XLMRobertaConfig")
class XLMRobertaLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "XLMRobertaForSequenceClassification"

    def get_model_spec(self, model):
        assert model.config.position_embedding_type == "absolute"

        encoder_spec = transformer_spec.TransformerEncoderSpec(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=False,
            activation=_SUPPORTED_ACTIVATIONS[model.config.hidden_act],
            layernorm_embedding=True,
            num_source_embeddings=2,
            embeddings_merge=common_spec.EmbeddingsMerge.ADD,
        )

        if model.roberta.pooler is None:
            pooling_layer = False
        else:
            pooling_layer = True

        spec = transformer_spec.TransformerEncoderModelSpec(
            encoder_spec,
            pooling_layer=pooling_layer,
            pooling_activation=common_spec.Activation.Tanh,
        )

        spec.encoder.scale_embeddings = False

        self.set_embeddings(
            spec.encoder.embeddings[0], model.roberta.embeddings.word_embeddings
        )
        self.set_embeddings(
            spec.encoder.embeddings[1], model.roberta.embeddings.token_type_embeddings
        )
        self.set_position_encodings(
            spec.encoder.position_encodings,
            model.roberta.embeddings.position_embeddings,
        )
        self.set_layer_norm(
            spec.encoder.layernorm_embedding, model.roberta.embeddings.LayerNorm
        )
        if pooling_layer:
            self.set_linear(spec.pooler_dense, model.roberta.pooler.dense)

        for layer_spec, layer in zip(spec.encoder.layer, model.roberta.encoder.layer):
            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(split_layers[0], layer.attention.self.query)
            self.set_linear(split_layers[1], layer.attention.self.key)
            self.set_linear(split_layers[2], layer.attention.self.value)
            utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)

            self.set_linear(
                layer_spec.self_attention.linear[1], layer.attention.output.dense
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.attention.output.LayerNorm
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.intermediate.dense)
            self.set_linear(layer_spec.ffn.linear_1, layer.output.dense)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.output.LayerNorm)

        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = model.config.layer_norm_eps

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weight
        offset = getattr(module, "padding_idx", 0)
        if offset > 0:
            spec.encodings = spec.encodings[offset + 1 :]


@register_loader("RobertaConfig")
class RobertaLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "RobertaModel"

    def get_model_spec(self, model):
        assert model.config.position_embedding_type == "absolute"

        encoder_spec = transformer_spec.TransformerEncoderSpec(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=False,
            activation=_SUPPORTED_ACTIVATIONS[model.config.hidden_act],
            layernorm_embedding=True,
            num_source_embeddings=2,
            embeddings_merge=common_spec.EmbeddingsMerge.ADD,
        )

        if model.pooler is None:
            pooling_layer = False
        else:
            pooling_layer = True

        spec = transformer_spec.TransformerEncoderModelSpec(
            encoder_spec,
            pooling_layer=pooling_layer,
            pooling_activation=common_spec.Activation.Tanh,
        )

        spec.encoder.scale_embeddings = False

        self.set_embeddings(
            spec.encoder.embeddings[0], model.embeddings.word_embeddings
        )
        self.set_embeddings(
            spec.encoder.embeddings[1], model.embeddings.token_type_embeddings
        )
        self.set_position_encodings(
            spec.encoder.position_encodings,
            model.embeddings.position_embeddings,
        )
        self.set_layer_norm(
            spec.encoder.layernorm_embedding, model.embeddings.LayerNorm
        )
        if pooling_layer:
            self.set_linear(spec.pooler_dense, model.pooler.dense)

        for layer_spec, layer in zip(spec.encoder.layer, model.encoder.layer):
            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(split_layers[0], layer.attention.self.query)
            self.set_linear(split_layers[1], layer.attention.self.key)
            self.set_linear(split_layers[2], layer.attention.self.value)
            utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)

            self.set_linear(
                layer_spec.self_attention.linear[1], layer.attention.output.dense
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.attention.output.LayerNorm
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.intermediate.dense)
            self.set_linear(layer_spec.ffn.linear_1, layer.output.dense)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.output.LayerNorm)

        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = model.config.layer_norm_eps

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weight
        offset = getattr(module, "padding_idx", 0)
        if offset > 0:
            spec.encodings = spec.encodings[offset + 1 :]


@register_loader("CamembertConfig")
class CamembertLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "CamembertModel"

    def get_model_spec(self, model):
        assert model.config.position_embedding_type == "absolute"

        encoder_spec = transformer_spec.TransformerEncoderSpec(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            pre_norm=False,
            activation=_SUPPORTED_ACTIVATIONS[model.config.hidden_act],
            layernorm_embedding=True,
            num_source_embeddings=2,
            embeddings_merge=common_spec.EmbeddingsMerge.ADD,
        )

        if model.pooler is None:
            pooling_layer = False
        else:
            pooling_layer = True

        spec = transformer_spec.TransformerEncoderModelSpec(
            encoder_spec,
            pooling_layer=pooling_layer,
            pooling_activation=common_spec.Activation.Tanh,
        )

        spec.encoder.scale_embeddings = False

        self.set_embeddings(
            spec.encoder.embeddings[0], model.embeddings.word_embeddings
        )
        self.set_embeddings(
            spec.encoder.embeddings[1], model.embeddings.token_type_embeddings
        )
        self.set_position_encodings(
            spec.encoder.position_encodings,
            model.embeddings.position_embeddings,
        )
        self.set_layer_norm(
            spec.encoder.layernorm_embedding, model.embeddings.LayerNorm
        )
        if pooling_layer:
            self.set_linear(spec.pooler_dense, model.pooler.dense)

        for layer_spec, layer in zip(spec.encoder.layer, model.encoder.layer):
            split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(split_layers[0], layer.attention.self.query)
            self.set_linear(split_layers[1], layer.attention.self.key)
            self.set_linear(split_layers[2], layer.attention.self.value)
            utils.fuse_linear(layer_spec.self_attention.linear[0], split_layers)

            self.set_linear(
                layer_spec.self_attention.linear[1], layer.attention.output.dense
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.attention.output.LayerNorm
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.intermediate.dense)
            self.set_linear(layer_spec.ffn.linear_1, layer.output.dense)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.output.LayerNorm)

        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.unk_token = tokenizer.unk_token
        config.layer_norm_epsilon = model.config.layer_norm_eps

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weight
        offset = getattr(module, "padding_idx", 0)
        if offset > 0:
            spec.encodings = spec.encodings[offset + 1 :]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Name of the pretrained model to download, "
            "or path to a directory containing the pretrained model."
        ),
    )
    parser.add_argument(
        "--activation_scales",
        help=(
            "Path to the pre-computed activation scales. Models may "
            "use them to rescale some weights to smooth the intermediate activations "
            "and improve the quantization accuracy. See "
            "https://github.com/mit-han-lab/smoothquant."
        ),
    )
    parser.add_argument(
        "--copy_files",
        nargs="+",
        help=(
            "List of filenames to copy from the Hugging Face model to the converted "
            "model directory."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Revision of the model to download from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Enable the flag low_cpu_mem_usage when loading the model with from_pretrained.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow converting models using custom code.",
    )

    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = TransformersConverter(
        args.model,
        activation_scales=args.activation_scales,
        copy_files=args.copy_files,
        load_as_float16=args.quantization in ("float16", "int8_float16"),
        revision=args.revision,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()


# Cross-attention heads that are highly correlated to the word-level timing,
# i.e. the alignment between audio and text tokens.
# Obtained from https://github.com/openai/whisper/blob/v20231106/whisper/__init__.py#L32-L47
_WHISPER_ALIGNMENT_HEADS = {
    "openai/whisper-tiny.en": [
        (1, 0),
        (2, 0),
        (2, 5),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
    ],
    "openai/whisper-tiny": [(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
    "openai/whisper-base.en": [(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)],
    "openai/whisper-base": [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ],
    "openai/whisper-small.en": [
        (6, 6),
        (7, 0),
        (7, 3),
        (7, 8),
        (8, 2),
        (8, 5),
        (8, 7),
        (9, 0),
        (9, 4),
        (9, 8),
        (9, 10),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 6),
        (10, 11),
        (11, 2),
        (11, 4),
    ],
    "openai/whisper-small": [
        (5, 3),
        (5, 9),
        (8, 0),
        (8, 4),
        (8, 7),
        (8, 8),
        (9, 0),
        (9, 7),
        (9, 9),
        (10, 5),
    ],
    "openai/whisper-medium.en": [
        (11, 4),
        (14, 1),
        (14, 12),
        (14, 14),
        (15, 4),
        (16, 0),
        (16, 4),
        (16, 9),
        (17, 12),
        (17, 14),
        (18, 7),
        (18, 10),
        (18, 15),
        (20, 0),
        (20, 3),
        (20, 9),
        (20, 14),
        (21, 12),
    ],
    "openai/whisper-medium": [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
    "openai/whisper-large": [
        (9, 19),
        (11, 2),
        (11, 4),
        (11, 17),
        (22, 7),
        (22, 11),
        (22, 17),
        (23, 2),
        (23, 15),
    ],
    "openai/whisper-large-v2": [
        (10, 12),
        (13, 17),
        (16, 11),
        (16, 12),
        (16, 13),
        (17, 15),
        (17, 16),
        (18, 4),
        (18, 11),
        (18, 19),
        (19, 11),
        (21, 2),
        (21, 3),
        (22, 3),
        (22, 9),
        (22, 12),
        (23, 5),
        (23, 7),
        (23, 13),
        (25, 5),
        (26, 1),
        (26, 12),
        (27, 15),
    ],
    "openai/whisper-large-v3": [
        (7, 0),
        (10, 17),
        (12, 18),
        (13, 12),
        (16, 1),
        (17, 14),
        (19, 11),
        (21, 4),
        (24, 1),
        (25, 6),
    ],
}


# Paper: https://arxiv.org/pdf/2504.06225
@register_loader("T5GemmaConfig")
class T5GemmaLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "T5GemmaForConditionalGeneration"

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight.data + 1.0

    def get_model_spec(self, model):
        encoder_config = model.config.encoder
        decoder_config = model.config.decoder
        sliding_window = getattr(model.config, "sliding_window", 4096)

        encoder_num_heads = encoder_config.num_attention_heads
        encoder_num_heads_kv = getattr(
            encoder_config, "num_key_value_heads", encoder_num_heads
        )
        if encoder_num_heads_kv == encoder_num_heads:
            encoder_num_heads_kv = None

        encoder = transformer_spec.TransformerEncoderSpec(
            encoder_config.num_hidden_layers,
            encoder_config.num_attention_heads,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[encoder_config.hidden_activation],
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=encoder_config.head_dim,
            rotary_interleave=False,
            rotary_base=getattr(encoder_config, "rope_theta", 10000),
            sliding_window=sliding_window,
            pre_post_layer_norm=True,
            num_heads_kv=encoder_num_heads_kv,
            head_dim=encoder_config.head_dim,
        )

        decoder_num_heads = decoder_config.num_attention_heads
        decoder_num_heads_kv = getattr(
            decoder_config, "num_key_value_heads", decoder_num_heads
        )
        if decoder_num_heads_kv == decoder_num_heads:
            decoder_num_heads_kv = None

        decoder = transformer_spec.TransformerDecoderSpec(
            decoder_config.num_hidden_layers,
            decoder_config.num_attention_heads,
            pre_norm=True,
            activation=_SUPPORTED_ACTIVATIONS[decoder_config.hidden_activation],
            ffn_glu=True,
            rms_norm=True,
            with_encoder_attention=True,
            rotary_dim=decoder_config.head_dim,
            rotary_interleave=False,
            rotary_base=getattr(decoder_config, "rope_theta", 10000),
            sliding_window=sliding_window,
            pre_post_layer_norm=True,
            external_pre_post_encoder_layers=True,
            num_heads_kv=decoder_num_heads_kv,
            head_dim=decoder_config.head_dim,
        )

        spec = transformer_spec.TransformerSpec(encoder, decoder)

        self.set_encoder(spec.encoder, model.model.encoder, encoder_config)

        self.set_decoder(
            spec.decoder,
            model.model.decoder,
            decoder_config,
            common_spec.Quantization.CT2,
        )

        # Tie_word_embeddings
        self.set_linear(spec.decoder.projection, model.model.decoder.embed_tokens)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_source_vocabulary(tokens)
        spec.register_target_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

        if hasattr(model.config, "encoder"):
            config.layer_norm_epsilon = model.config.encoder.rms_norm_eps
        elif hasattr(model.config, "rms_norm_eps"):
            config.layer_norm_epsilon = model.config.rms_norm_eps
        else:
            config.layer_norm_epsilon = 1e-6

        config.decoder_start_token = tokenizer.bos_token

    def set_encoder(
        self, spec, encoder, encoder_config, quant_type=common_spec.Quantization.CT2
    ):
        spec.scale_embeddings = True

        encoder_emb_spec = (
            spec.embeddings[0] if isinstance(spec.embeddings, list) else spec.embeddings
        )

        self.set_embeddings(encoder_emb_spec, encoder.embed_tokens)
        encoder_emb_spec.multiply_by_sqrt_depth = encoder_config.hidden_size**0.5
        self.set_layer_norm(spec.layer_norm, encoder.norm)

        module = encoder
        for i, (layer_spec, layer) in enumerate(zip(spec.layer, module.layers)):
            self.set_layer_norm(
                layer_spec.input_layer_norm, layer.pre_self_attn_layernorm
            )
            self.set_layer_norm(
                layer_spec.post_attention_layer_norm, layer.post_self_attn_layernorm
            )

            # T5GemmaSelfAttention
            qkv_split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(
                qkv_split_layers[0], layer.self_attn.q_proj, quant_type=quant_type
            )
            self.set_linear(
                qkv_split_layers[1], layer.self_attn.k_proj, quant_type=quant_type
            )
            self.set_linear(
                qkv_split_layers[2], layer.self_attn.v_proj, quant_type=quant_type
            )
            utils.fuse_linear(layer_spec.self_attention.linear[0], qkv_split_layers)
            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )

            # T5GemmaRMSNorm
            self.set_layer_norm(
                layer_spec.pre_feedforward_layer_norm, layer.pre_feedforward_layernorm
            )
            # T5GemmaRMSNorm
            self.set_layer_norm(
                layer_spec.post_feedforward_layer_norm, layer.post_feedforward_layernorm
            )

            # T5GemmaMLP
            self.set_linear(
                layer_spec.ffn.linear_0, layer.mlp.gate_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_0_noact, layer.mlp.up_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            # Clean up
            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()

    def set_decoder(
        self, spec, module, decoder_config, quant_type=common_spec.Quantization.CT2
    ):
        spec.scale_embeddings = True
        spec.start_from_zero_embedding = False

        self.set_embeddings(spec.embeddings, module.embed_tokens)
        spec.embeddings.multiply_by_sqrt_depth = decoder_config.hidden_size**0.5
        self.set_layer_norm(spec.layer_norm, module.norm)

        for i, (layer_spec, layer) in enumerate(zip(spec.layer, module.layers)):
            # Self-attention block
            self.set_layer_norm(
                layer_spec.input_layer_norm, layer.pre_self_attn_layernorm
            )
            self.set_layer_norm(
                layer_spec.post_attention_layer_norm, layer.post_self_attn_layernorm
            )

            # T5GemmaSelfAttention - QKV projections
            qkv_split_layers = [common_spec.LinearSpec() for _ in range(3)]
            self.set_linear(
                qkv_split_layers[0], layer.self_attn.q_proj, quant_type=quant_type
            )
            self.set_linear(
                qkv_split_layers[1], layer.self_attn.k_proj, quant_type=quant_type
            )
            self.set_linear(
                qkv_split_layers[2], layer.self_attn.v_proj, quant_type=quant_type
            )
            utils.fuse_linear(layer_spec.self_attention.linear[0], qkv_split_layers)
            self.set_linear(
                layer_spec.self_attention.linear[1],
                layer.self_attn.o_proj,
                quant_type=quant_type,
            )

            # Pre and post cross-attention layer norm
            self.set_layer_norm(
                layer_spec.external_pre_encoder_attention_layer_norm,
                layer.pre_cross_attn_layernorm,
            )

            self.set_layer_norm(
                layer_spec.external_post_encoder_attention_layer_norm,
                layer.post_cross_attn_layernorm,
            )

            # Cross-attention Q projection
            self.set_linear(
                layer_spec.attention.linear[0],
                layer.cross_attn.q_proj,
                quant_type=quant_type,
            )

            # Cross-attention K+V fused
            kv_split_layers = [common_spec.LinearSpec() for _ in range(2)]
            self.set_linear(
                kv_split_layers[0],
                layer.cross_attn.k_proj,
                quant_type=quant_type,
            )
            self.set_linear(
                kv_split_layers[1],
                layer.cross_attn.v_proj,
                quant_type=quant_type,
            )
            utils.fuse_linear(layer_spec.attention.linear[1], kv_split_layers)

            # Cross-attention output projection
            self.set_linear(
                layer_spec.attention.linear[2],
                layer.cross_attn.o_proj,
                quant_type=quant_type,
            )

            # Feed-forward block
            self.set_layer_norm(
                layer_spec.pre_feedforward_layer_norm, layer.pre_feedforward_layernorm
            )
            self.set_layer_norm(
                layer_spec.post_feedforward_layer_norm, layer.post_feedforward_layernorm
            )

            # T5GemmaMLP
            self.set_linear(
                layer_spec.ffn.linear_0, layer.mlp.gate_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_0_noact, layer.mlp.up_proj, quant_type=quant_type
            )
            self.set_linear(
                layer_spec.ffn.linear_1, layer.mlp.down_proj, quant_type=quant_type
            )

            # Clean up
            delattr(layer, "self_attn")
            delattr(layer, "cross_attn")
            delattr(layer, "mlp")
            gc.collect()

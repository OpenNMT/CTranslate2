import argparse
import json
import re

import numpy as np

from safetensors.torch import safe_open

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import (
    TransformerDecoderSpec,
    TransformerEncoderSpec,
    TransformerSpec,
)
from ctranslate2.specs.common_spec import Activation
from ctranslate2.specs.moonshine_spec import MoonshineSpec


class MoonshineConverter(Converter):
    def __init__(self, safetensor_file, vocab_file, moonshine_variant):
        self.safetensor_file = safetensor_file
        self.vocab_file = vocab_file
        if moonshine_variant == "tiny":
            self.layers = 6
            self.heads = 8
        elif moonshine_variant == "base":
            self.layers = 8
            self.heads = 8
        else:
            raise ValueError('moonshine_variant must be one of ["tiny", "base"]')

    def _load(self):
        spec = MoonshineSpec(
            num_encoder_layers=self.layers,
            num_encoder_heads=self.heads,
            num_decoder_layers=self.layers,
            num_decoder_heads=self.heads,
        )
        self.load_preprocessor(spec.preprocessor)
        self.load_encoder(spec.encoder)
        self.load_decoder(spec.decoder)
        spec.register_vocabulary(self.load_vocab())
        return spec

    def load_vocab(self):
        tokens_dict = {}
        with open(self.vocab_file, encoding="utf-8") as f:
            tokenizer_dict = json.load(f)
            d = tokenizer_dict["model"]["vocab"]
            for token in d.keys():
                idx = d[token]
                token = re.sub(r"\\([^x])", r"\1", token)
                token = token[1:-1]
                if token.startswith("\\x"):
                    # Convert the digraph \x to the actual escaped sequence.
                    token = chr(int(token[2:], base=16))
                elif token.startswith("'") and token.endswith("'"):
                    token = token[1:-1]
                    token = token.replace("''", "'")
                if idx is not None:
                    tokens_dict[idx] = token
            added_tokens = tokenizer_dict["added_tokens"]
            for t in added_tokens:
                tokens_dict[t["id"]] = t["content"]

        return [tokens_dict[idx] for idx in sorted(tokens_dict.keys())]

    def load_attention(self, att_spec, st_prefix, self_attention=True):
        st = safe_open(self.safetensor_file, framework="pt", device="cpu")
        attn_w = [
            st.get_tensor(f"{st_prefix}.to_{dst}.weight") for dst in ["q", "k", "v"]
        ]
        if self_attention:
            att_spec.linear[0].weight = np.concatenate(attn_w)
        else:
            att_spec.linear[0].weight = attn_w[0]
            att_spec.linear[1].weight = np.concatenate(attn_w[1:])
        att_spec.linear[-1].weight = st.get_tensor(f"{st_prefix}.to_out.weight")

    def load_ffn(self, ffn_spec, st_prefix, swiglu=False):
        st = safe_open(self.safetensor_file, framework="pt", device="cpu")
        if swiglu:
            ffn_spec.linear_0_noact.weight = st.get_tensor(
                f"{st_prefix}.ff_noact.weight"
            )
            ffn_spec.linear_0.weight = st.get_tensor(f"{st_prefix}.ff_proj.weight")
            ffn_spec.linear_0_noact.bias = st.get_tensor(f"{st_prefix}.ff_noact.bias")
            ffn_spec.linear_0.bias = st.get_tensor(f"{st_prefix}.ff_proj.bias")
            ffn_spec.linear_1.weight = st.get_tensor(f"{st_prefix}.ff_out.weight")
            ffn_spec.linear_1.bias = st.get_tensor(f"{st_prefix}.ff_out.bias")
        else:
            ffn_spec.linear_0.weight = st.get_tensor(f"{st_prefix}.ff.0.weight")
            ffn_spec.linear_0.bias = st.get_tensor(f"{st_prefix}.ff.0.bias")
            ffn_spec.linear_1.weight = st.get_tensor(f"{st_prefix}.ff.2.weight")
            ffn_spec.linear_1.bias = st.get_tensor(f"{st_prefix}.ff.2.bias")

    def load_layernorm(self, ln_spec, ln_prefix):
        st = safe_open(self.safetensor_file, framework="pt", device="cpu")
        ln_spec.gamma = st.get_tensor(f"{ln_prefix}.weight")
        ln_spec.beta = np.zeros(ln_spec.gamma.shape)

    def load_embeddings(self, embedding_spec, embedding_prefix):
        st = safe_open(self.safetensor_file, framework="pt", device="cpu")
        embedding_spec.weight = st.get_tensor(f"{embedding_prefix}.weight")

    def load_preprocessor(self, preprocess_spec):
        st = safe_open(self.safetensor_file, framework="pt", device="cpu")
        preprocess_prefix = "model.preprocessor.audio_preprocess"
        preprocess_spec.conv1.weight = st.get_tensor(f"{preprocess_prefix}.0.weight")
        preprocess_spec.layernorm.gamma = st.get_tensor(f"{preprocess_prefix}.2.weight")
        preprocess_spec.layernorm.beta = st.get_tensor(f"{preprocess_prefix}.2.bias")
        preprocess_spec.conv2.weight = st.get_tensor(f"{preprocess_prefix}.3.weight")
        preprocess_spec.conv2.bias = st.get_tensor(f"{preprocess_prefix}.3.bias")
        preprocess_spec.conv3.weight = st.get_tensor(f"{preprocess_prefix}.5.weight")
        preprocess_spec.conv3.bias = st.get_tensor(f"{preprocess_prefix}.5.bias")

    def load_encoder(self, encoder_spec):
        self.load_layernorm(encoder_spec.layer_norm, "model.encoder.post_norm")
        for idx, l in enumerate(encoder_spec.layer):
            self.load_attention(
                l.self_attention, f"model.encoder.layers.{idx}.attention"
            )
            self.load_layernorm(
                l.self_attention.layer_norm, f"model.encoder.layers.{idx}.norm1"
            )
            self.load_ffn(l.ffn, f"model.encoder.layers.{idx}.ff")
            self.load_layernorm(l.ffn.layer_norm, f"model.encoder.layers.{idx}.norm2")

    def load_decoder(self, decoder_spec):
        self.load_layernorm(decoder_spec.layer_norm, "model.decoder.final_norm")
        self.load_embeddings(decoder_spec.embeddings, "model.decoder.token_embedding")
        decoder_spec.projection.weight = decoder_spec.embeddings.weight
        for idx, l in enumerate(decoder_spec.layer):
            self.load_attention(
                l.self_attention, f"model.decoder.layers.{idx}.self_attention"
            )
            self.load_layernorm(
                l.self_attention.layer_norm, f"model.decoder.layers.{idx}.norm1"
            )
            self.load_attention(
                l.attention,
                f"model.decoder.layers.{idx}.cross_attention",
                self_attention=False,
            )
            self.load_layernorm(
                l.attention.layer_norm, f"model.decoder.layers.{idx}.norm2"
            )
            self.load_ffn(l.ffn, f"model.decoder.layers.{idx}.ff", swiglu=True)
            self.load_layernorm(l.ffn.layer_norm, f"model.decoder.layers.{idx}.norm3")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the model .safetensor file."
    )
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="Path to tokenizer.json config file.",
    )
    parser.add_argument(
        "--moonshine_variant",
        required=True,
        help="Moonshine variant to convert. Must be one of ['tiny', 'base']",
    )

    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = MoonshineConverter(
        args.model_path, args.vocab_path, args.moonshine_variant
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()

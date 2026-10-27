"""Regression tests for Drama3 tensor mapping and conversion."""
import json
import tempfile
import unittest

from pathlib import Path

import numpy as np

from drama_test_utils import shard

from ctranslate2.converters import drama3_weights
from ctranslate2.converters.transformers import Drama3Loader as converter
from ctranslate2.specs.drama3_spec import Drama3Spec


def config():
    return {
        "audio_decoder_config": {"n_head": 2, "n_local_heads": 1, "head_dim": 2},
        "quantization_config": {
            "weight_block_size": [128, 128],
            "quant_method": "bsgemm-moe",
        },
    }


def optimized_variables(variables):
    spec = Drama3Spec(variables)
    spec.validate()
    spec.optimize(quantization="float32")
    values = spec.variables()

    def resolve(name):
        value = values[name]
        return resolve(value) if isinstance(value, str) else value.numpy()

    return {name: resolve(name) for name in values}


class DramaModelConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_explicit_backbone_and_tower_names(self):
        cases = {
            "model.language_model.layers.3.mlp.gate.weight": "backbone/layers_3/mlp/router/weight",
            "model.language_model.layers.0.mlp.experts.12.up_proj.weight": (
                "backbone/layers_0/mlp/experts_12/up_proj/weight"
            ),
            "model.language_model.embed_tokens.weight": "backbone/embeddings/weight",
            "model.language_model.norm.weight": "backbone/norm/weight",
            "lm_head.weight": "backbone/lm_head/weight",
            "audio_decoder.codebook_embeddings.weight": "tower/codebook_embeddings/weight",
            "audio_decoder_mtp_shared.layers.0.transformer_layer.attention.wo.weight": (
                "tower/shared_layers_0/transformer_layer/attention/o_proj/weight"
            ),
            "audio_decoder_mtp.layers.2.transformer_layer.feed_forward.w1.weight": (
                "tower/residual_layers_2/transformer_layer/feed_forward/gate_proj/weight"
            ),
            "audio_decoder_mtp.layers.2.transformer_layer.feed_forward.w3.weight": (
                "tower/residual_layers_2/transformer_layer/feed_forward/up_proj/weight"
            ),
            "audio_decoder_mtp.layers.2.transformer_layer.feed_forward.w2.weight": (
                "tower/residual_layers_2/transformer_layer/feed_forward/down_proj/weight"
            ),
            "audio_decoder_mtp.heads.8.weight": "tower/heads_8/weight",
        }
        for source, target in cases.items():
            self.assertEqual(converter.canonical_name(source), target)
        with self.assertRaises(ValueError):
            converter.canonical_name("unknown.weight")

    def test_bf16_tower_row_split_across_chunk_boundaries(self):
        name = (
            "audio_decoder_mtp_shared.layers.0.transformer_layer.attention.wqkv.weight"
        )
        original = np.arange(36, dtype=np.float32).reshape(12, 3)
        bf16 = (original.view(np.uint32) >> 16).astype("<u2")
        path = self.root / "tower.safetensors"
        shard(path, {name: (bf16, "BF16")})
        reader = drama3_weights.SafeTensorMatrixReader(path)
        for kind, rows in (
            ("q_proj", [0, 1, 4, 5, 2, 3, 6, 7]),
            ("k_proj", [8, 9]),
            ("v_proj", [10, 11]),
        ):
            item = {"source": name, "tower_slice": kind, "shape": (len(rows), 2)}
            chunks = list(
                converter.tower_matrix_chunks(
                    reader, item, config()["audio_decoder_config"], chunk_rows=3
                )
            )
            self.assertTrue(all(len(chunk) <= 3 for chunk in chunks))
            np.testing.assert_array_equal(np.concatenate(chunks), original[rows])
        metadata = {name: {**reader.tensors[name], "shard": path.name}}
        plan = converter.build_plan(metadata, config())
        self.assertEqual([item["shape"] for item in plan], [[8, 3], [2, 3], [2, 3]])
        loaded = converter.load_model_tensors(
            plan, {path.name: reader}, config(), chunk_rows=3
        )
        prefix = "tower/shared_layers_0/transformer_layer/attention/"
        np.testing.assert_array_equal(
            loaded[prefix + "q_proj/weight"], original[[0, 1, 4, 5, 2, 3, 6, 7]]
        )
        output = optimized_variables(loaded)
        prefix = "tower/shared_layers_0/transformer_layer/attention/"
        np.testing.assert_array_equal(
            output[prefix + "q_proj/weight"], original[[0, 1, 4, 5, 2, 3, 6, 7]]
        )
        self.assertNotIn(prefix + "q_proj/weight_scale", output)

    def test_cross_shard_fp8_bytes_and_scales_are_preserved_exactly(self):
        name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        scale_name = name + "_scale_inv"
        weight_path, scale_path = (
            self.root / "weights.safetensors",
            self.root / "scales.safetensors",
        )
        shard(weight_path, {name: (np.full((129, 129), 0x38, np.uint8), "F8_E4M3FN")})
        scales = np.array([[0.5, 1], [2, 4]], np.float32)
        shard(scale_path, {scale_name: (scales, "F32")})
        readers = {
            p.name: drama3_weights.SafeTensorMatrixReader(p)
            for p in (weight_path, scale_path)
        }
        tensors = {
            n: {**t, "shard": p}
            for p, r in readers.items()
            for n, t in r.tensors.items()
        }
        plan = converter.build_plan(tensors, config())
        self.assertEqual(plan[0]["scale_shard"], scale_path.name)
        loaded = converter.load_model_tensors(plan, readers, config(), chunk_rows=1)
        output = optimized_variables(loaded)
        target = "backbone/layers_0/mlp/experts_0/gate_proj/weight"
        np.testing.assert_array_equal(
            output[target].view(np.uint8), np.full((129, 129), 0x38, np.uint8)
        )
        np.testing.assert_array_equal(output[target + "_scale_inv"], scales)
        self.assertNotIn(target + "_scale", output)

    def test_missing_or_wrong_source_scale_rejected(self):
        name = "lm_head.weight"
        tensors = {name: {"shape": (4, 4), "dtype": "F8_E4M3FN", "shard": "a"}}
        with self.assertRaisesRegex(ValueError, "explicit inverse scale"):
            converter.build_plan(tensors, config())
        tensors[name + "_scale_inv"] = {"shape": (2, 2), "dtype": "F32", "shard": "b"}
        with self.assertRaisesRegex(ValueError, "shape/type"):
            converter.build_plan(tensors, config())
        tensors[name]["dtype"] = "BF16"
        with self.assertRaisesRegex(ValueError, "non-FP8"):
            converter.build_plan(tensors, config())

    def test_vectors_and_convolutions_keep_float32_and_normalization_offsets(self):
        name = "model.language_model.layers.0.linear_attn.norm.weight"
        path = self.root / "vectors.safetensors"
        values = np.array([-0.5, 0.3], dtype="<f4")
        shard(path, {name: (values, "F32")})
        reader = drama3_weights.SafeTensorMatrixReader(path)
        metadata = {name: {**reader.tensors[name], "shard": path.name}}
        plan = converter.build_plan(metadata, config())
        self.assertEqual(plan[0]["dtype"], "float32")
        output = converter.load_model_tensors(plan, {path.name: reader}, config())
        np.testing.assert_array_equal(
            output["backbone/layers_0/linear_attn/norm/weight"], values
        )

    def test_depthwise_convolution_preserves_rank_and_flat_chunk_contents(self):
        name = "model.language_model.layers.0.linear_attn.conv1d.weight"
        original = np.arange(24, dtype=np.float32).reshape(6, 1, 4)
        path = self.root / "convolution.safetensors"
        shard(path, {name: (original, "F32")})
        reader = drama3_weights.SafeTensorMatrixReader(path)
        metadata = {name: {**reader.tensors[name], "shard": path.name}}
        plan = converter.build_plan(metadata, config())
        loaded = converter.load_model_tensors(plan, {path.name: reader}, config())
        target = "backbone/layers_0/linear_attn/conv1d/weight"
        np.testing.assert_array_equal(loaded[target], original)
        optimized = optimized_variables(loaded)
        np.testing.assert_array_equal(optimized[target], original)
        self.assertNotIn(target + "_scale", optimized)

    def test_index_exact_coverage_and_path_guards(self):
        name = "lm_head.weight"
        p = self.root / "model-00001-of-00001.safetensors"
        shard(p, {name: (np.ones((2, 2), np.float32), "F32")})
        index_path = self.root / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"weight_map": {name: p.name}}))
        readers, tensors = converter.inspect_shards(self.root)
        self.assertEqual(len(readers), 1)
        self.assertEqual(set(tensors), {name})
        for bad in ("../outside", "/tmp/outside", "sub/weights", "..\\outside"):
            with self.assertRaises(ValueError):
                converter.source_file(self.root, bad)
        index_path.write_text(json.dumps({"weight_map": {"missing": p.name}}))
        with self.assertRaises(ValueError):
            converter.inspect_shards(self.root)

    def test_orphan_scales_and_nonfinite_vectors_rejected(self):
        with self.assertRaisesRegex(ValueError, "orphan"):
            converter.build_plan(
                {
                    "unused.weight_scale_inv": {
                        "shape": (1, 1),
                        "dtype": "F32",
                        "shard": "a",
                    }
                },
                config(),
            )
        path = self.root / "nonfinite.safetensors"
        name = "model.language_model.norm.weight"
        shard(path, {name: (np.array([np.nan], np.float32), "F32")})
        reader = drama3_weights.SafeTensorMatrixReader(path)
        with self.assertRaises(ValueError):
            list(converter.float_chunks(reader, name))


if __name__ == "__main__":
    unittest.main()

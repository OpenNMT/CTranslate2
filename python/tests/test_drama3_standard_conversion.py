"""End-to-end checks of the standard model-spec conversion path."""
import contextlib
import io
import json
import struct
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

import numpy as np

from drama_test_utils import shard

from ctranslate2.converters import Converter, TransformersConverter, drama3_codec
from ctranslate2.converters.transformers import Drama3Loader as c
from ctranslate2.specs.drama3_spec import Drama3Spec
from ctranslate2.specs.model_spec import ACCEPTED_MODEL_TYPES


def tensor_offset(path, name):
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return 8 + header_size + header[name]["data_offsets"][0]


def read_model(path):
    with path.open("rb") as stream:

        def integer(size):
            return int.from_bytes(stream.read(size), "little")

        def string():
            return stream.read(integer(2))[:-1].decode()

        assert integer(4) == 6 and string() == "Drama3Spec" and integer(4) == 1
        values = {}
        for _ in range(integer(4)):
            name = string()
            shape = tuple(integer(4) for _ in range(integer(1)))
            dtype, size = integer(1), integer(4)
            values[name] = np.frombuffer(
                stream.read(size),
                dtype={0: "<f4", 1: "i1", 2: "<i2", 4: "<f2", 5: "<u2"}[dtype],
            ).reshape(shape)
        for _ in range(integer(4)):
            alias, target = string(), string()
            values[alias] = values[target]
        return values


class ConversionFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.output = Path(self.temp.name) / "converted"
        for name in c.ASSETS + ("codec.pth",):
            (self.source / name).write_text("fixture\n")
        config = {
            "model_type": "fish_omni",
            "audio_decoder_config": {"n_head": 1, "n_local_heads": 1, "head_dim": 2},
            "quantization_config": {
                "weight_block_size": [128, 128],
                "quant_method": "bsgemm-moe",
            },
        }
        (self.source / "config.json").write_text(json.dumps(config))
        self.expert = "model.language_model.layers.0.mlp.experts.0.up_proj.weight"
        self.sensitive = np.array([[1.0001, -0.33331], [0.12345, -0.76543]], np.float32)
        self.convolution = np.arange(6, dtype=np.float32).reshape(2, 1, 3) / np.float32(
            7
        )
        tensors = {
            self.expert: (
                np.array([[0x38, 0xB8], [0x30, 0x28]], np.uint8),
                "F8_E4M3FN",
            ),
            self.expert + "_scale_inv": (np.array([[1.0001]], np.float32), "F32"),
            "lm_head.weight": (self.sensitive, "F32"),
            "model.language_model.layers.0.linear_attn.conv1d.weight": (
                self.convolution,
                "F32",
            ),
        }
        shard_name = "model-00001-of-00001.safetensors"
        shard(self.source / shard_name, tensors)
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard_name for name in tensors}})
        )
        codec = drama3_codec
        self.codec = np.array([[0.234567, -0.345678]], np.float32)
        self.enterContext(
            patch.object(
                codec, "codec_parameter_shapes", return_value={"decoder.weight": (1, 2)}
            )
        )
        self.enterContext(
            patch.object(
                codec, "load_checkpoint", return_value={"decoder.weight": self.codec}
            )
        )


class StandardConversionTest(ConversionFixture):
    def test_float32_conversion_preserves_scaled_fp8_without_quantization(self):
        TransformersConverter(self.source).convert(
            str(self.output), quantization="float32"
        )
        values = read_model(self.output / "model.bin")
        expert = "backbone/layers_0/mlp/experts_0/up_proj/weight"
        np.testing.assert_array_equal(
            values[expert].view(np.uint8),
            np.array([[0x38, 0xB8], [0x30, 0x28]], np.uint8),
        )
        np.testing.assert_array_equal(
            values[expert + "_scale_inv"], np.array([[1.0001]], np.float32)
        )
        self.assertFalse(any(name.endswith("_scale") for name in values))

    def test_invalid_codec_is_rejected_before_loading_backbone(self):
        with patch.object(
            drama3_codec, "conversion_plan", side_effect=ValueError("invalid codec")
        ), patch.object(c, "load_model_tensors") as load:
            with self.assertRaisesRegex(ValueError, "invalid codec"):
                TransformersConverter(self.source).convert(str(self.output))
        load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_save_failure_leaves_output_directory(self):
        with patch.object(Drama3Spec, "save", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                TransformersConverter(self.source).convert(str(self.output))
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "model.bin").exists())

    def test_standard_api_and_force(self):
        converter = TransformersConverter(self.source)
        self.assertIsInstance(converter, Converter)
        report = converter.inspect()
        self.assertTrue(report["dry_run"])
        self.assertFalse(self.output.exists())
        self.assertEqual(converter.convert(str(self.output)), str(self.output))
        values = read_model(self.output / "model.bin")
        self.assertEqual(
            values["backbone/layers_0/mlp/experts_0/up_proj/weight"].dtype, "int8"
        )
        with self.assertRaises(RuntimeError):
            converter.convert(str(self.output))
        converter.convert(str(self.output), quantization="float32", force=True)
        values = read_model(self.output / "model.bin")
        self.assertFalse(any(key.endswith("_scale") for key in values))

    def test_force_save_failure_replaces_old_output(self):
        converter = TransformersConverter(self.source)
        converter.convert(str(self.output))
        with patch.object(Drama3Spec, "save", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                converter.convert(str(self.output), force=True)
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "model.bin").exists())

    def test_reject_unsupported_quantization_before_loading(self):
        for quantization in ("invalid", "auto", "int4"):
            with self.subTest(quantization=quantization), patch.object(
                TransformersConverter, "_load"
            ) as load:
                with self.assertRaisesRegex(ValueError, "quantization|float32"):
                    TransformersConverter(self.source).convert(
                        str(self.output), quantization=quantization
                    )
            load.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_all_standard_quantizations(self):
        expert = "backbone/layers_0/mlp/experts_0/up_proj/weight"
        original = np.array([[1, -1], [0.5, 0.25]], np.float32) * np.float32(1.0001)
        converter = TransformersConverter(self.source)
        for quantization in ACCEPTED_MODEL_TYPES:
            with self.subTest(quantization=quantization):
                converter.convert(
                    str(self.output), quantization=quantization, force=True
                )
                values = read_model(self.output / "model.bin")
                report = converter.conversion_report
                self.assertEqual(report["quantization"], quantization)
                self.assertEqual(
                    sum(value.nbytes for value in values.values()),
                    report["estimated_payload_bytes"],
                )
                self.assertEqual(len(values), report["variable_count"])
                if quantization == "float32":
                    self.assertIn(expert + "_scale_inv", values)
                    continue
                self.assertNotIn(expert + "_scale_inv", values)
                for name, expected in (
                    (expert, original),
                    ("backbone/lm_head/weight", self.sensitive),
                ):
                    if quantization.startswith("int8"):
                        scales = np.float32(127) / np.max(np.abs(expected), axis=1)
                        np.testing.assert_array_equal(
                            values[name],
                            np.rint(expected * scales[:, None]).astype(np.int8),
                        )
                        np.testing.assert_array_equal(values[name + "_scale"], scales)
                    elif quantization == "int16":
                        scale = np.float32(1024 / np.max(np.abs(expected)))
                        np.testing.assert_array_equal(
                            values[name], np.rint(expected * scale).astype(np.int16)
                        )
                        np.testing.assert_array_equal(values[name + "_scale"], scale)
                    elif quantization == "float16":
                        np.testing.assert_array_equal(
                            values[name], expected.astype(np.float16)
                        )
                    else:
                        import torch

                        np.testing.assert_array_equal(
                            values[name],
                            torch.from_numpy(expected)
                            .bfloat16()
                            .view(torch.uint16)
                            .numpy(),
                        )
                np.testing.assert_array_equal(
                    values["codec/decoder/weight"], self.codec
                )
                np.testing.assert_array_equal(
                    values["backbone/layers_0/linear_attn/conv1d/weight"],
                    self.convolution,
                )
                self.assertEqual(
                    json.loads((self.output / "config.json").read_text())[
                        "quantization_policy"
                    ],
                    c.STANDARD_POLICY,
                )
        converter.convert(str(self.output), force=True)
        self.assertIn(expert + "_scale_inv", read_model(self.output / "model.bin"))

    def test_drama3_loader_builds_spec_through_base_loader(self):
        loader = c(quantization="int8")
        spec = loader(self.source, None)
        np.testing.assert_array_equal(
            spec.variables()["backbone/lm_head/weight"], self.sensitive
        )
        self.assertEqual(loader.conversion_report["quantization"], "int8")
        self.assertFalse(loader.conversion_report["dry_run"])
        self.assertEqual(spec.config.model_type, "drama3")

    def test_transformers_converter_dispatches_local_drama3(self):
        from ctranslate2.converters import TransformersConverter

        converter = TransformersConverter(str(self.source))
        self.assertTrue(converter.inspect()["dry_run"])
        converter.convert(str(self.output), quantization="float32")
        values = read_model(self.output / "model.bin")
        np.testing.assert_array_equal(values["backbone/lm_head/weight"], self.sensitive)
        self.assertEqual(converter.conversion_report["model_name"], "Drama3Spec")

    def test_repeated_conversion_preserves_default_quantization(self):
        converter = TransformersConverter(self.source)
        converter.convert(str(self.output), quantization="float32")
        converter.convert(str(self.output), force=True)
        values = read_model(self.output / "model.bin")
        expert = "backbone/layers_0/mlp/experts_0/up_proj/weight"
        self.assertIn(expert + "_scale_inv", values)
        self.assertNotIn(expert + "_scale", values)
        self.assertEqual(converter.conversion_report["quantization"], "float32")

    def test_transformers_cli_conversion_writes_report(self):
        from ctranslate2.converters.transformers import main

        report = self.output.parent / "report.json"
        argv = [
            "ct2-transformers-converter",
            "--model",
            str(self.source),
            "--output_dir",
            str(self.output),
            "--quantization",
            "float32",
            "--report",
            str(report),
        ]
        with patch("sys.argv", argv):
            main()
        values = read_model(self.output / "model.bin")
        np.testing.assert_array_equal(values["backbone/lm_head/weight"], self.sensitive)
        self.assertEqual(
            json.loads(report.read_text()),
            json.loads((self.output / "conversion.json").read_text()),
        )

    def test_transformers_cli_dry_run_writes_report_without_loading(self):
        from ctranslate2.converters.transformers import main

        report = self.output.parent / "report.json"
        argv = [
            "ct2-transformers-converter",
            "--model",
            str(self.source),
            "--output_dir",
            str(self.output),
            "--dry-run",
            "--report",
            str(report),
        ]
        with patch("sys.argv", argv), contextlib.redirect_stdout(
            io.StringIO()
        ), patch.object(c, "load_model_tensors") as load:
            main()
        load.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertTrue(json.loads(report.read_text())["dry_run"])

    def test_transformers_other_models_keep_standard_conversion(self):
        from ctranslate2.converters import TransformersConverter

        (self.source / "config.json").write_text('{"model_type": "llama"}')
        with patch.object(
            Converter, "convert", return_value=str(self.output)
        ) as convert:
            result = TransformersConverter(str(self.source)).convert(
                str(self.output), quantization="float16", force=True
            )
        self.assertEqual(result, str(self.output))
        convert.assert_called_once_with(
            str(self.output), vmap=None, quantization="float16", force=True
        )


class LosslessFP8ConversionTest(ConversionFixture):
    """The standard serializer must preserve source FP8 bytes and block scales."""

    def test_standard_serialization_preserves_original_fp8_bytes_and_scales(self):
        spec, report = c.load_spec(self.source)
        expert = "backbone/layers_0/mlp/experts_0/up_proj/weight"
        self.assertEqual(spec.variables()[expert].dtype, np.int8)
        self.assertEqual(report["quantization"], "float32")
        self.assertEqual(report["fp8_format"], "e4m3fn_block128")
        self.assertEqual(
            report["estimated_loading_bytes"], report["estimated_payload_bytes"]
        )
        self.assertEqual(report["matrix_counts"]["fp8"], 1)

        class LoadedConverter(Converter):
            def _load(self):
                return spec

        LoadedConverter().convert(str(self.output), quantization="float32")
        values = read_model(self.output / "model.bin")
        np.testing.assert_array_equal(
            values[expert].view(np.uint8),
            np.array([[0x38, 0xB8], [0x30, 0x28]], np.uint8),
        )
        np.testing.assert_array_equal(
            values[expert + "_scale_inv"], np.array([[1.0001]], np.float32)
        )
        self.assertNotIn(expert + "_scale", values)
        np.testing.assert_array_equal(values["backbone/lm_head/weight"], self.sensitive)
        np.testing.assert_array_equal(values["codec/decoder/weight"], self.codec)
        package = json.loads((self.output / "config.json").read_text())
        self.assertEqual(package["fp8_format"], "e4m3fn_block128")
        self.assertEqual(package["quantization_policy"], c.FP8_POLICY)
        self.assertEqual(
            sum(value.nbytes for value in values.values()),
            report["estimated_payload_bytes"],
        )

    def test_both_safetensors_e4m3fn_names_preserve_raw_storage(self):
        _, tensors = c.inspect_shards(self.source)
        config = json.loads((self.source / "config.json").read_text())
        for dtype in ("F8_E4M3", "F8_E4M3FN"):
            changed = {key: dict(value) for key, value in tensors.items()}
            changed[self.expert]["dtype"] = dtype
            plan = c.build_plan(changed, config, c.FP8_POLICY)
            expert = next(item for item in plan if item["source"] == self.expert)
            self.assertEqual(expert["dtype"], "fp8")

    def test_transformers_default_uses_lossless_fp8(self):
        converter = TransformersConverter(self.source)
        self.assertEqual(converter.inspect()["quantization_policy"], c.FP8_POLICY)
        converter.convert(str(self.output))
        values = read_model(self.output / "model.bin")
        expert = "backbone/layers_0/mlp/experts_0/up_proj/weight"
        np.testing.assert_array_equal(
            values[expert].view(np.uint8),
            np.array([[0x38, 0xB8], [0x30, 0x28]], np.uint8),
        )
        self.assertIn(expert + "_scale_inv", values)
        self.assertEqual(converter.conversion_report["quantization"], "float32")

    def test_rejects_legacy_policies(self):
        for policy in ("all_matrices_int8", "experts_int8_sensitive_float32"):
            with self.subTest(policy=policy), patch.object(
                c, "load_model_tensors"
            ) as load:
                with self.assertRaisesRegex(ValueError, "policy"):
                    c.load_spec(self.source, quantization_policy=policy)
            load.assert_not_called()

    def test_rejects_unsupported_source_format_before_materialization(self):
        readers, tensors = c.inspect_shards(self.source)
        config = json.loads((self.source / "config.json").read_text())
        for block in ([2, 2], [128, 64]):
            config["quantization_config"]["weight_block_size"] = block
            with self.assertRaisesRegex(ValueError, "128x128"):
                c.build_plan(tensors, config, c.FP8_POLICY)
        config["quantization_config"]["weight_block_size"] = [128, 128]
        for name, dtype in (
            (self.expert, "F8_E5M2"),
            (self.expert + "_scale_inv", "F16"),
        ):
            changed = {key: dict(value) for key, value in tensors.items()}
            changed[name]["dtype"] = dtype
            with self.assertRaisesRegex(ValueError, "E4M3FN|F32"):
                c.build_plan(changed, config, c.FP8_POLICY)

    def test_rejects_nonfinite_bytes_and_invalid_inverse_scales(self):
        readers, tensors = c.inspect_shards(self.source)
        reader = readers[tensors[self.expert]["shard"]]
        weight_offset = tensor_offset(reader.path, self.expert)
        for bad in (0x7F, 0xFF):
            with reader.path.open("r+b") as stream:
                stream.seek(weight_offset)
                stream.write(bytes([bad]))
            with self.assertRaisesRegex(ValueError, "nonfinite E4M3FN"):
                c.load_spec(self.source)
        with reader.path.open("r+b") as stream:
            stream.seek(weight_offset)
            stream.write(bytes([0x38]))
        # Reinspect after each mutation: readers reject changed checkpoint files.
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            readers, tensors = c.inspect_shards(self.source)
            current = readers[tensors[self.expert]["shard"]]
            offset = tensor_offset(current.path, self.expert + "_scale_inv")
            with current.path.open("r+b") as stream:
                stream.seek(offset)
                stream.write(np.array([bad], np.float32).tobytes())
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                c.load_spec(self.source)


if __name__ == "__main__":
    unittest.main()

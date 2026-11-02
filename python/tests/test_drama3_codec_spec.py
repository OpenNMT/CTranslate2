"""The Python codec specification must reject incomplete or incompatible models."""

import unittest

from ctranslate2.converters import drama3_codec as codec
from ctranslate2.specs.drama3_spec import (
    codec_parameter_shapes,
    reference_codec_parameter_shapes,
)


class CodecSpecTest(unittest.TestCase):
    def setUp(self):
        self.schema = {
            name: {"shape": list(shape), "dtype": "float32"}
            for name, shape in codec_parameter_shapes().items()
        }

    def test_every_required_weight_is_checked_for_presence_and_shape(self):
        for name in self.schema:
            with self.subTest(name=name):
                missing = dict(self.schema)
                del missing[name]
                with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                    codec.conversion_plan(missing)
                wrong = dict(self.schema)
                wrong_shape = list(self.schema[name]["shape"])
                wrong_shape[0] += 1
                wrong[name] = {"shape": wrong_shape, "dtype": "float32"}
                with self.assertRaisesRegex(ValueError, "shape"):
                    codec.conversion_plan(wrong)

    def test_unused_encoder_weights_are_excluded(self):
        self.schema["encoder.unused.weight"] = {"shape": [1], "dtype": "float32"}
        plan = codec.conversion_plan(self.schema)
        self.assertEqual(len(plan), 214)
        self.assertFalse(any("encoder" in item["target"] for item in plan))

    def test_weight_normalization_layouts_and_ambiguities(self):
        source = "decoder.model.0.conv.weight"
        shape = self.schema[source]["shape"]
        stem = source.removesuffix(".weight")
        for g, v in (
            (stem + ".weight_g", stem + ".weight_v"),
            (
                stem + ".parametrizations.weight.original0",
                stem + ".parametrizations.weight.original1",
            ),
        ):
            with self.subTest(magnitude=g):
                schema = dict(self.schema)
                del schema[source]
                schema[g] = {"shape": [shape[0], 1, 1], "dtype": "float32"}
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    codec.conversion_plan(schema)
                schema[v] = self.schema[source]
                plan = codec.conversion_plan(schema)
                item = next(item for item in plan if item["source_name"] == source)
                self.assertEqual(item["value_key"], v)
                self.assertEqual(item["magnitude_key"], g)
                schema[source] = self.schema[source]
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    codec.conversion_plan(schema)

    def test_nonfloating_tensors_are_rejected(self):
        name = next(iter(self.schema))
        self.schema[name] = dict(self.schema[name], dtype="int8")
        with self.assertRaisesRegex(ValueError, "non-floating"):
            codec.conversion_plan(self.schema)


class ReferenceCodecSpecTest(unittest.TestCase):
    def test_complete_reference_graph_is_converted(self):
        shapes = codec_parameter_shapes()
        shapes.update(reference_codec_parameter_shapes())
        schema = {
            name: {"shape": list(shape), "dtype": "float32"}
            for name, shape in shapes.items()
        }
        plan = codec.conversion_plan(schema)
        self.assertEqual(len(plan), len(shapes))
        self.assertTrue(
            any(item["target"].startswith("codec/encoder/") for item in plan)
        )
        for name in reference_codec_parameter_shapes():
            with self.subTest(name=name):
                incomplete = dict(schema)
                del incomplete[name]
                with self.assertRaises(ValueError):
                    codec.conversion_plan(incomplete)
